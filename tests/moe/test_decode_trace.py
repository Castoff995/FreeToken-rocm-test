import io
import os
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest import mock

import torch

from freetoken.kernel.pinned import HostMappingResult
from freetoken.moe import decode_trace


def _batch(phase: str):
    return SimpleNamespace(phase=phase, is_decode=phase == "decode", size=1)


def _host_cache(*sources: torch.Tensor):
    names = tuple(f"bank_{index}" for index in range(len(sources)))
    return SimpleNamespace(
        bank_schema=names,
        bank_sources={name: [source] for name, source in zip(names, sources)},
        layer_residency=["pinned"],
    )


def _mapped(host_ptr: int) -> HostMappingResult:
    return HostMappingResult(host_ptr + 0x1000, "hip_runtime", False, "ok")


class DecodeTraceTests(unittest.TestCase):
    def setUp(self):
        decode_trace._reset_for_tests()

    def tearDown(self):
        decode_trace._reset_for_tests()

    def test_trace_mode_is_inert_when_disabled(self):
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {decode_trace.TRACE_ENV: ""}),
            redirect_stderr(output),
        ):
            self.assertFalse(decode_trace.begin_first_decode(_batch("decode")))
            decode_trace.trace_stage("must_not_print")
            decode_trace.synchronize("must_not_sync")
        self.assertEqual(output.getvalue(), "")

        # A disabled call does not consume the one-shot claim.
        with (
            mock.patch.dict(os.environ, {decode_trace.TRACE_ENV: "1"}),
            mock.patch.object(
                decode_trace, "_stream_fields", return_value={"stream": "test"}
            ),
            redirect_stderr(io.StringIO()),
        ):
            self.assertTrue(decode_trace.begin_first_decode(_batch("decode")))

    def test_trace_claims_only_first_real_decode(self):
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {decode_trace.TRACE_ENV: "1"}),
            mock.patch.object(
                decode_trace, "_stream_fields", return_value={"stream": "test"}
            ),
            redirect_stderr(output),
        ):
            self.assertFalse(decode_trace.begin_first_decode(_batch("prefill")))
            self.assertTrue(decode_trace.begin_first_decode(_batch("decode")))
            decode_trace.finish_trace()
            self.assertFalse(decode_trace.begin_first_decode(_batch("decode")))

        rendered = output.getvalue()
        self.assertEqual(rendered.count("stage=decode_entry"), 1)
        self.assertEqual(rendered.count("stage=decode_trace_complete"), 1)

    def test_invalid_slot_ids_are_rejected(self):
        cases = [
            (torch.tensor([[0, -1]], dtype=torch.int32), ValueError, "out of range"),
            (torch.tensor([[0, 6]], dtype=torch.int32), ValueError, "out of range"),
            (torch.tensor([[0, 1]], dtype=torch.int64), TypeError, "torch.int32"),
            (
                torch.arange(4, dtype=torch.int32).view(2, 2).t(),
                ValueError,
                "contiguous",
            ),
        ]
        for slot_ids, error, message in cases:
            with self.subTest(slot_ids=slot_ids, error=error):
                with self.assertRaisesRegex(error, message):
                    decode_trace._validate_slot_ids(slot_ids, num_cache_slots=6)

    def test_valid_slot_ids_and_populated_owners_pass(self):
        raw = torch.tensor([[2, 1, 2]], dtype=torch.int32)
        slots = torch.tensor([[5, 0, 5]], dtype=torch.int32)
        owners = torch.full((6,), -1, dtype=torch.int32)
        owners[0] = 1
        owners[5] = 2

        self.assertEqual(decode_trace._validate_slot_ids(slots, 6), (0, 5))
        self.assertEqual(
            decode_trace.validate_remapped_slots(
                raw,
                slots,
                owners,
                layer_id=0,
                num_experts=4,
                num_cache_slots=6,
            ),
            (0, 5, 2),
        )

    def test_stale_or_wrong_layer_slot_owner_is_rejected(self):
        raw = torch.tensor([[2, 1]], dtype=torch.int32)
        slots = torch.tensor([[5, 0]], dtype=torch.int32)
        owners = torch.full((6,), -1, dtype=torch.int32)
        owners[0] = 1
        owners[5] = 6  # layer 1 / expert 2, not layer 0 / expert 2

        with self.assertRaisesRegex(ValueError, "not populated by the routed expert"):
            decode_trace.validate_remapped_slots(
                raw,
                slots,
                owners,
                layer_id=0,
                num_experts=4,
                num_cache_slots=6,
            )

    def test_disabled_debug_path_never_synchronizes_nvidia_or_rocm(self):
        calls = []
        with (
            mock.patch.dict(os.environ, {decode_trace.TRACE_ENV: ""}),
            mock.patch.object(torch.cuda, "synchronize", side_effect=calls.append),
        ):
            decode_trace.synchronize("disabled", torch.device("cuda"))

        self.assertEqual(calls, [])

    def test_host_mapping_preflight_backend_unavailable_is_unsafe(self):
        cache = _host_cache(torch.empty(4, dtype=torch.uint8))
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {decode_trace.TRACE_ENV: "1"}),
            mock.patch.object(decode_trace, "_stream_fields", return_value={"stream": "test"}),
            mock.patch.object(decode_trace.sys, "platform", "win32"),
            mock.patch.object(torch.version, "hip", "test-rocm"),
            mock.patch(
                "freetoken.kernel.pinned.resolve_host_mapping",
                return_value=HostMappingResult(
                    None, "unavailable", False, "mapping_backend_unavailable"
                ),
            ),
            redirect_stderr(output),
        ):
            self.assertTrue(decode_trace.begin_first_decode(_batch("decode")))
            with self.assertRaisesRegex(RuntimeError, "mapping_backend_unavailable"):
                decode_trace.preflight_windows_rocm_host_mapping(cache, 0)

        marker = next(
            line for line in output.getvalue().splitlines() if "stage=host_mapping_preflight" in line
        )
        self.assertIn("pinned_extension_loaded=false", marker)
        self.assertIn("mapping_backend=unavailable", marker)
        self.assertIn("device_ptr=unavailable", marker)
        self.assertIn("safe_for_gpu_deref=false", marker)
        self.assertIn("reason=mapping_backend_unavailable", marker)

    def test_host_mapping_preflight_unavailable_mapping_is_unsafe(self):
        cache = _host_cache(torch.empty(4, dtype=torch.uint8))
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {decode_trace.TRACE_ENV: "1"}),
            mock.patch.object(decode_trace, "_stream_fields", return_value={"stream": "test"}),
            mock.patch.object(decode_trace.sys, "platform", "win32"),
            mock.patch.object(torch.version, "hip", "test-rocm"),
            mock.patch(
                "freetoken.kernel.pinned.resolve_host_mapping",
                return_value=HostMappingResult(
                    None, "unavailable", True, "host_device_mapping_failed"
                ),
            ),
            redirect_stderr(output),
        ):
            self.assertTrue(decode_trace.begin_first_decode(_batch("decode")))
            with self.assertRaisesRegex(RuntimeError, "host_device_mapping_failed"):
                decode_trace.preflight_windows_rocm_host_mapping(cache, 0)

        marker = next(
            line for line in output.getvalue().splitlines() if "stage=host_mapping_preflight" in line
        )
        self.assertIn("pinned_extension_loaded=true", marker)
        self.assertIn("mapped_bank_count=0", marker)
        self.assertIn("safe_for_gpu_deref=false", marker)
        self.assertIn("reason=host_device_mapping_failed", marker)

    def test_host_mapping_preflight_valid_mapped_sources_are_safe(self):
        sources = (
            torch.empty(4, dtype=torch.uint8),
            torch.empty(8, dtype=torch.uint8),
        )
        cache = _host_cache(*sources)
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {decode_trace.TRACE_ENV: "1"}),
            mock.patch.object(decode_trace, "_stream_fields", return_value={"stream": "test"}),
            mock.patch.object(decode_trace.sys, "platform", "win32"),
            mock.patch.object(torch.version, "hip", "test-rocm"),
            mock.patch(
                "freetoken.kernel.pinned.resolve_host_mapping",
                side_effect=_mapped,
            ),
            redirect_stderr(output),
        ):
            self.assertTrue(decode_trace.begin_first_decode(_batch("decode")))
            self.assertTrue(decode_trace.preflight_windows_rocm_host_mapping(cache, 0))

        marker = next(
            line for line in output.getvalue().splitlines() if "stage=host_mapping_preflight" in line
        )
        self.assertIn("bank_count=2", marker)
        self.assertIn("mapped_bank_count=2", marker)
        self.assertIn("mapping_backend=hip_runtime", marker)
        self.assertIn(f"host_ptr={hex(sources[0].data_ptr())}", marker)
        self.assertIn(f"device_ptr={hex(sources[0].data_ptr() + 0x1000)}", marker)
        self.assertIn("safe_for_gpu_deref=true", marker)
        self.assertIn("reason=ok", marker)

    def test_host_mapping_preflight_is_inert_when_trace_disabled(self):
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {decode_trace.TRACE_ENV: ""}),
            mock.patch.object(decode_trace.sys, "platform", "win32"),
            mock.patch.object(torch.version, "hip", "test-rocm"),
            mock.patch(
                "freetoken.kernel.pinned.resolve_host_mapping",
                side_effect=AssertionError("mapping resolver must remain untouched"),
            ) as resolver,
            redirect_stderr(output),
        ):
            self.assertFalse(
                decode_trace.preflight_windows_rocm_host_mapping(object(), layer_id=0)
            )

        resolver.assert_not_called()
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
