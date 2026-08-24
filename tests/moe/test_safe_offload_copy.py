import io
import os
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest import mock

import torch

from freetoken.kernel.pinned import HostMappingResult
from freetoken.moe import decode_trace
from freetoken.moe.offload_cache import SAFE_OFFLOAD_COPY_ENV, OffloadMoeCache


def _mapped(host_ptr: int) -> HostMappingResult:
    return HostMappingResult(host_ptr + 0x1000, "hip_runtime", False, "ok")


def _cache() -> OffloadMoeCache:
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
    )
    gate_up = torch.stack(
        [torch.full((2, 3), expert + 1, dtype=torch.float32) for expert in range(4)]
    )
    down = torch.stack(
        [torch.full((3, 2), 10 + expert, dtype=torch.float32) for expert in range(4)]
    )
    # Building a CPU fixture must never discover or call an installed HIP runtime.
    with mock.patch.object(torch.version, "hip", None):
        cache.set_bank_sources({"gate_up": [gate_up], "down": [down]})
    cache._pending_src_layer = 0
    return cache


def _batch():
    return SimpleNamespace(phase="decode", is_decode=True, size=1)


class SafeOffloadCopyTests(unittest.TestCase):
    def setUp(self):
        decode_trace._reset_for_tests()

    def tearDown(self):
        decode_trace._reset_for_tests()

    def test_unmapped_windows_rocm_selects_fallback_automatically(self):
        cache = _cache()
        with (
            mock.patch.dict(os.environ, {SAFE_OFFLOAD_COPY_ENV: ""}),
            mock.patch("freetoken.moe.offload_cache.sys.platform", "win32"),
            mock.patch.object(torch.version, "hip", "test-rocm"),
            mock.patch(
                "freetoken.kernel.pinned.resolve_host_mapping",
                return_value=HostMappingResult(
                    None, "unavailable", False, "mapping_backend_unavailable"
                ),
            ),
        ):
            self.assertTrue(cache.should_use_safe_offload_copy(0))

    def test_unloadable_mapping_extension_fails_closed_to_safe_copy(self):
        cache = _cache()
        with (
            mock.patch.dict(os.environ, {SAFE_OFFLOAD_COPY_ENV: ""}),
            mock.patch("freetoken.moe.offload_cache.sys.platform", "win32"),
            mock.patch.object(torch.version, "hip", "test-rocm"),
            mock.patch(
                "freetoken.kernel.pinned.resolve_host_mapping",
                side_effect=OSError("mapping backend unavailable"),
            ),
        ):
            self.assertTrue(cache.should_use_safe_offload_copy(0))

    def test_mapped_windows_rocm_keeps_existing_fast_path(self):
        cache = _cache()
        with (
            mock.patch.dict(os.environ, {SAFE_OFFLOAD_COPY_ENV: ""}),
            mock.patch("freetoken.moe.offload_cache.sys.platform", "win32"),
            mock.patch.object(torch.version, "hip", "test-rocm"),
            mock.patch(
                "freetoken.kernel.pinned.resolve_host_mapping",
                side_effect=_mapped,
            ),
            mock.patch.object(cache, "_copy_missing_safe") as safe_copy,
            mock.patch("freetoken.kernel.fast_index_copy_jit") as fast_copy,
        ):
            self.assertFalse(cache.should_use_safe_offload_copy(0))
            cache.copy_missing()

        safe_copy.assert_not_called()
        self.assertEqual(fast_copy.call_count, len(cache.banks))

    def test_force_override_selects_safe_copy_for_mapped_windows_rocm(self):
        cache = _cache()
        with (
            mock.patch.dict(os.environ, {SAFE_OFFLOAD_COPY_ENV: "1"}),
            mock.patch("freetoken.moe.offload_cache.sys.platform", "win32"),
            mock.patch.object(torch.version, "hip", "test-rocm"),
            mock.patch(
                "freetoken.kernel.pinned.resolve_host_mapping",
                side_effect=AssertionError("force override must not inspect mapping"),
            ) as resolver,
        ):
            self.assertTrue(cache.should_use_safe_offload_copy(0))
        resolver.assert_not_called()

    def test_nvidia_keeps_existing_fast_path(self):
        cache = _cache()
        with (
            mock.patch.dict(os.environ, {SAFE_OFFLOAD_COPY_ENV: "1"}),
            mock.patch("freetoken.moe.offload_cache.sys.platform", "win32"),
            mock.patch.object(torch.version, "hip", None),
        ):
            self.assertFalse(cache.should_use_safe_offload_copy(0))

    def test_copy_missing_bypasses_fast_kernels_for_unsafe_windows_rocm(self):
        cache = _cache()
        cache._copy_fused_ok = True
        with (
            mock.patch.dict(os.environ, {SAFE_OFFLOAD_COPY_ENV: ""}),
            mock.patch("freetoken.moe.offload_cache.sys.platform", "win32"),
            mock.patch.object(torch.version, "hip", "test-rocm"),
            mock.patch(
                "freetoken.kernel.pinned.resolve_host_mapping",
                return_value=HostMappingResult(
                    None, "unavailable", False, "mapping_backend_unavailable"
                ),
            ),
            mock.patch.object(cache, "_copy_missing_safe") as safe_copy,
        ):
            cache.copy_missing()

        safe_copy.assert_called_once_with(0)

    def test_copy_missing_keeps_nvidia_per_bank_path(self):
        cache = _cache()
        with (
            mock.patch.dict(os.environ, {SAFE_OFFLOAD_COPY_ENV: "1"}),
            mock.patch("freetoken.moe.offload_cache.sys.platform", "win32"),
            mock.patch.object(torch.version, "hip", None),
            mock.patch.object(cache, "_copy_missing_safe") as safe_copy,
            mock.patch("freetoken.kernel.fast_index_copy_jit") as fast_copy,
        ):
            cache.copy_missing()

        safe_copy.assert_not_called()
        self.assertEqual(fast_copy.call_count, len(cache.banks))

    def test_mapping_result_is_cached_per_layer(self):
        cache = _cache()
        with (
            mock.patch.dict(os.environ, {SAFE_OFFLOAD_COPY_ENV: ""}),
            mock.patch("freetoken.moe.offload_cache.sys.platform", "win32"),
            mock.patch.object(torch.version, "hip", "test-rocm"),
            mock.patch(
                "freetoken.kernel.pinned.resolve_host_mapping",
                side_effect=_mapped,
            ) as resolver,
        ):
            self.assertFalse(cache.should_use_safe_offload_copy(0))
            self.assertFalse(cache.should_use_safe_offload_copy(0))

        self.assertEqual(resolver.call_count, len(cache.bank_schema))

    def test_safe_copy_plan_preserves_staged_lru_pairs(self):
        cache = _cache()
        cache.num_indices.fill_(3)
        cache.evict_slots[:3] = torch.tensor([5, 1, 4], dtype=torch.int32)
        cache.src_indices[:3] = torch.tensor([2, 0, 3], dtype=torch.int32)

        self.assertEqual(cache._safe_copy_plan(), ((5, 2), (1, 0), (4, 3)))

    def test_safe_copy_preserves_lru_slot_and_remap_semantics(self):
        cache = _cache()
        cache.num_indices.fill_(2)
        cache.evict_slots[:2] = torch.tensor([4, 1], dtype=torch.int32)
        cache.src_indices[:2] = torch.tensor([3, 0], dtype=torch.int32)
        cache.slot_for_id[0, 3] = 4
        cache.slot_for_id[0, 0] = 1
        cache.id_of_slot[4] = 3
        cache.id_of_slot[1] = 0
        slot_for_id_before = cache.slot_for_id.clone()
        id_of_slot_before = cache.id_of_slot.clone()
        for _, destination in cache.banks:
            destination.zero_()

        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {decode_trace.TRACE_ENV: "1"}),
            mock.patch.object(decode_trace, "_stream_fields", return_value={"stream": "test"}),
            mock.patch.object(torch.cuda, "synchronize") as synchronize,
            redirect_stderr(output),
        ):
            self.assertTrue(decode_trace.begin_first_decode(_batch()))
            cache._copy_missing_safe(0)

        synchronize.assert_called_once_with(cache.device)
        torch.testing.assert_close(cache.slot_for_id, slot_for_id_before)
        torch.testing.assert_close(cache.id_of_slot, id_of_slot_before)
        for per_layer, destination in cache.banks:
            source = per_layer[0]
            torch.testing.assert_close(destination[4], source[3])
            torch.testing.assert_close(destination[1], source[0])
            self.assertEqual(int(torch.count_nonzero(destination[[0, 2, 3, 5]])), 0)

        rendered = output.getvalue()
        self.assertIn("stage=safe_copy_start", rendered)
        self.assertEqual(rendered.count("stage=safe_copy_bank_start"), 2)
        self.assertEqual(rendered.count("stage=safe_copy_bank_complete"), 2)
        self.assertIn("stage=safe_copy_sync_complete", rendered)

    def test_tracing_disabled_has_no_markers_or_debug_synchronization(self):
        cache = _cache()
        cache.num_indices.fill_(1)
        cache.evict_slots[0] = 2
        cache.src_indices[0] = 1
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {decode_trace.TRACE_ENV: ""}),
            mock.patch.object(decode_trace, "trace_stage") as trace_stage,
            mock.patch.object(decode_trace, "synchronize") as trace_synchronize,
            mock.patch.object(torch.cuda, "synchronize") as cuda_synchronize,
            redirect_stderr(output),
        ):
            cache._copy_missing_safe(0)

        trace_stage.assert_not_called()
        trace_synchronize.assert_not_called()
        cuda_synchronize.assert_not_called()
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
