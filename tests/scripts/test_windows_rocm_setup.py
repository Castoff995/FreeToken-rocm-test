from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "setup_windows_rocm_gfx1201.py"
SPEC = importlib.util.spec_from_file_location("setup_windows_rocm_gfx1201", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def _pristine_tvm_fixture() -> str:
    return '''\
def generate():
    tvm_ffi_lib_name = tvm_ffi_lib.stem
    if IS_WINDOWS:
        default_cuda_cflags = ["-Xcompiler", "/std:c++17", "/O2"]
    extra_cflags_list = [flag.strip() for flag in extra_cflags]
    cflags = default_cflags + extra_cflags_list
    cxxflags = default_cxxflags + extra_cflags_list
    ninja = []
    ninja.append("cxx = {}".format(os.environ.get("CXX", "cl" if IS_WINDOWS else "c++")))
    ninja.append("rule compile")
    if IS_WINDOWS:
        ninja.append("  command = $cxx /showIncludes $cxxflags -c $in /Fo$out")
        ninja.append("  deps = msvc")
    if not object_mode:
        ninja.append("rule link")
        if IS_WINDOWS:
            ninja.append("  command = $cxx $in /link $ldflags /out:$out")
'''


def test_tvm_patch_is_deterministic_and_covers_cpp_only_aot():
    patched = setup.patch_tvm_ffi_source(_pristine_tvm_fixture())
    assert setup.TVM_FFI_PATCH_MARKER in patched
    assert 'bool(os.environ.get("HIP_PATH"))' in patched
    assert "--offload-arch" not in patched  # _get_rocm_target emits it at runtime.
    assert "default_cuda_cflags += _get_rocm_target()" in patched
    assert "$cxx -MMD -MF $out.d" in patched
    assert "$cxx -shared $in" in patched
    assert setup.patch_tvm_ffi_source(patched) == patched


def test_tvm_patch_rejects_unknown_source():
    with pytest.raises(setup.SetupError, match="anchor count"):
        setup.patch_tvm_ffi_source("unknown source")


def _fake_sdk(tmp_path: Path) -> Path:
    sdk = tmp_path / "_rocm_sdk_core"
    for relative in (
        "lib/llvm/bin/clang-cl.exe",
        "lib/llvm/bin/clang++.exe",
        "bin/hipcc.exe",
        "bin/amdhip64_7.dll",
    ):
        path = sdk / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (sdk / "lib/llvm/amdgcn/bitcode").mkdir(parents=True)
    return sdk


def test_build_environment_uses_one_sdk_and_external_caches(tmp_path):
    sdk = _fake_sdk(tmp_path)
    env = setup.build_environment(sdk, tmp_path / "cache")
    assert env["HIP_PATH"] == env["ROCM_HOME"] == env["ROCM_PATH"] == str(sdk)
    assert env["ROCM_SDK_TARGET_FAMILY"] == "gfx120X-all"
    assert env["PYTORCH_ROCM_ARCH"] == "gfx1201"
    assert env["TRITON_OVERRIDE_ARCH"] == "gfx1201"
    assert env["TVM_FFI_ROCM_ARCH_LIST"] == "gfx1201"
    assert Path(env["TRITON_CACHE_DIR"]).is_dir()
    assert Path(env["TVM_FFI_CACHE_DIR"]).is_dir()


def test_cache_root_rejects_space_unsafe_path(tmp_path):
    with pytest.raises(setup.SetupError, match="contains spaces"):
        setup._cache_root(str(tmp_path / "bad cache"))


def test_runtime_validation_detects_gfx1201_and_same_sdk(monkeypatch, tmp_path):
    sdk = _fake_sdk(tmp_path)
    versions = {
        "rocm-sdk-core": "7.13.0",
        "rocm-sdk-devel": "7.13.0",
        "rocm-sdk-libraries-gfx120X-all": "7.13.0",
        "triton-windows": "3.7.1.post27",
        "apache-tvm-ffi": "0.1.13.post3",
    }
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _: SimpleNamespace(gcnArchName="gfx1201"),
        get_device_name=lambda _: "AMD Radeon RX 9070 XT",
    )
    fake_torch = SimpleNamespace(
        __version__="2.11.0+rocm7.13.0",
        version=SimpleNamespace(hip="7.13.99004"),
        cuda=cuda,
    )
    fake_triton = SimpleNamespace(__version__="3.7.1")
    monkeypatch.setattr(setup.sys, "platform", "win32")
    monkeypatch.setattr(setup, "_distribution_version", versions.__getitem__)
    monkeypatch.setattr(setup, "_loaded_hip_runtime_path", lambda: sdk / "bin/amdhip64_7.dll")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    result = setup.validate_runtime(sdk)
    assert result["gfx"] == "gfx1201"
    assert result["hip"] == "7.13.99004"


def test_runtime_validation_rejects_wrong_gfx(monkeypatch, tmp_path):
    sdk = _fake_sdk(tmp_path)
    monkeypatch.setattr(setup.sys, "platform", "win32")
    monkeypatch.setattr(setup, "_distribution_version", lambda _: {
        "rocm-sdk-core": "7.13.0",
        "rocm-sdk-devel": "7.13.0",
        "rocm-sdk-libraries-gfx120X-all": "7.13.0",
        "triton-windows": "3.7.1.post27",
        "apache-tvm-ffi": "0.1.13.post3",
    }[_])
    fake_torch = SimpleNamespace(
        __version__="2.11.0+rocm7.13.0",
        version=SimpleNamespace(hip="7.13.99004"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(gcnArchName="gfx1200"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="3.7.1"))
    with pytest.raises(setup.SetupError, match="expected gfx1201"):
        setup.validate_runtime(sdk)
