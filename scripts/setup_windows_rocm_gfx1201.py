"""Validate and configure the supported Windows ROCm gfx1201 runtime.

This script is the reproducible replacement for hand-editing site-packages.  FreeToken
handles Triton PDL and Uvicorn's event loop in repository code; only TVM-FFI 0.1.13.post3
requires a deterministic installed-package patch for Windows HIP/C++ JIT compilation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import site
import sys
from pathlib import Path


SUPPORTED_PYTHON = (3, 12)
SUPPORTED_TORCH = "2.11.0+rocm7.13.0"
SUPPORTED_HIP = "7.13.99004"
SUPPORTED_TRITON = "3.7.1"
SUPPORTED_TRITON_WINDOWS = "3.7.1.post27"
SUPPORTED_TVM_FFI = "0.1.13.post3"
SUPPORTED_ROCM = "7.13.0"
SUPPORTED_ROCM_FAMILY = "gfx120X-all"
SUPPORTED_GFX = "gfx1201"
TVM_FFI_PRISTINE_SHA256 = "092e53375e0e91694417b32f8782a164c9154036227142467291da8d5581cdf1"
TVM_FFI_PATCH_MARKER = "FREETOKEN_WINDOWS_ROCM_GFX1201_PATCH_V1"


class SetupError(RuntimeError):
    pass


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise SetupError(f"required package is missing: {name}") from exc


def _find_rocm_sdk() -> Path:
    candidates: list[Path] = []
    for root in site.getsitepackages():
        candidates.append(Path(root) / "_rocm_sdk_core")
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site) / "_rocm_sdk_core")
    found = [path.resolve() for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise SetupError(f"expected exactly one venv ROCm SDK, found: {found}")
    return found[0]


def _cache_root(value: str | None) -> Path:
    root = Path(value).expanduser() if value else Path.home() / "FreeToken-rocm-cache"
    root = root.resolve()
    if " " in str(root):
        raise SetupError(
            f"cache path contains spaces, which is unsafe for the Windows Ninja path: {root}; "
            "pass --cache-root with a space-free path"
        )
    return root


def _probe_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".freetoken-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        if probe.read_text(encoding="utf-8") != "ok":
            raise SetupError(f"cache write verification failed: {path}")
    except OSError as exc:
        raise SetupError(f"cache is not writable: {path}: {exc}") from exc
    finally:
        probe.unlink(missing_ok=True)


def build_environment(sdk: Path, cache_root: Path) -> dict[str, str]:
    clang_root = sdk / "lib" / "llvm" / "bin"
    bitcode = sdk / "lib" / "llvm" / "amdgcn" / "bitcode"
    clang_cl = clang_root / "clang-cl.exe"
    clangxx = clang_root / "clang++.exe"
    hipcc = sdk / "bin" / "hipcc.exe"
    for path in (clang_cl, clangxx, hipcc, bitcode):
        if not path.exists():
            raise SetupError(f"required ROCm toolchain path is missing: {path}")

    triton_cache = cache_root / "triton"
    tvm_cache = cache_root / "tvm-ffi"
    _probe_writable(triton_cache)
    _probe_writable(tvm_cache)
    return {
        "ROCM_SDK_TARGET_FAMILY": SUPPORTED_ROCM_FAMILY,
        "PYTORCH_ROCM_ARCH": SUPPORTED_GFX,
        "TRITON_OVERRIDE_ARCH": SUPPORTED_GFX,
        "TVM_FFI_ROCM_ARCH_LIST": SUPPORTED_GFX,
        "TVM_FFI_GPU_BACKEND": "hip",
        "HIP_PATH": str(sdk),
        "ROCM_HOME": str(sdk),
        "ROCM_PATH": str(sdk),
        "HIP_DEVICE_LIB_PATH": str(bitcode),
        "CC": str(clang_cl),
        "CXX": str(clangxx),
        "TRITON_CACHE_DIR": str(triton_cache),
        "TVM_FFI_CACHE_DIR": str(tvm_cache),
    }


def _loaded_hip_runtime_path() -> Path:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    kernel32.GetModuleFileNameW.restype = ctypes.c_uint
    handle = kernel32.GetModuleHandleW("amdhip64_7.dll")
    if not handle:
        raise SetupError("amdhip64_7.dll is not loaded by the active torch process")
    buffer = ctypes.create_unicode_buffer(32768)
    if not kernel32.GetModuleFileNameW(handle, buffer, len(buffer)):
        raise SetupError("could not resolve the loaded amdhip64_7.dll path")
    return Path(buffer.value).resolve()


def validate_runtime(sdk: Path) -> dict[str, str]:
    if sys.platform != "win32":
        raise SetupError(f"this setup supports Windows only, got {sys.platform}")
    if sys.version_info[:2] != SUPPORTED_PYTHON:
        raise SetupError(
            f"Python {SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]} is required, "
            f"got {sys.version_info.major}.{sys.version_info.minor}"
        )
    if _distribution_version("rocm-sdk-core") != SUPPORTED_ROCM:
        raise SetupError(f"rocm-sdk-core must be {SUPPORTED_ROCM}")
    if _distribution_version("rocm-sdk-devel") != SUPPORTED_ROCM:
        raise SetupError(f"rocm-sdk-devel must be {SUPPORTED_ROCM}")
    family_dist = f"rocm-sdk-libraries-{SUPPORTED_ROCM_FAMILY}"
    if _distribution_version(family_dist) != SUPPORTED_ROCM:
        raise SetupError(f"{family_dist} must be {SUPPORTED_ROCM}")
    if _distribution_version("triton-windows") != SUPPORTED_TRITON_WINDOWS:
        raise SetupError(f"triton-windows must be {SUPPORTED_TRITON_WINDOWS}")
    if _distribution_version("apache-tvm-ffi") != SUPPORTED_TVM_FFI:
        raise SetupError(f"apache-tvm-ffi must be {SUPPORTED_TVM_FFI}")

    import torch
    import triton

    if torch.__version__ != SUPPORTED_TORCH:
        raise SetupError(f"torch must be {SUPPORTED_TORCH}, got {torch.__version__}")
    if str(torch.version.hip) != SUPPORTED_HIP:
        raise SetupError(f"torch HIP must be {SUPPORTED_HIP}, got {torch.version.hip}")
    if triton.__version__ != SUPPORTED_TRITON:
        raise SetupError(f"triton import must report {SUPPORTED_TRITON}, got {triton.__version__}")
    if not torch.cuda.is_available():
        raise SetupError("ROCm GPU is not available")
    arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0]
    if arch.lower() != SUPPORTED_GFX:
        raise SetupError(f"expected {SUPPORTED_GFX}, got {arch}")

    loaded_runtime = _loaded_hip_runtime_path()
    expected_runtime = (sdk / "bin" / "amdhip64_7.dll").resolve()
    if loaded_runtime != expected_runtime:
        raise SetupError(
            "mixed ROCm runtimes detected: "
            f"torch loaded {loaded_runtime}, configured SDK provides {expected_runtime}"
        )
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip": str(torch.version.hip),
        "triton": triton.__version__,
        "triton_windows": _distribution_version("triton-windows"),
        "tvm_ffi": _distribution_version("apache-tvm-ffi"),
        "device": torch.cuda.get_device_name(0),
        "gfx": arch,
        "hip_runtime": str(loaded_runtime),
    }


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SetupError(f"TVM-FFI {label} anchor count must be 1, got {count}")
    return source.replace(old, new, 1)


def patch_tvm_ffi_source(source: str) -> str:
    if TVM_FFI_PATCH_MARKER in source:
        return source
    source = _replace_once(
        source,
        "    tvm_ffi_lib_name = tvm_ffi_lib.stem\n",
        "    tvm_ffi_lib_name = tvm_ffi_lib.stem\n"
        f"    # {TVM_FFI_PATCH_MARKER}\n"
        "    # HIP_PATH is set for the whole supported Windows ROCm process, including\n"
        "    # C++-only AOT modules such as radix.cpp that have no .cu source.\n"
        "    _use_clang_host = IS_WINDOWS and bool(os.environ.get(\"HIP_PATH\"))\n",
        "host-mode",
    )
    source = _replace_once(
        source,
        '        default_cuda_cflags = ["-Xcompiler", "/std:c++17", "/O2"]',
        "        if with_hip:\n"
        '            default_cuda_cflags = ["-O2", "-D__HIP_PLATFORM_AMD__=1"]\n'
        "            default_cuda_cflags += _get_rocm_target()\n"
        "        else:\n"
        '            default_cuda_cflags = ["-Xcompiler", "/std:c++17", "/O2"]',
        "HIP flags",
    )
    source = _replace_once(
        source,
        "    extra_cflags_list = [flag.strip() for flag in extra_cflags]\n"
        "    cflags = default_cflags + extra_cflags_list\n"
        "    cxxflags = default_cxxflags + extra_cflags_list",
        "    extra_cflags_list = [flag.strip() for flag in extra_cflags]\n"
        "    if _use_clang_host:\n"
        '        default_cxxflags = ["-O2", "-D__HIP_PLATFORM_AMD__=1"]\n'
        '        default_cflags = ["-O2"]\n'
        "        rocm_home = _find_rocm_home()\n"
        "        default_ldflags = [\n"
        '            "-L" + tvm_ffi_lib_path.replace("\\\\", "/"),\n'
        '            "-ltvm_ffi",\n'
        '            "-L" + str(Path(rocm_home) / "lib"),\n'
        '            "-lamdhip64",\n'
        "        ]\n"
        "    cflags = default_cflags + extra_cflags_list\n"
        "    cxxflags = default_cxxflags + extra_cflags_list",
        "host flags",
    )
    source = _replace_once(
        source,
        '    ninja.append("cxx = {}".format(os.environ.get("CXX", "cl" if IS_WINDOWS else "c++")))',
        "    if _use_clang_host:\n"
        '        clangxx = str(Path(_find_rocm_home()) / "lib" / "llvm" / "bin" / "clang++.exe")\n'
        '        ninja.append("cxx = {}".format(os.environ.get("CXX", clangxx)))\n'
        "    else:\n"
        '        ninja.append("cxx = {}".format(os.environ.get("CXX", "cl" if IS_WINDOWS else "c++")))',
        "CXX",
    )
    source = _replace_once(
        source,
        "    if IS_WINDOWS:\n"
        '        ninja.append("  command = $cxx /showIncludes $cxxflags -c $in /Fo$out")\n'
        '        ninja.append("  deps = msvc")',
        "    if IS_WINDOWS:\n"
        "        if _use_clang_host:\n"
        '            ninja.append("  depfile = $out.d")\n'
        '            ninja.append("  deps = gcc")\n'
        '            ninja.append("  command = $cxx -MMD -MF $out.d $cxxflags -c $in -o $out")\n'
        "        else:\n"
        '            ninja.append("  command = $cxx /showIncludes $cxxflags -c $in /Fo$out")\n'
        '            ninja.append("  deps = msvc")',
        "compile rule",
    )
    source = _replace_once(
        source,
        "        if IS_WINDOWS:\n"
        '            ninja.append("  command = $cxx $in /link $ldflags /out:$out")',
        "        if _use_clang_host:\n"
        '            ninja.append("  command = $cxx -shared $in $ldflags -o $out")\n'
        "        elif IS_WINDOWS:\n"
        '            ninja.append("  command = $cxx $in /link $ldflags /out:$out")',
        "link rule",
    )
    compile(source, "tvm_ffi_extension_patched.py", "exec")
    return source


def patch_tvm_ffi() -> dict[str, str]:
    import tvm_ffi.cpp.extension as extension

    path = Path(extension.__file__).resolve()
    state_path = path.with_suffix(path.suffix + ".freetoken-gfx1201.json")
    raw_source = path.read_bytes()
    source = raw_source.decode("utf-8").replace("\r\n", "\n")
    current_hash = _sha256_bytes(raw_source)
    if TVM_FFI_PATCH_MARKER in source:
        if not state_path.is_file():
            raise SetupError(f"patched TVM-FFI has no setup manifest: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("patched_sha256") != current_hash:
            raise SetupError("TVM-FFI changed after the reproducible patch was applied")
        return state
    if current_hash != TVM_FFI_PRISTINE_SHA256:
        raise SetupError(
            "unsupported or manually modified TVM-FFI source: "
            f"expected {TVM_FFI_PRISTINE_SHA256}, got {current_hash}"
        )
    patched = patch_tvm_ffi_source(source)
    state = {
        "package": f"apache-tvm-ffi=={SUPPORTED_TVM_FFI}",
        "path": str(path),
        "original_sha256": current_hash,
        "patched_sha256": _sha256(patched),
        "patch_marker": TVM_FFI_PATCH_MARKER,
    }
    path.write_bytes(patched.encode("utf-8"))
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def write_activation_script(environment: dict[str, str]) -> Path:
    scripts = Path(sys.prefix) / "Scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    path = scripts / "freetoken-rocm-gfx1201.ps1"
    lines = [
        "# Generated by scripts/setup_windows_rocm_gfx1201.py",
        "$ErrorActionPreference = 'Stop'",
    ]
    for key, value in environment.items():
        lines.append(f"$env:{key} = '{value.replace("'", "''")}'")
    sdk = Path(environment["HIP_PATH"])
    tool_paths = f"{sdk / 'bin'};{sdk / 'lib' / 'llvm' / 'bin'}"
    lines.append(f"$env:PATH = '{tool_paths}' + ';' + $env:PATH")
    lines.append("Write-Host 'FreeToken Windows ROCm gfx1201 environment configured.'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", help="space-free external cache root")
    parser.add_argument("--check-only", action="store_true", help="validate without patching")
    args = parser.parse_args(argv)
    try:
        sdk = _find_rocm_sdk()
        cache_root = _cache_root(args.cache_root)
        environment = build_environment(sdk, cache_root)
        os.environ.update(environment)
        runtime = validate_runtime(sdk)
        patch_state = None if args.check_only else patch_tvm_ffi()
        activation = None if args.check_only else write_activation_script(environment)
    except SetupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("FreeToken Windows ROCm gfx1201 validation: PASS")
    for key, value in runtime.items():
        print(f"{key}={value}")
    for key, value in environment.items():
        print(f"{key}={value}")
    if patch_state:
        print(f"TVM_FFI_PATCHED_SHA256={patch_state['patched_sha256']}")
    if activation:
        print(f"ACTIVATION_SCRIPT={activation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
