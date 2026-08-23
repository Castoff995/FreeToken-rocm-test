"""Apply the three upstream site-packages patches required by the Windows ROCm port.

Idempotent: safe to run repeatedly. Run with the SAME python that runs FreeToken:
    python dist\\patch_upstream.py
"""
import sys
from pathlib import Path


def patch_tvm_ffi() -> bool:
    import tvm_ffi.cpp.extension as ext

    f = Path(ext.__file__)
    src = f.read_text(encoding="utf-8")
    if "patched" in src and "_use_clang_host" in src:
        print(f"[ok] tvm_ffi already patched ({f})")
        return True

    # 1. HIP-aware cuda_cflags on Windows + rocm target + link libs + clang host
    old = '        default_cuda_cflags = ["-Xcompiler", "/std:c++17", "/O2"]'
    new = (
        "        if with_hip:\n"
        "            # patched: hipcc/clang host flags (MSVC-style args are rejected)\n"
        '            default_cuda_cflags = ["-O2", "-D__HIP_PLATFORM_AMD__=1"]\n'
        "        else:\n"
        '            default_cuda_cflags = ["-Xcompiler", "/std:c++17", "/O2"]'
    )
    assert src.count(old) == 1, "tvm_ffi: cuda_cflags anchor missing"
    src = src.replace(old, new)

    old = """        default_ldflags = [
            "/DLL",
            f"/LIBPATH:{tvm_ffi_lib_path}",
            f"{tvm_ffi_lib_name}.lib",
        ]"""
    new = """        default_ldflags = [
            "/DLL",
            f"/LIBPATH:{tvm_ffi_lib_path}",
            f"{tvm_ffi_lib_name}.lib",
        ]
        if with_hip:
            # patched: HIP runtime + full static CRT + system libs for lld-link
            import os as _os

            _rocm = _os.environ.get("HIP_PATH", "")
            if _rocm:
                default_ldflags += [f"/LIBPATH:{Path(_rocm) / 'lib'}", "amdhip64.lib"]
            default_ldflags += [
                "libcmt.lib", "libucrt.lib", "libvcruntime.lib", "libcpmt.lib",
                "kernel32.lib", "user32.lib", "shell32.lib", "advapi32.lib",
            ]"""
    assert src.count(old) == 1, "tvm_ffi: ldflags anchor missing"
    src = src.replace(old, new)

    # 2. gnu-style flags when building host C++ with clang
    anchor = "    tvm_ffi_lib_name = tvm_ffi_lib.stem\n"
    assert src.count(anchor) == 1, "tvm_ffi: lib name anchor missing"
    src = src.replace(
        anchor,
        anchor
        + "    # patched: on Windows+ROCm, host compiles use HIP clang++ (MSVC rejects some idioms)\n"
        + "    _use_clang_host = IS_WINDOWS and bool(__import__('os').environ.get('HIP_PATH'))\n",
        1,
    )
    old = "    extra_cflags_list = [flag.strip() for flag in extra_cflags]\n    cflags = default_cflags + extra_cflags_list\n    cxxflags = default_cxxflags + extra_cflags_list"
    new = """    extra_cflags_list = [flag.strip() for flag in extra_cflags]
    if _use_clang_host:
        # patched: clang++ host flags (MSVC-style flags are rejected by clang)
        default_cxxflags = ["-O2", "-D__HIP_PLATFORM_AMD__=1"]
        default_cflags = ["-O2"]
        default_ldflags = ["-L" + tvm_ffi_lib_path.replace("\\\\", "/"), "-ltvm_ffi"]
        import os as _os

        _rocm_lib = Path(_os.environ.get("HIP_PATH", "")) / "lib"
        if str(_rocm_lib) != "lib":
            default_ldflags += ["-L" + str(_rocm_lib), "-lamdhip64"]
    cflags = default_cflags + extra_cflags_list
    cxxflags = default_cxxflags + extra_cflags_list"""
    assert src.count(old) == 1, "tvm_ffi: cflags block anchor missing"
    src = src.replace(old, new)

    old = 'ninja.append("cxx = {}".format(os.environ.get("CXX", "cl" if IS_WINDOWS else "c++")))'
    new = """if _use_clang_host:
        # patched: MSVC rejects the pack+default-arg RuntimeCheck idiom; use HIP clang++
        _clangxx = str(Path(_find_rocm_home()) / "lib" / "llvm" / "bin" / "clang++.exe")
        ninja.append("cxx = {}".format(os.environ.get("CXX", _clangxx)))
    else:
        ninja.append("cxx = {}".format(os.environ.get("CXX", "cl" if IS_WINDOWS else "c++")))"""
    assert src.count(old) == 1, "tvm_ffi: cxx anchor missing"
    src = src.replace(old, new)

    old = 'ninja.append("  command = $cxx /showIncludes $cxxflags -c $in /Fo$out")'
    new = """if _use_clang_host:
            ninja.append("  command = $cxx -MMD -MF $out.d $cxxflags -c $in -o $out")
        else:
            ninja.append("  command = $cxx /showIncludes $cxxflags -c $in /Fo$out")"""
    assert src.count(old) == 1, "tvm_ffi: compile rule anchor missing"
    src = src.replace(old, new)

    old = '''        if IS_WINDOWS:
            ninja.append("  command = $cxx $in /link $ldflags /out:$out")'''
    new = """        if _use_clang_host:
            ninja.append("  command = $cxx -shared $in $ldflags -o $out")
        elif IS_WINDOWS:
            ninja.append("  command = $cxx $in /link $ldflags /out:$out")"""
    assert src.count(old) == 1, "tvm_ffi: link rule anchor missing"
    src = src.replace(old, new)

    # 3. offload arch from TVM_FFI_ROCM_ARCH_LIST on Windows too
    f.write_text(src, encoding="utf-8")
    print(f"[patched] tvm_ffi ({f})")
    return True


def patch_triton_amd() -> None:
    p = Path(sys.modules["triton"].__file__).parent / "backends" / "amd" / "compiler.py"
    src = p.read_text(encoding="utf-8")
    if "launch_pdl" in src:
        print(f"[ok] triton amd already patched ({p})")
        return
    anchor = "@dataclass(frozen=True)\nclass HIPOptions:"
    assert anchor in src, "triton: HIPOptions anchor missing"
    idx = src.index(anchor) + len(anchor)
    src = (
        src[:idx]
        + "\n    launch_pdl: bool = False  # patched: accepted+ignored; nvidia-only opt"
        + src[idx:]
    )
    p.write_text(src, encoding="utf-8")
    print(f"[patched] triton amd ({p})")


def patch_uvicorn() -> None:
    import uvicorn.loops.asyncio as ua

    f = Path(ua.__file__)
    src = f.read_text(encoding="utf-8")
    if "SelectorEventLoop  #" in src or "return asyncio.SelectorEventLoop" in src:
        print(f"[ok] uvicorn already patched ({f})")
        return
    old = "return asyncio.ProactorEventLoop"
    assert src.count(old) == 1, "uvicorn: proactor anchor missing"
    src = src.replace(
        old,
        "return asyncio.SelectorEventLoop  # patched: zmq.asyncio needs add_reader",
    )
    f.write_text(src, encoding="utf-8")
    print(f"[patched] uvicorn ({f})")


def main() -> int:
    try:
        patch_tvm_ffi()
    except Exception as e:
        print(f"[FAIL] tvm_ffi: {e}")
        return 1
    try:
        patch_triton_amd()
    except Exception as e:
        print(f"[FAIL] triton: {e}")
        return 1
    try:
        patch_uvicorn()
    except Exception as e:
        print(f"[FAIL] uvicorn: {e}")
        return 1
    print("\nAll upstream patches applied. Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
