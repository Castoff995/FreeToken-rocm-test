# FreeToken ROCm Watch

This branch exists only to keep one long-lived pull request open as the notification channel for the scheduled GitHub monitor.

Do not merge or close that pull request while monitoring is enabled.

The monitor runs every two hours and comments only when it finds a meaningful change affecting the Windows + ROCm + RDNA4/gfx1201 FreeToken port, Qwen3.6 serving, MoE host-memory offload, Triton/TVM-FFI JIT, or the supporting ROCm/PyTorch stack.
