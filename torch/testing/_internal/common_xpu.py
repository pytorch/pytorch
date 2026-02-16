from torch._inductor.codegen.xpu.xpu_env import get_xpu_arch


PLATFORM_SUPPORTS_SYCLTLA: bool = get_xpu_arch() == "Xe20"
