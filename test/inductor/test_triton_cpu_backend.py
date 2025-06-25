# Owner(s): ["module: inductor"]
from torch._inductor import config
from torch._inductor.test_case import run_tests
from torch.testing._internal.inductor_utils import (
    get_inductor_device_type_test_class,
    HAS_CPU,
    TRITON_HAS_CPU,
)


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor


if HAS_CPU and TRITON_HAS_CPU:
    TestTorchInductorTritonCPU = get_inductor_device_type_test_class(
        test_module=test_torchinductor,
        generic_test_cls_name=test_torchinductor.TEST_TORCHINDUCTOR_GENERIC_CLS_NAME,
        backend="triton",
        device="cpu",
    )

    @config.patch(cpu_backend="triton")
    class SweepInputsCpuTritonTest(test_torchinductor.SweepInputsCpuTest):
        pass


if __name__ == "__main__":
    if HAS_CPU and TRITON_HAS_CPU:
        run_tests(needs="filelock")
