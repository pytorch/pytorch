# mypy: disable-error-code="no-untyped-def"
import torch.cuda


class cudaError_t:
    cudaSuccess = True


def cudaFree(n):
    return (cudaError_t.cudaSuccess,)


def cudaGetDeviceProperties(d):
    class DummyError:
        value = False

    return (DummyError(), torch.cuda.get_device_properties(d))
