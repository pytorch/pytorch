r"""Decorators in this module are the preferred ways to check whether certain CUDA features are available."""

if __name__ == '__main__':
    from common_cuda import TEST_CUDA, TEST_MULTIGPU, CUDA_DEVICE, TEST_CUDNN, TEST_CUDNN_VERSION, TEST_MAGMA


def skipIfNoCuda(function):
    def wrapper(*args, **kwargs):
        if TEST_CUDA:
            return function(*args, **kwargs)
        else:
            print("CUDA unavailable")
    return wrapper


def skipIfNoMultiGpu(function):
    def wrapper(*args, **kwargs):
        if TEST_MULTIGPU:
            return function(*args, **kwargs)
        else:
            print("only one GPU detected")
    return wrapper


def skipIfNoCudnn(function):
    def wrapper(*args, **kwargs):
        if TEST_CUDNN:
            return function(*args, **kwargs)
        else:
            print("CUDNN not available")
    return wrapper


def skipIfCudnnVersionLessThan(min_version):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if TEST_CUDNN and TEST_CUDNN_VERSION >= min_version:
                return function(*args, **kwargs)
            else:
                print("needs cudnn >= " + str(min_version))
        return wrapper
    return decorator


def skipIfNoMagma(function):
    def wrapper(*args, **kwargs):
        if TEST_MAGMA:
            return function(*args, **kwargs)
        else:
            print("no MAGMA library detected")
    return wrapper
