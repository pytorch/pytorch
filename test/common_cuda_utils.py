def skipIfNoCuda(function):
    def wrapper(*args, **kwargs):
        if TEST_CUDA:
            return function(*args, **kwargs)
    return wrapper


def skipIfNoMultiGpu(function):
    def wrapper(*args, **kwargs):
        if TEST_MULTIGPU:
            return function(*args, **kwargs)
    return wrapper


def skipIfNoCudnn(function):
    def wrapper(*args, **kwargs):
        if TEST_CUDNN:
            return function(*args, **kwargs)
    return wrapper


def skipIfNoMagma(function):
    def wrapper(*args, **kwargs):
        if TEST_MAGMA:
            return function(*args, **kwargs)
    return wrapper
