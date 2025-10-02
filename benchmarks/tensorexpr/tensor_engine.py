tensor_engine = None


def unsupported(func):
    def wrapper(self):
        return func(self)

    wrapper.is_supported = False
    return wrapper


def is_supported(method):
    if hasattr(method, "is_supported"):
        return method.is_supported
    return True


def set_engine_mode(mode):
    global tensor_engine
    if mode == "tf":
        from . import tf_engine

        tensor_engine = tf_engine.TensorFlowEngine()
    elif mode == "pt":
        from . import pt_engine

        tensor_engine = pt_engine.TorchTensorEngine()
    elif mode == "topi":
        from . import topi_engine

        tensor_engine = topi_engine.TopiEngine()
    elif mode == "relay":
        from . import relay_engine

        tensor_engine = relay_engine.RelayEngine()
    elif mode == "nnc":
        from . import nnc_engine

        tensor_engine = nnc_engine.NncEngine()
    else:
        raise ValueError(f"invalid tensor engine mode: {mode}")
    tensor_engine.mode = mode


def get_engine():
    if tensor_engine is None:
        raise ValueError("use of get_engine, before calling set_engine_mode is illegal")
    return tensor_engine
