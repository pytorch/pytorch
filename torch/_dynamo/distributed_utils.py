import warnings
from torch._dynamo import config
from torch._dynamo.exc import Unsupported, UserError


def p2p_compile_guard():
    if config.enable_p2p_compilation:
        return
    msg = (
        "Encountered P2P Op during torch.compile, "
        "P2P compilation is disabled. "
        "Falling back to eager for this region. "
        "Set TORCHDYNAMO_ENABLE_P2P_COMPILATION=1 to enable."
    )
    warnings.warn(msg, UserWarning, stacklevel=3)
    raise Unsupported(msg)
