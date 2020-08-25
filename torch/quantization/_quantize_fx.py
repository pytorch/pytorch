from .fx import Fuser  # noqa: F401
from .fx import Quantizer  # noqa: F401
from .fx import QuantType  # noqa: F401

def fuse_fx(graph_module, inplace=False):
    fuser = Fuser()
    return fuser.fuse(graph_module, inplace)

def _prepare_fx(graph_module, qconfig_dict, inplace, quant_type):
    pass

def prepare_fx(graph_module, qconfig_dict, inplace):
    """ If graph_module is in training mode
    """
    pass

def prepare_dynamic_fx(graph_module, qconfig_dict, inplace):
    pass

def convert_fx(graph_module, inplace):
    pass
