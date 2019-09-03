import builtins
from .modules import * 
from .parameter import Parameter as Parameter
from .parallel import DataParallel as DataParallel
from . import init as init
from . import utils as utils

class Parameter:
    @overload
    def __init__(self, data, requires_grad: builtins.bool): ...
