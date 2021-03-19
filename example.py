from monkeytype import trace
from _monkey_config_jit import my_config
from example_1 import add_sum
import torch

with trace(my_config):
    def p(a, b):
        c = torch.jit.script(add_sum)
        d = c(a, b)
        return d
    k = p(2, 4)
