import torch
from test_jit import get_execution_plan
from common_utils import enable_profiling_mode

with enable_profiling_mode():
    @torch.jit.script
    def fct_loop(x):
        for i in range(3):
            x = torch.cat((x, x), 0)
        return x

    x = torch.ones(2, 3, 4, dtype=torch.float32)
    # profile
    fct_loop(x)
    fct_loop(x)

    dstate = fct_loop.get_debug_state()
    eplan = get_execution_plan(dstate)
    num_bailouts = eplan.code.num_bailouts()

    for i in range(0, num_bailouts):
        eplan.code.request_bailout(i)
        fct_loop(x)


