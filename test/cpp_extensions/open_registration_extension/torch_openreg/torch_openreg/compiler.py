import logging

import torch
from torch._dynamo.backends.registry import register_backend

log = logging.getLogger(__name__)


def openreg(gm, example_inputs):
    gm.graph.lint()

    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            fake = node.meta["val"]
            if isinstance(fake, torch.Tensor):
                assert fake.device.type in ("openreg", "cpu"), (
                    f"Unexpected device {fake.device} in openreg backend"
                )

    gm.graph.eliminate_dead_code()
    gm.recompile()

    code = gm.graph.python_code("self")
    log.debug("Compiled graph source:\n%s", code.src)

    return gm.forward


register_backend(compiler_fn=openreg, name="openreg")

