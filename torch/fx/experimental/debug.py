import torch.fx as fx

def set_trace(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Sets a breakpoint in `gm`'s generated python code. It drops into pdb when
    `gm` gets run.

    Args:
        gm: graph module to insert breakpoint. It is then recompiled for it to
            take effect.

    Returns:
        the `gm` with breakpoint inserted.
    """
    def insert_pdb(body):
        return ["import pdb; pdb.set_trace()\n", *body]

    with gm.graph.on_generate_code(
        make_transformer=lambda cur_transform: (
            # new code transformer to register
            lambda body: (
                insert_pdb(
                    cur_transform(body) if cur_transform
                    else body
                )
            )
        )
    ):
        gm.recompile()

    return gm
