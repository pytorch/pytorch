import mkldnn
from torch import nn

# TODO: out of place version
# TODO, cleanup, this is copy-pasta'd from online,
# verify that the forward of `m` is the same as nn.Sequential.forward
def fuse_module(m):
    last_conv = None
    last_conv_name = None

    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2d) and isinstance(m, nn.Sequential)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = fuse_conv_bn_eval(last_conv, child)
            m._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            m._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            last_conv = None
            fuse_module(child)
    return m



def freeze(module):
    if module.training:
        raise RuntimeError(
            "Freezing is currently only implemented for modules in eval mode. "
            "Please call .eval() on your module before freezing."
        )

    fused_conv_bn_model = fuse_module(module)

    # TODO, check that the weights are float32/float16, maybe warn if theyre not
    return mkldnn.to_mkldnn(fused_conv_bn_model, convert_batchnorm=False)
