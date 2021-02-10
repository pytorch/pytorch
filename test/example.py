import torch


class ExampleModule(torch.nn.Module):
    def __init__(self, hidden_size, scale_factor, use_mask):
        super().__init__()
        self.scale = int(hidden_size / scale_factor)
        self.use_mask = use_mask

    def forward(
        self,
        inputs : torch.Tensor,
        mask : torch.Tensor=torch.Tensor()
    ):
        tmp = inputs / self.scale
        if self.use_mask :
            tmp = tmp + mask
        outputs = torch.nn.functional.softmax(tmp, dim=-1)

        return outputs

mod = torch.jit.script(ExampleModule(2, 3, True))
out = torch.jit._script.RecursiveScriptModule(torch._C._freeze_module(mod._c, preserveParameters=True))
print(out)
# RecursiveScriptModule._finalize_scriptmodule(out)
# # torch._C._freeze_module(mod._c)
# # out = torch._C._freeze_module(module, preserveParameters=True)
# # add back on nice python wrapping
# torch.jit._script.RecursiveScriptModule._finalize_scriptmodule(out)
# print(out.graph)
