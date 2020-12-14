class OuterModuleSingleIO(Module):
  __parameters__ = []
  __buffers__ = []
  __forward_pre_hooks__ = ["pre_hook_140334059210208", "pre_hook_140334059210208", ]
  __forward_hooks__ = ["forward_hook_140334059210208", "forward_hook_140334059210208", ]
  training : bool
  name : str
  submodule : __torch__.InnerModuleSingleIO
  def forward(self: __torch__.OuterModuleSingleIO,
    input: str) -> str:
    input0 = torch.add(input, "_outermod")
    return (self.submodule).forward(input0, )
  def forward_hook_140334059210208(self: __torch__.OuterModuleSingleIO,
    input: Tuple[str],
    output: str) -> str:
    _0 = torch.eq(self.name, "outer_mod_name")
    if _0:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    _1, = input
    _2 = torch.eq([_1], ["pre_hook_overrid_name"])
    if _2:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    return torch.add(output, "_fh")
  def pre_hook_140334059210208(self: __torch__.OuterModuleSingleIO,
    input: Tuple[str]) -> Tuple[str]:
    _3 = torch.eq(self.name, "outer_mod_name")
    if _3:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    return ("pre_hook_overrid_name",)
class InnerModuleSingleIO(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  name : str
  def forward(self: __torch__.InnerModuleSingleIO,
    input: str) -> str:
    return torch.add(input, "_inner_mod")
