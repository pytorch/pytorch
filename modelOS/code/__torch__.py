class NestedModuleOS(Module):
  __parameters__ = []
  __buffers__ = []
  __forward_pre_hooks__ = [pre_hook_os, ]
  __forward_hooks__ = [forward_hook_os, ]
  training : bool
  name : str
  submodule : __torch__.BasicModuleOS
  def forward(self: __torch__.NestedModuleOS,
    input: List[str]) -> List[str]:
    _0 = torch.append(input, self.name)
    _1 = (self).other_method_top()
    return (self.submodule).forward(input, )
  def other_method_top(self: __torch__.NestedModuleOS) -> None:
    print("other_method!")
    return None
  def forward_hook_os(self: __torch__.NestedModuleOS,
    input: Tuple[List[str]],
    output: List[str]) -> List[str]:
    _2 = torch.eq(self.name, "outer_mod_name")
    if _2:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    _3 = torch.eq(((input)[0])[0], "pre_hook_overrid_name")
    if _3:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    return torch.add(output, ["overrid"])
  def pre_hook_os(self: __torch__.NestedModuleOS,
    input: Tuple[List[str]]) -> Tuple[List[str]]:
    _4 = torch.eq(self.name, "outer_mod_name")
    if _4:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    return (["pre_hook_overrid_input"],)
class BasicModuleOS(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  name : str
  def forward(self: __torch__.BasicModuleOS,
    input: List[str]) -> List[str]:
    _5 = torch.append(input, self.name)
    _6 = (self).other_method_sub()
    return input
  def other_method_sub(self: __torch__.BasicModuleOS) -> None:
    print("other_method!")
    return None
