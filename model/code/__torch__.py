class NestedModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  name : str
  submodule : __torch__.BasicModule
  def forward(self: __torch__.NestedModule,
    input: List[str]) -> List[str]:
    _0 = torch.append(input, self.name)
    return (self.submodule).forward(input, )
class BasicModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  name : str
  def forward(self: __torch__.BasicModule,
    input: List[str]) -> List[str]:
    _1 = torch.append(input, self.name)
    return input
