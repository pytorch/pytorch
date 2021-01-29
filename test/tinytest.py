import torch
from typing import Dict

@torch.jit.script
def fn():
    x: Dict[str, int] = dict()
    x["foo"] = 1
    return x



    #return dict(dict(foo=1, bar=2, baz=3))
    return dict({})
    #return dict({chr(65+i) : i for i in range(4)}, foo=2, bar=3)
    #return dict(foo=1, bar=2, baz=3)
    #return list(range(5))
    #return {i : chr(65+i) for i in range(4)}
    #return [i for i in range(4)]

print(fn.graph)
print(fn())


#class M(torch.nn.Module):
#    def __init__(self):
#        super(M, self).__init__()
#        self.choices = torch.nn.ModuleDict({
#                'conv': torch.nn.Conv2d(10, 10, 3),
#                'pool': torch.nn.MaxPool2d(3)
#        })

#    def forward(self, x):
#        return dict(self.choices.items())

#m = torch.jit.script(M())
#print(m.forward.graph)
