import torch
from torchao.utils import benchmark_model
from torch.sparse import to_sparse_semi_structured

torch.sparse.semi_structured.SparseSemiStructuredTensor._FORCE_CUTLASS = False


model = torch.nn.Linear(1024, 128).half().cuda().eval()
# model.weight = torch.nn.Parameter(to_sparse_semi_structured(model.weight.data))

input = torch.randn(1, 1024).half().cuda()


model = torch.compile(model, mode="max-autotune", fullgraph=True)

benchmark_model(model, 10, args=(input,))
print("Dense benchmark: ", benchmark_model(model, 100, args=(input,)))

del model

model = torch.nn.Linear(1024, 128).half().cuda().eval()
model.weight = torch.nn.Parameter(to_sparse_semi_structured(model.weight.data))
model = torch.compile(model, mode="max-autotune", fullgraph=True)

benchmark_model(model, 10, args=(input,))
print("Sparse benchmark: ", benchmark_model(model, 100, args=(input,)))
