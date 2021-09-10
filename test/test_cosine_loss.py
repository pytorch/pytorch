import torch
import torch.nn as nn

loss = nn.CosineEmbeddingLoss(reduction='none')
# x1 = torch.randn(100, 128, requires_grad=True)
# x2 = torch.randn(100, 128, requires_grad=True)
# target = torch.ones(100)
# m = torch.empty_like(target).bernoulli_().bool()
# target[m] = -1
# assert (target == -1).sum() > 0
# assert (target == 1).sum() > 0
# output_batch = loss(x1, x2, target)
# for i in range(100):
#     output_no_batch = loss(x1[i], x2[i], target[i])
#     assert output_no_batch == output_batch[i]

x = torch.randn(1, 10)
y = torch.randn(1, 10)
# print(loss(x, y, torch.tensor([0]).sign()))
print(loss(torch.randn(10), torch.randn(10), torch.tensor([0]).sign()))
