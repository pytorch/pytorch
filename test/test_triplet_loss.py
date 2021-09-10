import torch
import torch.nn as nn

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, reduction='none')
anchor = torch.randn(100, 128, requires_grad=True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)
output = triplet_loss(anchor, positive, negative)

for i in range(100):
    output_no_batch = triplet_loss(anchor[i], positive[i], negative[i])
    assert output_no_batch == output[i]

