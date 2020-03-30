import torch
input = []
input.append(torch.tensor([1.0, 2.0, 3.0, 4.0]))
input.append(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
input.append(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
input[0].shape[0]
input[1].shape[1]
input[2].shape[2]
