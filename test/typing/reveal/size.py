import torch
input = []
input.append(torch.tensor([1.0, 2.0, 3.0, 4.0]))
input.append(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
input.append(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
reveal_type(input[0].shape[0])  # E: int
reveal_type(input[1].shape[1])  # E: int
reveal_type(input[2].shape[2])  # E: int
