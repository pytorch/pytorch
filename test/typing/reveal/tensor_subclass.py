import torch


class TensorSubclass(torch.Tensor):
    pass


t = torch.randn(2, 3)
reveal_type(t.clone())  # E: {Tensor}
reveal_type(t.detach())  # E: {Tensor}
reveal_type(t.new_empty((2, 3)))  # E: {Tensor}

s = TensorSubclass()
reveal_type(s.clone())  # E: {TensorSubclass}
reveal_type(s.detach())  # E: {TensorSubclass}
reveal_type(s.new_empty((2, 3)))  # E: {TensorSubclass}
reveal_type(s.new_empty_strided((2, 3), (3, 1)))  # E: {TensorSubclass}
reveal_type(s.new_full((2, 3), 1.0))  # E: {TensorSubclass}
reveal_type(s.new_ones((2, 3)))  # E: {TensorSubclass}
reveal_type(s.new_zeros((2, 3)))  # E: {TensorSubclass}
reveal_type(s.new_tensor([1.0, 2.0]))  # E: {TensorSubclass}
