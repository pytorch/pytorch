import torch

first_activation_dim = 1000
num_batches = 1000
output_channel = 1024
input_channel = 512
activation = torch.rand(num_batches, first_activation_dim, input_channel, device="cuda", dtype=torch.float16)
weight = torch.rand(output_channel, input_channel, device="cuda", dtype=torch.float16)
res = torch.empty(num_batches, first_activation_dim, output_channel, device="cuda", dtype=torch.float16)
bias = torch.rand(output_channel, device="cuda", dtype=torch.float16)

cusparse_linear = torch.classes.cusparselt.CusparseLtLinear(weight)
cusparse_linear.init(activation, res, bias)
cusparse_linear.prune()
# cusparse_linear.compress()
# cusparse_linear.masked_mm()
