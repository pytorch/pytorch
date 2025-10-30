import torch
import torch.nn as nn

# Simple example to show how backward knows about compiled forward

model = nn.Linear(10, 5)
compiled_model = torch.compile(model)

# Run forward
input_data = torch.randn(64, 10, requires_grad=True)
output = compiled_model(input_data)

# Let's inspect the grad_fn
print("Output grad_fn:", output.grad_fn)
print("Output grad_fn class:", type(output.grad_fn))
print("Output grad_fn name:", output.grad_fn.name())

# The key: the grad_fn points to CompiledFunction!
# When you call loss.backward(), PyTorch's autograd engine 
# walks the computation graph and calls the backward method
# of each grad_fn it encounters.

loss = output.sum()
print("\nLoss grad_fn:", loss.grad_fn)
print("Loss grad_fn next_functions:", loss.grad_fn.next_functions)

# When we call backward, it automatically uses the compiled backward
# because the output tensor's grad_fn is connected to CompiledFunction.backward
loss.backward()

print("\nBackward completed successfully!")
print("Input gradient shape:", input_data.grad.shape)
