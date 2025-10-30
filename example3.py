import torch
import torch.nn as nn

# 1. Define a simple neural network.
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Instantiate the model.
model = SimpleNet()

# Create dummy input data.
input_data = torch.randn(1, 5)

# --- After the forward pass, before the backward pass ---
# 2. Perform a forward pass.
output = model(input_data)

# Print gradients before backward(). They will be None.
print("--- Gradients after forward pass (before backward pass) ---")
print("fc1.weight.grad:", model.fc1.weight.grad)
print("fc1.bias.grad:", model.fc1.bias.grad)
print("fc2.weight.grad:", model.fc2.weight.grad)
print("fc2.bias.grad:", model.fc2.bias.grad)
print("-" * 50)

# 3. Calculate a scalar loss.
target_data = torch.randn(1, 1)
loss = nn.MSELoss()(output, target_data)

# --- After the backward pass ---
# 4. Perform the backward pass to compute gradients.
loss.backward()

# Print gradients after backward(). They will now be populated with values.
print("--- Gradients after backward pass ---")
print("fc1.weight.grad:\n", model.fc1.weight.grad)
print("fc1.bias.grad:\n", model.fc1.bias.grad)
print("fc2.weight.grad:\n", model.fc2.weight.grad)
print("fc2.bias.grad:\n", model.fc2.bias.grad)