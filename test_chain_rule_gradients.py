import torch
import torch.nn as nn

print("=" * 80)
print("WHY GRADIENTS AREN'T ALWAYS 1: THE CHAIN RULE IN ACTION")
print("=" * 80)

print("\nKey Insight:")
print("- The gradient STARTS at 1 (dL/dL = 1)")
print("- But gets MULTIPLIED by derivatives as it flows backward")
print("- Final gradients are products of many terms (rarely 1!)")

print("\n" + "=" * 80)
print("EXAMPLE 1: Simple Chain - Manual Computation")
print("=" * 80)

# Simple: L = (W2 * (W1 * x))
x = torch.tensor([2.0], requires_grad=True)
W1 = torch.tensor([3.0], requires_grad=True)
W2 = torch.tensor([4.0], requires_grad=True)

a1 = W1 * x      # First layer: a1 = 3 * 2 = 6
y = W2 * a1      # Second layer: y = 4 * 6 = 24
L = y            # Loss (for simplicity)

print(f"\nForward pass:")
print(f"  x = {x.item()}")
print(f"  W1 = {W1.item()}")
print(f"  W2 = {W2.item()}")
print(f"  a1 = W1 * x = {a1.item()}")
print(f"  y = W2 * a1 = {y.item()}")
print(f"  L = y = {L.item()}")

print(f"\nBackward pass (chain rule):")
print(f"  At L: gradient = 1.0 (dL/dL = 1)")

# Compute gradients
L.backward()

print(f"\n  At W2: gradient = dL/dL * dL/dW2 = 1.0 * da1 = 1.0 * {a1.item()} = {W2.grad.item()}")
print(f"  Manual: dL/dW2 = d(W2*a1)/dW2 = a1 = {a1.item()} ✓")

print(f"\n  At W1: gradient = dL/dL * dL/dy * dy/da1 * da1/dW1")
print(f"         = 1.0 * 1.0 * W2 * x")
print(f"         = 1.0 * 1.0 * {W2.item()} * {x.item()} = {W1.grad.item()}")
print(f"  Manual: dL/dW1 = W2 * x = {W2.item()} * {x.item()} = {W2.item() * x.item()} ✓")

print(f"\nNotice: Gradient started at 1, but became {W1.grad.item()} for W1!")

print("\n" + "=" * 80)
print("EXAMPLE 2: With Activation Function (ReLU)")
print("=" * 80)

x = torch.tensor([2.0], requires_grad=True)
W1 = torch.tensor([3.0], requires_grad=True)
W2 = torch.tensor([4.0], requires_grad=True)

z1 = W1 * x           # z1 = 6
a1 = torch.relu(z1)   # a1 = max(0, 6) = 6
z2 = W2 * a1          # z2 = 24
L = z2                # Loss

print(f"\nForward pass:")
print(f"  z1 = W1 * x = {z1.item()}")
print(f"  a1 = ReLU(z1) = {a1.item()}")
print(f"  z2 = W2 * a1 = {z2.item()}")
print(f"  L = z2 = {L.item()}")

L.backward()

print(f"\nBackward pass:")
print(f"  At L: gradient = 1.0")
print(f"  At W2: gradient = 1.0 * a1 = {W2.grad.item()}")
print(f"  At a1: gradient = 1.0 * W2 = {W2.item()}")
print(f"  At z1: gradient = (1.0 * W2) * d(ReLU)/dz1")
print(f"         Since z1={z1.item()} > 0, d(ReLU)/dz1 = 1")
print(f"         gradient = {W2.item()} * 1 = {W2.item()}")
print(f"  At W1: gradient = ({W2.item()}) * x = {W2.item()} * {x.item()} = {W1.grad.item()}")

print("\n" + "=" * 80)
print("EXAMPLE 3: Real Neural Network")
print("=" * 80)

# Two-layer network: y = W2 * ReLU(W1 * x + b1) + b2
model = nn.Sequential(
    nn.Linear(3, 4),   # W1: (4,3), b1: (4,)
    nn.ReLU(),
    nn.Linear(4, 1)    # W2: (1,4), b2: (1,)
)

x = torch.randn(2, 3)  # Batch of 2 samples
target = torch.randn(2, 1)

# Forward
output = model(x)
loss = nn.MSELoss()(output, target)

print(f"\nForward pass:")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Loss value: {loss.item():.6f}")

# Backward
loss.backward()

W1 = model[0].weight
b1 = model[0].bias
W2 = model[2].weight
b2 = model[2].bias

print(f"\nBackward pass - Gradients computed:")
print(f"  At loss: gradient = 1.0 (implicit)")
print(f"  At W2: gradient shape = {W2.grad.shape}, max grad = {W2.grad.abs().max().item():.6f}")
print(f"  At b2: gradient shape = {b2.grad.shape}, value = {b2.grad.item():.6f}")
print(f"  At W1: gradient shape = {W1.grad.shape}, max grad = {W1.grad.abs().max().item():.6f}")
print(f"  At b1: gradient shape = {b1.grad.shape}, max grad = {b1.grad.abs().max().item():.6f}")

print(f"\nNotice: None of these gradients are 1!")
print(f"They're computed via chain rule from the starting gradient of 1.")

print("\n" + "=" * 80)
print("EXAMPLE 4: Demonstrating the Chain Rule Step-by-Step")
print("=" * 80)

# Simple network with hooks to capture intermediate gradients
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
W1 = torch.tensor([[0.5, 0.3], [0.2, 0.4]], requires_grad=True)
b1 = torch.tensor([[0.1, 0.2]], requires_grad=True)
W2 = torch.tensor([[0.6], [0.7]], requires_grad=True)
b2 = torch.tensor([[0.05]], requires_grad=True)

# Forward
z1 = x @ W1.T + b1  # Shape: (1, 2)
a1 = torch.relu(z1)
z2 = a1 @ W2.T + b2  # Shape: (1, 1)
loss = z2.sum()

print(f"\nForward values:")
print(f"  x = {x.tolist()}")
print(f"  z1 = x @ W1.T + b1 = {z1.tolist()}")
print(f"  a1 = ReLU(z1) = {a1.tolist()}")
print(f"  z2 = a1 @ W2.T + b2 = {z2.tolist()}")
print(f"  loss = {loss.item():.6f}")

# Save intermediate values for manual gradient computation
z1_val = z1.detach().clone()
a1_val = a1.detach().clone()

# Backward
loss.backward()

print(f"\nBackward gradients:")
print(f"  dL/dL = 1.0 (starting point)")
print(f"  dL/dz2 = 1.0 (since loss = z2.sum())")
print(f"  dL/db2 = {b2.grad.item():.6f}")
print(f"  dL/dW2 = dL/dz2 * a1.T = {W2.grad.squeeze().tolist()}")
print(f"  Expected: a1 = {a1_val.squeeze().tolist()}")
print(f"  dL/da1 = dL/dz2 * W2 = W2 = {W2.squeeze().tolist()}")
print(f"  dL/dz1 = dL/da1 * d(ReLU)/dz1")
print(f"  dL/dW1 = dL/dz1 * x.T")
print(f"  W1.grad = {W1.grad.tolist()}")
print(f"  b1.grad = {b1.grad.tolist()}")

print("\n" + "=" * 80)
print("KEY TAKEAWAY: THE CHAIN RULE MULTIPLIES GRADIENTS")
print("=" * 80)

print("\n1. Gradient flow ALWAYS starts at 1:")
print("   dL/dL = 1")

print("\n2. As it flows backward, it gets MULTIPLIED by local derivatives:")
print("   dL/dW1 = dL/dL × dL/dz2 × dz2/da1 × da1/dz1 × dz1/dW1")
print("          = 1    × 1     × W2    × ReLU' × x")

print("\n3. The PRODUCT of all these terms is almost never 1:")
print("   - Each layer contributes its local derivative")
print("   - Activations contribute their derivatives (ReLU', sigmoid', etc.)")
print("   - The final gradient is the PRODUCT of all these")

print("\n4. This is why we need the chain rule:")
print("   - We compute dL/dL = 1 (trivial)")
print("   - We multiply by local derivatives going backward")
print("   - Each parameter gets a unique gradient based on its position")

print("\n5. Common gradient magnitudes you'll see:")
print("   - Small (< 0.001): Vanishing gradients (deep networks)")
print("   - Medium (0.001 - 1.0): Typical range")
print("   - Large (> 10): Exploding gradients (needs gradient clipping)")
print("   - Exactly 1: Rare! Only for trivial cases like dL/dL")

print("=" * 80)
