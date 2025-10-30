import torch
import torch.nn as nn

# Example to show: Do we start with gradient 1 or loss value?

print("=" * 60)
print("EXAMPLE 1: Simple case")
print("=" * 60)

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2  # y = 4.0
z = y + 3   # z = 7.0

print(f"Forward pass: x={x.item()}, y={y.item()}, z={z.item()}")

z.backward()

print(f"x.grad = {x.grad.item()}")
print(f"Expected: dz/dx = d(x²+3)/dx = 2x = 2*2 = 4.0")
print(f"Note: We started with gradient=1 at z, NOT with z's value (7.0)")

print("\n" + "=" * 60)
print("EXAMPLE 2: With actual loss")
print("=" * 60)

# Reset
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

# Loss is some value (e.g., 0.5)
loss = 0.5 * y  # loss = 0.5 * 4 = 2.0
print(f"Forward pass: x={x.item()}, y={y.item()}, loss={loss.item()}")

loss.backward()

print(f"\nx.grad = {x.grad.item()}")
print(f"Expected: d(loss)/dx = d(0.5*x²)/dx = 0.5*2x = 0.5*2*2 = 2.0")
print(f"\nKey point: We started with gradient=1 at loss,")
print(f"           NOT with loss's value (2.0)!")

print("\n" + "=" * 60)
print("EXAMPLE 3: Proving it starts at 1, not loss value")
print("=" * 60)

# If we started with loss value, gradient would be wrong!
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
loss = 0.5 * y  # loss = 2.0

loss.backward()

print(f"loss value = {loss.item()}")
print(f"x.grad = {x.grad.item()}")
print(f"\nIf we started with loss value (2.0):")
print(f"  x.grad would be: 2.0 * 2x = 2.0 * 4 = 8.0 ❌ WRONG!")
print(f"\nBut we actually got: {x.grad.item()} ✓ CORRECT!")
print(f"Because we start with gradient=1, then:")
print(f"  1.0 * d(0.5*x²)/dx = 1.0 * 2x = 4.0 ✓")

print("\n" + "=" * 60)
print("EXAMPLE 4: Explicitly setting starting gradient")
print("=" * 60)

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y + 3

# We can explicitly set the starting gradient
z.backward(gradient=torch.tensor([5.0]))

print(f"z value = {z.item()}")
print(f"Starting gradient = 5.0 (not 1.0)")
print(f"x.grad = {x.grad.item()}")
print(f"Expected: 5.0 * dz/dx = 5.0 * 2x = 5.0 * 4 = 20.0 ✓")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("✓ Backward pass ALWAYS starts with gradient = 1")
print("✓ The loss VALUE is used in the forward pass")
print("✓ The gradient (derivative) is computed in the backward pass")
print("✓ d(loss)/d(loss) = 1 (derivative of anything w.r.t. itself is 1)")
print("✓ This '1' is the seed that propagates backward via chain rule")
