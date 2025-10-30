import torch
import torch.nn as nn

print("=" * 70)
print("WHY DO WE CALL .backward() ON THE LOSS?")
print("=" * 70)

print("\nThe answer: We call .backward() on the TENSOR we want to differentiate")
print("with respect to! It's not about the loss VALUE, it's about POSITION")
print("in the computation graph!")

print("\n" + "=" * 70)
print("EXAMPLE 1: Calling backward on LOSS (normal case)")
print("=" * 70)

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2        # y = 4.0
z = y + 3         # z = 7.0
loss = z * 0.5    # loss = 3.5

print(f"\nComputation: x={x.item()} → y=x²={y.item()} → z=y+3={z.item()} → loss=z*0.5={loss.item()}")

# Call backward on LOSS
loss.backward()

print(f"\nCalling loss.backward() computes: d(loss)/dx")
print(f"Result: x.grad = {x.grad.item()}")
print(f"Expected: d(loss)/dx = d(0.5*(x²+3))/dx = 0.5*2x = 0.5*2*2 = 2.0 ✓")

print("\n" + "=" * 70)
print("EXAMPLE 2: Calling backward on Z instead of LOSS")
print("=" * 70)

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2        # y = 4.0
z = y + 3         # z = 7.0
loss = z * 0.5    # loss = 3.5

print(f"\nSame computation: x={x.item()} → y={y.item()} → z={z.item()} → loss={loss.item()}")

# Call backward on Z instead!
z.backward()

print(f"\nCalling z.backward() computes: d(z)/dx (not d(loss)/dx!)")
print(f"Result: x.grad = {x.grad.item()}")
print(f"Expected: d(z)/dx = d(x²+3)/dx = 2x = 2*2 = 4.0 ✓")
print(f"\nNotice: Different gradient because we differentiated w.r.t. Z, not LOSS!")

print("\n" + "=" * 70)
print("EXAMPLE 3: Calling backward on Y")
print("=" * 70)

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2        # y = 4.0
z = y + 3         # z = 7.0
loss = z * 0.5    # loss = 3.5

print(f"\nSame computation: x={x.item()} → y={y.item()} → z={z.item()} → loss={loss.item()}")

# Call backward on Y!
y.backward()

print(f"\nCalling y.backward() computes: d(y)/dx (not d(loss)/dx!)")
print(f"Result: x.grad = {x.grad.item()}")
print(f"Expected: d(y)/dx = d(x²)/dx = 2x = 2*2 = 4.0 ✓")

print("\n" + "=" * 70)
print("EXAMPLE 4: Real ML scenario - Why loss matters")
print("=" * 70)

model = nn.Linear(5, 3)
x = torch.randn(2, 5)
target = torch.randint(0, 3, (2,))

# Forward pass
output = model(x)           # Model predictions
probabilities = torch.softmax(output, dim=1)  # Convert to probabilities
loss = nn.CrossEntropyLoss()(output, target)   # How wrong we are

print(f"\nComputation graph:")
print(f"  x → model → output → softmax → probabilities")
print(f"           ↘ loss (CrossEntropyLoss)")
print(f"\nLoss value: {loss.item():.4f}")

# Option 1: Backward on LOSS
model.zero_grad()
loss.backward()
grad_from_loss = model.weight.grad.clone()
print(f"\nOption 1: loss.backward()")
print(f"  Computes: d(loss)/d(weights)")
print(f"  Purpose: Update weights to MINIMIZE loss")
print(f"  Gradient magnitude: {grad_from_loss.abs().mean().item():.6f}")

# Option 2: Backward on OUTPUT (wrong!)
model.zero_grad()
# Can't call output.backward() directly because output is not scalar
# But conceptually, if we did:
print(f"\nOption 2: output.backward() [conceptually]")
print(f"  Would compute: d(output)/d(weights)")
print(f"  Problem: Doesn't tell us how to MINIMIZE LOSS!")
print(f"  We don't care about d(output)/d(weights), we care about d(LOSS)/d(weights)!")

print("\n" + "=" * 70)
print("KEY INSIGHT: Why we call backward on LOSS")
print("=" * 70)

print("\n1. .backward() means: 'Compute gradients WITH RESPECT TO this tensor'")
print("\n2. In ML, we want to minimize LOSS, so we compute:")
print("   d(loss)/d(weights) ← This tells us how to change weights to reduce loss")
print("\n3. The loss VALUE (e.g., 3.5) doesn't matter for backward")
print("   - Backward starts with gradient=1 at the loss tensor")
print("   - It computes d(loss)/d(everything_before_loss)")
print("\n4. We MUST call backward on loss because:")
print("   - Loss is the OBJECTIVE we want to minimize")
print("   - Gradients w.r.t. loss tell us the direction to move parameters")
print("   - Calling backward on anything else would give wrong update direction!")

print("\n" + "=" * 70)
print("VISUAL ANALOGY")
print("=" * 70)
print("\nThink of it like asking different questions:")
print("\n  y.backward()    asks: 'How does x affect y?'")
print("  z.backward()    asks: 'How does x affect z?'")
print("  loss.backward() asks: 'How does x affect LOSS?'")
print("\nIn ML, we only care about the LAST question!")
print("We want to know how parameters affect LOSS, not intermediate values.")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("✓ We call .backward() on LOSS because it's the END of our graph")
print("✓ Not because of its VALUE, but because of its POSITION")
print("✓ .backward() means: 'differentiate w.r.t. THIS tensor'")
print("✓ We want d(loss)/d(params), not d(something_else)/d(params)")
print("✓ The gradient starts at 1, but flows from LOSS backward to params")
print("=" * 70)
