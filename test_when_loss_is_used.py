import torch
import torch.nn as nn

print("=" * 70)
print("WHEN IS THE LOSS VALUE USED?")
print("=" * 70)

# Simple model
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

input_data = torch.randn(5, 10)
target = torch.randn(5, 1)

print("\n1. FORWARD PASS - Loss value is computed")
print("-" * 70)
output = model(input_data)
loss = criterion(output, target)
print(f"Loss value: {loss.item():.4f}")
print(f"Purpose: Measures how wrong the model is")

print("\n2. BACKWARD PASS - Gradient starts at 1, NOT loss value")
print("-" * 70)
loss.backward()
print(f"Loss value: {loss.item():.4f} (unchanged)")
print(f"Starting gradient: 1.0 (d(loss)/d(loss) = 1)")
print(f"Gradients computed: model.weight.grad.shape = {model.weight.grad.shape}")
print(f"Purpose: Compute derivatives for optimization")

print("\n3. OPTIMIZATION - Gradients (NOT loss value) update parameters")
print("-" * 70)
print(f"Before: model.weight[0,0] = {model.weight[0,0].item():.4f}")
optimizer.step()
print(f"After:  model.weight[0,0] = {model.weight[0,0].item():.4f}")
print(f"Purpose: Update weights using gradients")
print(f"Note: Loss VALUE is not used in optimizer.step()!")

print("\n" + "=" * 70)
print("WHERE THE LOSS VALUE IS ACTUALLY USED:")
print("=" * 70)

print("\n✓ 1. MONITORING TRAINING PROGRESS")
print("-" * 70)
print(f"   Epoch 1: loss = {loss.item():.4f}  ← Printed to console")
print(f"   Epoch 2: loss = 0.5234  ← Track if model is improving")

print("\n✓ 2. EARLY STOPPING")
print("-" * 70)
print("   if loss.item() < best_loss:")
print("       best_loss = loss.item()")
print("       save_checkpoint()")
print("   elif loss.item() > best_loss + threshold:")
print("       break  # Stop training")

print("\n✓ 3. LEARNING RATE SCHEDULING")
print("-" * 70)
print("   scheduler = ReduceLROnPlateau(optimizer, 'min')")
print("   scheduler.step(loss.item())  ← Uses loss VALUE")
print("   # Reduces LR if loss plateaus")

print("\n✓ 4. LOGGING & VISUALIZATION")
print("-" * 70)
print("   tensorboard.add_scalar('loss', loss.item(), epoch)")
print("   wandb.log({'loss': loss.item()})")

print("\n✓ 5. MODEL SELECTION")
print("-" * 70)
print("   if val_loss.item() < best_val_loss:")
print("       best_model = model.state_dict()")

print("\n" + "=" * 70)
print("DEMONSTRATION: Loss value vs Gradients")
print("=" * 70)

# Reset for demo
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Scenario 1: Low loss
output1 = model(input_data)
target1 = output1.clone().detach()  # Perfect match
loss1 = criterion(output1, target1)

print(f"\nScenario 1: Perfect prediction")
print(f"  Loss VALUE: {loss1.item():.6f}  ← Very small, model is good!")
print(f"  Loss used for: Monitoring (tells us model is performing well)")

loss1.backward()
print(f"  Gradient flow: Starts at 1.0, flows to compute weight gradients")
print(f"  Max gradient magnitude: {model.weight.grad.abs().max().item():.6f}")
print(f"  Gradients used for: Small parameter updates (model already good)")

# Scenario 2: High loss
model.zero_grad()
output2 = model(input_data)
target2 = output2 + 10.0  # Very different
loss2 = criterion(output2, target2)

print(f"\nScenario 2: Bad prediction")
print(f"  Loss VALUE: {loss2.item():.6f}  ← Large, model is bad!")
print(f"  Loss used for: Monitoring (tells us model needs more training)")

loss2.backward()
print(f"  Gradient flow: Starts at 1.0, flows to compute weight gradients")
print(f"  Max gradient magnitude: {model.weight.grad.abs().max().item():.6f}")
print(f"  Gradients used for: Large parameter updates (model needs fixing)")

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("Loss VALUE: Human-readable metric (monitoring, logging, decisions)")
print("Loss GRADIENT: Mathematical tool (computing parameter updates)")
print("\nBoth come from the SAME forward computation,")
print("but serve DIFFERENT purposes!")
print("=" * 70)
