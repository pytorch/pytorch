# Where to apply torch.compile?

We recommend applying `torch.compile` to the highest-level function that doesn’t cause excessive problems.
Typically, it is:
- your `train` or `eval` step with the optimizer but without the loop,
- your top-level `nn.Module`
- or some sub-`nn.Module`s.

`torch.compile` specifically doesn’t handle distributed wrapper modules like DDP or FSDP very well,
so consider applying `torch.compile` to the inner module passed to the wrapper.

```python
# inference
model = ...
model.compile()

for _ in range(N_ITERS):
    inp = ...
    out = model(inp)
```

```python
# training
model = ...
opt = torch.optim.Adam(model.parameters())

@torch.compile
def train(mod, data):
    opt.zero_grad(True)
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()
    opt.step()

for _ in range(N_ITERS):
    inp = ...
    train(model, inp)
```

```python
# DistributedDataParallel
model = ...
model.compile()
model_ddp = DistributedDataParallel(model, ...)

for _ in range(N_ITERS):
    inp = ...
    out = model_ddp(inp)
```

<!-- TODO add examples for specific model domains, compile(model) vs. model.compile()-->

## `compile(model)` vs `model.compile()`

Due to nuances to how `torch.compile` interacts with `nn.Module` instances,
we advise using the `.compile()` method of `nn.Module` instances if you wish to compile them as
top-level functions. Nested module calls will be traced correctly -
there is no need to call `.compile()` in that case.

```python
# DO NOT DO THIS
model = MyModel()
model = torch.compile(model)
model(inp)

# DO THIS
model = MyModel()
model.compile()
model(inp)

# this is also acceptable
@torch.compile
def fn(model, inp):
    return model(inp)
model = MyModel()
fn(model, inp)
```
