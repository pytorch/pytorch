import torch
from torch.utils._pytree import tree_map_with_path

# Repro for: torch._dynamo.exc.Unsupported: Failed to trace builtin operator
# Explanation: Dynamo does not know how to trace builtin operator `repr` with
# argument types ['tuple']
#
# In eager mode: raises ValueError("Expected Tensor at path=...")
# Under compile: raises Unsupported("Failed to trace builtin operator repr")


def check_tensor(path, x):
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected Tensor at {path=}")
    return x * 2


def fn(tree):
    return tree_map_with_path(check_tensor, tree)


if __name__ == "__main__":
    tree = {"a": torch.randn(10), "b": 5}  # b is not a tensor 

    print("=== Eager mode ===")
    try:
        fn(tree)
    except ValueError as e:
        print(f"ValueError: {e}")

    print("\n=== Compiled mode ===")
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    try:
        compiled_fn(tree)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
