"""
Spike: Recursive Dynamo Tracing - Step 2

Goal: Demonstrate that torch.compile can trace through a pre-existing torch.compile
region and have the graph invocation appear as an invoke_subgraph HOP in the graph.

With the modifications to PyTorch:
1. When Dynamo encounters a function with a cached compilation, it uses invoke_subgraph
   instead of inlining the function
2. The cached GraphModule is installed in the outer graph's nn_modules
3. No guard evaluation - we assume the first cache entry hits
"""

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def simple_fn(x, y):
    """A simple function to compile."""
    return x * 2 + y


# Test inputs
x = torch.randn(3, 3)
y = torch.randn(3, 3)


# =============================================================================
# Test 1: make_fx tracing (previous test - should still work)
# =============================================================================
print("=" * 60)
print("Test 1: make_fx tracing with invoke_subgraph")
print("=" * 60)

from torch.fx.experimental.proxy_tensor import make_fx

# Step 1: Create a compiled function
compiled_fn = torch.compile(simple_fn, backend="eager")

# Step 2: Warm up with FakeTensor to match the tracing context
print("Warming up compiled function with FakeTensor...")
fake_mode = FakeTensorMode()
with fake_mode:
    fake_x = fake_mode.from_tensor(x)
    fake_y = fake_mode.from_tensor(y)
    warmup_result = compiled_fn(fake_x, fake_y)
    print(f"Warmup result shape: {warmup_result.shape}")
print()


# Step 3: Define an outer function that calls the compiled function
def outer_fn(x, y):
    z = x + 1
    result = compiled_fn(z, y)
    return result * 2


# Step 4: Trace the outer function with make_fx
print("Tracing outer function with make_fx(tracing_mode='symbolic')...")
try:
    traced = make_fx(outer_fn, tracing_mode="symbolic")(x, y)
    print("\n✓ SUCCESS! Traced graph:")
    print(traced.code)

    # Check for invoke_subgraph in the graph
    print("\nAnalyzing graph nodes:")
    found_invoke_subgraph = False
    for node in traced.graph.nodes:
        if node.op == "call_function":
            if "invoke_subgraph" in str(node.target):
                found_invoke_subgraph = True
                print(f"  ✓ Found invoke_subgraph node: {node}")
                print(f"    Target: {node.target}")
                print(f"    Args: {node.args}")

    if not found_invoke_subgraph:
        print("  ✗ No invoke_subgraph node found")
        print("\n  All call_function nodes:")
        for node in traced.graph.nodes:
            if node.op == "call_function":
                print(f"    {node}: {node.target}")

    # Verify execution still works
    print("\nVerifying execution:")
    expected = (x + 1) * 2 + y  # z = x + 1, then z * 2 + y
    expected = expected * 2  # final * 2
    actual = traced(x, y)
    # Note: the graph might return a tuple - handle that
    if isinstance(actual, tuple):
        actual = actual[0]
    print(f"  Expected shape: {expected.shape}")
    print(f"  Actual shape: {actual.shape}")
    print(f"  Values match: {torch.allclose(actual, expected)}")

except Exception as e:
    import traceback
    print(f"\n✗ Error: {e}")
    traceback.print_exc()

print()

# =============================================================================
# Test 2: Recursive torch.compile - outer compile calls inner compile
# =============================================================================
print("=" * 60)
print("Test 2: Recursive torch.compile with invoke_subgraph")
print("=" * 60)

# Reset the cache and create fresh functions
torch._dynamo.reset()
from torch._dynamo.output_graph import _invoke_subgraph_cache
_invoke_subgraph_cache.clear()


def inner_fn(x, y):
    """The inner function that gets compiled first."""
    return x * 2 + y


# Step 1: Compile the inner function first
inner_compiled = torch.compile(inner_fn, backend="eager")

# Step 2: Warm up the inner compiled function
print("Step 1: Warming up inner compiled function...")
with FakeTensorMode() as fake_mode:
    fake_x = fake_mode.from_tensor(x)
    fake_y = fake_mode.from_tensor(y)
    inner_warmup = inner_compiled(fake_x, fake_y)
    print(f"  Inner warmup result shape: {inner_warmup.shape}")

print(f"  Cache entries: {len(_invoke_subgraph_cache)}")
print()


# Step 3: Define an outer function that calls the inner compiled function
def outer_fn_for_compile(x, y):
    z = x + 1
    result = inner_compiled(z, y)
    return result * 2


# Step 4: Compile the outer function - this should detect the inner compilation
#         and use invoke_subgraph instead of re-inlining
print("Step 2: Compiling outer function (should use invoke_subgraph for inner)...")

# Use a custom backend to capture the graph
captured_graphs = []

def capture_backend(gm, example_inputs):
    global captured_graphs
    captured_graphs.append(gm)
    print(f"\n  Captured graph #{len(captured_graphs)}:")
    print(gm.code)
    # Wrap with torch._dynamo.disable to prevent re-tracing during execution
    return torch._dynamo.disable(gm.forward)

outer_compiled = torch.compile(outer_fn_for_compile, backend=capture_backend, fullgraph=True)

try:
    # Run the outer compiled function - this triggers compilation
    result = outer_compiled(x, y)

    if captured_graphs:
        print(f"\nAnalyzing captured graphs ({len(captured_graphs)} total):")
        for i, graph in enumerate(captured_graphs):
            print(f"\n  Graph #{i+1}:")
            found_invoke_subgraph = False
            for node in graph.graph.nodes:
                if node.op == "call_function":
                    if "invoke_subgraph" in str(node.target):
                        found_invoke_subgraph = True
                        print(f"    ✓ Found invoke_subgraph node: {node}")
                        print(f"      Target: {node.target}")
                        print(f"      Args: {node.args}")

            if not found_invoke_subgraph:
                print("    ✗ No invoke_subgraph node found")
                print("    All call_function nodes:")
                for node in graph.graph.nodes:
                    if node.op == "call_function":
                        print(f"      {node}: {node.target}")

    # Verify correctness
    print("\nVerifying execution:")
    expected = (x + 1) * 2 + y  # z = x + 1, then z * 2 + y
    expected = expected * 2  # final * 2
    print(f"  Expected shape: {expected.shape}")
    # Handle tuple result
    actual = result[0] if isinstance(result, tuple) else result
    print(f"  Actual shape: {actual.shape}")
    print(f"  Values match: {torch.allclose(actual, expected)}")

except Exception as e:
    import traceback
    print(f"\n✗ Error: {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Done")
print("=" * 60)
