"""
This exercise is a gentle introduction to the Torchscript compiler in PyTorch.
Deep learning compilers can perform graph and tensor level optimizations and can
generate efficient machine code.  In this example, we test this hypothesis. We
will not go into the details of how deep learning compilers work.  Instead, we
will just see that compilers can actually lead to speedups.
After this exercise, you will learn following things
    * See that compilers can actually lead to speedup.
    * Understand how to use Torchscript to compile your model.
    * Understand how to use Torchscript with NVFuser to compiler your model.
"""
import torch
import time


# For running NVFuser, we will change this to 'cuda'
device = "cuda"

def timeme(fn, string_id):
    # Lets create our input tensor
    a = torch.randn(1024, device=device)

    # Lets warmup. Warmup is important for Torchscript because it performs
    # compilation on the first invocation and saves the compiled code to be used
    # for later invocations.
    n_warmup = 100
    time1 = time.time()
    for _ in range(0, n_warmup):
        a = fn(a)
    time2 = time.time()

    n_repeats = 100
    time1 = time.time()
    for _ in range(0, n_repeats):
        a = fn(a)
    time2 = time.time()

    total_time = (time2 - time1) * 10**3
    print(string_id, total_time, "ms")
    return total_time


# Let's create a simple unrealistic function. Its almost unlikely that you will
# see this type of function in real models. But, for the sake of simplicity of
# this exercise, we will create a model that has 10 cascaded Relu operations.
def fn(a):
    for _ in range(10):
        a = torch.nn.functional.relu(a)
    return a


# Our baseline is PyTorch execution. PyTorch is eager-first framework, i.e., it
# runs each op one-by-one. People use different names for PyTorch baseline. Some
# of them are PyTorch performance and PyTorch eager performance.
eager_baseline_time = timeme(fn, "PyTorch eager mode")

# Lets use the torchscript compiler. Torchscript compiler performs graph and
# tensor level optimizations for our fn. Interestingly, Torchscript is built in
# a manner so that it is compatible with multiple backend compilers. In this
# exercise, we will look at two of them - 1) NNC and 2) Nvidia NVFuser

# Torscript + NNC
# Lets compile our function with Torchscript + NNC compiler. NNC is our default
# backend compiler, so we do not need any specific argument to configure NNC
# backend. The compilation is as simple as the following line
a = torch.randn(1024, device=device)
torchscript_nnc_fn = torch.jit.trace(fn, (a,))
# You can run this compiled function similar to PyTorch eager function. Note
# that the user experience does not change. User has to add just one and the
# rest of the program stays like before.
torchscript_nnc_fn(a)

# Lets run the torchscript compiled function. We observe significant speedup.
torchscript_nnc_time = timeme(torchscript_nnc_fn, "PyTorch Torchscript NNC")
print("Speedup of Torchscript+NNC over PyTorch eager =", round(eager_baseline_time/torchscript_nnc_time, 2))


# Torscript + NVFuser
# Lets now compile with NVFuser backend. Nvidia has been building compiler tools
# to support Nvidia GPUs. NVfuser is one example of these tools which is pretty
# efficient for operator fusion. Since, this is not default, we have to
# configure Torchscript to use NVFuser. Note that it is probable that we make
# NVFuser as default backend for GPUs in future. In that case, this exercise
# will have to be modified.
torchscript_nvfuser_fn = torch.jit.script(fn)
with torch.jit.fuser("fuser2"): # This is how we invoke NVFuser
    torchscript_nvfuser_fn(a)

# Lets run the torchscript compiled function. We observe significant speedup here as well.
with torch.jit.fuser("fuser2"):
    torchscript_nvfuser_time = timeme(torchscript_nvfuser_fn, "PyTorch Torchscript NVFuser")
print("Speedup of Torchscript+NVFuser over PyTorch eager =", round(eager_baseline_time/torchscript_nvfuser_time, 2))


# Conclusion
# Hope this exercise piques your interest in deep learning compilers.
# We will drill down into different IRs in the next exercise.
