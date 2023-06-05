Profiling to understand torch.compile performance
=================================================

What to use torch.profiler for:
-------------------------------

torch.profiler is helpful for understanding the performance of your program at a kernel-level granularity - for example, it can show graph breaks and GPU utilization at the level of the program. The data provided by the profiler can often help users understand where to investigate further to understand model performance.

To understand kernel-level performance, other toosl exist. NVIDIA's ncu tool can be used, or :ref:`inductor's profiling tools<TorchInductor GPU Profiling>``.

See also the :ref:`general pytorch profiler guide <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`.

Basics of using torch.profiler and viewing traces
-------------------------------------------------

**Example program**: We'll use this example of profiling resnet18. Notice the following parts of this example program:

* Include a warm-up run to wait for compilation to complete (this will warm up systems like the CUDA caching allocator)
* Use :code:`torch.profiler.profile()` context for profiling the section we are interested in
* Use :code:`prof.export_chrome_trace("trace.json")` to export the profiling artifact.

.. code-block:: python

    import torch
    from torchvision.models import resnet18

    model = resnet18().cuda()
    inputs = [torch.randn((5, 3, 224, 224), device='cuda') for _ in range(10)]

    model_c = torch.compile(model)

    def fwd_bwd(inp):
        out = model_c(inp)
        out.sum().backward()

    # warm up
    fwd_bwd(inputs[0])

    with torch.profiler.profile() as prof:
        for i in range(1, 4):
            fwd_bwd(inputs[i])
            prof.step()

    prof.export_chrome_trace("trace.json")

**Viewing chrome traces**: In the Chrome browser, open :ref:`chrome://tracing` and load the json file. Use the “w” and “s” keys to zoom in and out, and use “a” and “d” to scroll left and right. “?” will show a “help” screen with a list of shortcuts.

.. figure:: ../_static/img/profiling_torch_compile/basic_chrome_trace.png
    :alt: Example of a basic chrome trace, visualized in the chrome://tracing viewer

Here, we observe:
* CompiledFunction and CompiledFunctionBackward events, which correspond to the dynamo-compiled regions.
* CPU events at the top, and GPU events at the bottom.

**Flows between CPU and GPU events**

Every kernel on the GPU occurs after being launched by code running on the CPU. The profiler can draw connections (i.e. “flows”) between the GPU and CPU events to show which CPU event launched a GPU kernel. This is particularly helpful because, with a few exceptions, GPU kernels are launched asynchronously.

To view a flow connection, click on a GPU kernel and click “ac2g”:

.. figure:: ../_static/img/profiling_torch_compile/ac2g.png
    :alt: Visualization in the chrome://trace viewer, showing an async flow between a kernel and its launching location.

Alternatively, turn on *all* flows with the “Flow events” dropdown at the top.

Working around CUDA Graph profiling issues
------------------------------------------

When CUDA graphs are enabled, some cuda configurations (driver version under 525.85.12 or CUDA < 12)  can encounter issues between the profiling tools and CUDA graphs. To fix these issues, add an empty profiling context at the top of your program:

.. code-block:: python

    import torch

    torch.profiler._utils._init_for_cuda_graphs()

    # ... rest of program

Understanding compilation time
------------------------------

To understand why compilation is taking a long time, you can profile the first invocation of a torch.compile-ed program. Keep in mind that profile traces of compilations can be distorted more than typical profiling, because compilation workloads can be quite different from typical PyTorch workloads. In some cases, trace files may also be quite large. Traces > 1GB can be difficult to open with the chrome tracing tool.

Note: roughly the same information can also be obtained in non-graphical format with :code:`torch._dynamo.utils.compile_times()`. This utility won’t show when the compilation steps occur, but it will show the amount of time spent on each step - and times will not be affected by any profiling overhead.

See an example below:

.. code-block:: python

    import torch
    from torchvision.models import resnet18

    model = resnet18().cuda()
    inputs = [torch.randn((5, 3, 224, 224), device='cuda') for _ in range(10)]

    model_c = torch.compile(model)

    def fwd_bwd(inp):
        out = model_c(inp)
        out.sum().backward()

    def warmup_compile():
        def fn(x):
            return x.sin().relu()

        x = torch.rand((2, 2), device='cuda', requires_grad=True)
        fn_c = torch.compile(fn)
        out = fn_c(x)
        out.sum().backward()

    with torch.profiler.profile() as prof:
        with torch.profiler.record_function("warmup compile"):
            warmup_compile()

        with torch.profiler.record_function("resnet18 compile"):
            fwd_bwd(inputs[0])

    prof.export_chrome_trace("trace_compile.json")

.. figure:: ../_static/img/profiling_torch_compile/compilation_profiling.png
    :alt: A visualization in the chrome://trace viewer, showing dynamo and inductor compilation steps

Note a few things:

* The first invocation should occur *during* profiling in order to capture compilation
* Add a warm-up compilation in order to initialize any systems that need to be lazily initialized.

Finding graph breaks
--------------------

Although there are logging tools for identifying graph breaks, the profiler provides a quick visual method of identifying graph breaks.

When gradients are required for any inputs, graph breaks are easy to identify: each graph break will interrupt a CompiledFunction block, splitting it in two.

See the synthetic example below for a demonstration:

.. code-block:: python

    import torch
    import torch._dynamo

    class ModelWithBreaks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            def create_sequential():
                return torch.nn.Sequential(
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                )
            self.mod1 = create_sequential()
            self.mod2 = create_sequential()
            self.mod3 = create_sequential()
            self.mod4 = create_sequential()

        def forward(self, inp):
            mod1 = self.mod1(inp)
            torch._dynamo.graph_break()
            mod2 = self.mod2(mod1)
            torch._dynamo.graph_break()
            mod3 = self.mod3(mod2)
            torch._dynamo.graph_break()
            mod4 = self.mod4(mod3)
            return mod4


    model = ModelWithBreaks().cuda()
    inputs = [torch.randn((128, 128), device='cuda') for _ in range(10)]

    model_c = torch.compile(model)

    def fwd_bwd(inp):
        out = model_c(inp)
        out.sum().backward()

    # warm up
    fwd_bwd(inputs[0])

    with torch.profiler.profile() as prof:
        for i in range(1, 4):
            fwd_bwd(inputs[i])
            prof.step()

    prof.export_chrome_trace("trace_break.json")

.. figure:: ../_static/img/profiling_torch_compile/graph_breaks.png
    :alt: Visualization in the chrome://trace viewer, showing multiple CompiledFunction events - indicating graph breaks.

Launch overhead
---------------

One common issue is bad GPU utilization. A quick way to identify this is if there are large gaps between kernels on the GPU:

.. figure:: ../_static/img/profiling_torch_compile/cpu_bound.png
    :alt: Visualization in the chrome://trace viewer, showing large gaps between GPU kernels. This indicates that the model is CPU bound, likely due to overhead during kernel launches.

This is often the result of CPU overhead, e.g. if the amount of time spent on the CPU between kernel launches is larger than the amount of time spent by the GPU to process the kernels. The issue is more common for small batch sizes.

When using inductor, enabling CUDA graphs can often help improve performance when launch overhead is a concern.
