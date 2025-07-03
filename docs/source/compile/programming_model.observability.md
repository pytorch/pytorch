(programming_model.observability)=

# tlparse / TORCH_TRACE

tlparse / `TORCH_TRACE` are a pair of tools that produce compilation reports that look [like this](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html).

Traces are very easy to collect. To collect a trace, run your model like so:

```bash
TORCH_TRACE="/tmp/tracedir" python foo.py
pip install tlparse
tlparse /tmp/tracedir
```

This approach works even if you are running a distributed job, providing a trace for each rank.
It will open your browser with HTML similar to what’s generated above.
If you are making a bug report for a complicated problem that you don’t have a standalone reproduction for,
you can still greatly assist PyTorch developers by attaching the trace log generated in `/tmp/tracedir`.

```{warning}
The trace log contains all of your model code.
Do not share the trace log if the model you are working on is sensitive. The trace log does NOT contain weights.
```

```{raw} html
    <style>
        .red {background-color:#ff0000;}
        .green {background-color:#00ff00;}
        .dark-green {background-color:#027f02;}
    </style>
```

```{eval-rst}
.. role:: red

.. role:: green

.. role:: dark-green
```

The output of `tlparse` is primarily aimed for PyTorch developers,
and the log format is easy to upload and share on GitHub.
However,  as a non-PyTorch developer, you can still extract useful information from it.
We recommend starting with the inline help text in the report, which explains its contents.
Here are some insights you can gain from a `tlparse`:

- What model code was compiled by looking at the stack trie?
  This is especially useful if you're not familiar with the codebase being compiled!
- How many graph breaks / distinct compilation regions are there?
  (Each distinct compile is its own color coded block like {dark-green}`[0/0]`).
  Frames that are potentially graph-broken are light green {green}`[2/4]`.
  If there are a lot of frames, that is suspicious, and suggests that you had some catastrophic graph breaks,
  or maybe your code isn't a good match for `torch.compile`.
- How many times did I recompile a particular frame? Something that recompiled a lot will look like:
  {dark-green}`[10/0]` {dark-green}`[10/1]` {dark-green}`[10/2]`
  \- if something is being recompiled a lot, that is very suspicious and worth looking into, even if it isn't the root cause of your problem.
- Was there a compilation error? Frames that errored will look like {red}`[0/1]`.
- What intermediate compiler products did I generate for a given frame?
  For example, you can look at the high-level generated FX graph or the generated Triton code.
- Is there relevant information for a particular frame? You can find these in `compilation_metrics`.

## TORCH_LOGS

You can use the `TORCH_LOGS` environment variable to selectively enable parts of the `torch.compile` stack to log.
`TORCH_LOGS` is in fact the source of logs for `tlparse`. The format of the `TORCH_LOGS` environment variable looks like this:

```bash
TORCH_LOGS="<option1>,<option2>,..." python foo.py
```

Useful high-level options include:

- `graph_breaks`: logs locations of graph breaks in user code and the reason for the graph break
- `guards`: logs guards that are generated
- `recompiles`: logs which function recompiled and the guards that failed, leading to the recompilation
- `dynamic`: logs related to dynamic shapes

Also, you can programmatically set logging options using `torch._logging.set_logs`:

```python
import logging
torch._logging.set_logs(graph_breaks=True, dynamic=logging.DEBUG)
```

More `TORCH_LOGS` options are {ref}`troubleshooting-torch-logs-options`.
For the full list of options, see [torch.\_logging](https://pytorch.org/docs/stable/logging.html)
and [torch.\_logging.set_logs](https://pytorch.org/docs/stable/generated/torch._logging.set_logs.html#torch._logging.set_logs).

## tlparse vs. TORCH_LOGS

Generally, we suggest first using `tlparse` when encountering issues.
`tlparse` is ideal for debugging large models and gaining a high-level overview of how your model was compiled.
On the other hand, `TORCH_LOGS` is preferred for small examples and fine-grained debugging detail,
when we already have an idea of which `torch.compile` component is causing the problem.
