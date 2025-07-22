# Working with Graph Breaks

As you might remember from (Dynamo Core Concepts)[programming_model.dynamo_core_concepts] that Dynamo performs a graph break when
it encounters code that can't be traced. In the default `torch.compile` settings, Dynamo compiles the FX graph
that has been determined up to that point, executes the unsupported code in regular Python, and then resumes tracing.

Graph breaks enable Dynamo to trace through arbitrary Python code and carve out functional
subgraphs that can each be individually optimized.

However, graph breaks may cause unexpected slowness in `torch.compile`.
If you're not seeing the expected speedups, we recommend checking for graph breaks and removing them.

The following sections outline strategies for addressing graph breaks.

```{toctree}
programming_model.fullgraph_true
programming_model.common_graph_breaks
programming_model.dynamo_nonstrict_trace
programming_model.custom_ops
```
