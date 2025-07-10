(programming_model.graph_breaks_index)=
# Working with Graph Breaks

Recall from (Dynamo Core Concepts)[programming_model.dynamo_core_concepts] that Dynamo graph breaks when
encountering code that can't be traced. In the default `torch.compile` settings, Dynamo compiles the FX graph
that has been determined so far, runs the unsupported code in regular Python, then resumes tracing after the unsupported code.

Graph breaks are a feature that allows Dynamo to run over arbitrary Python code and carve out functional
subgraphs that can each be individually optimized.

However, it is possible for graph breaks to lead to unexpected slowness in `torch.compile`.
If you're not getting the speedups you expect, we recommend checking for graph breaks and removing them.

The following sections describe some approaches to dealing with graph breaks.

```{toctree}
programming_model.fullgraph_true
programming_model.common_graph_breaks
```
