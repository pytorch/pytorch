# torch.compile Programming Model

The `torch.compile` programming model:
1. Clarifies some internal behaviors of `torch.compile` so that one can better predict compiler behavior on user code and
2. Provides ways for one to take more fine-grained control over `torch.compile`.

By understanding the `torch.compile` programming model, one can systematically unblock themselves when encountering issues with `torch.compile`.

```{toctree}
programming_model.dynamo_core_concepts
programming_model.graph_breaks_index
programming_model.non_strict_tracing_model
```
