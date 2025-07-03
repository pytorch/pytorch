(programming_model.fullgraph_false)=

# Working with `fullgraph=False`
While `fullgraph=False` is the default `torch.compile` setting, the semantics of resuming compilation upon encountering a graph break are more difficult to understand.
You can find details on the `fullgraph=False` semantics in the subsections.

The strategy for using `torch.compile(fullgraph=False)` is as follows:

1. Determine the ideal location to place `torch.compile`. Normally, it is the highest-level function that doesn’t result in excessive graph breaks.
   Functions that do a lot of preprocessing or I/O operations are examples of functions that result in many graph breaks and do not significantly benefit from `torch.compile`.
2. Apply `torch.compiler.disable` to functions in the compiled region that result in a lot of graph breaks
   and do not benefit from compilation. In this case, one graph break is better than potentially tens or hundreds.
3. Use `TORCH_LOGS="graph_breaks"` or tlparse (TODO: link) to investigate remaining graph breaks.
   Work around these graph breaks using the same approaches as working around graph breaks under
   the `fullgraph=True` programming model. Not all graph breaks need to be removed - some may
   impact performance more than others. The general rule is to focus on graph breaks that are happening during model computation.
     a. Apply `set_fullgraph(True)` to code that is performance sensitive.
