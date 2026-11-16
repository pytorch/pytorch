# Custom Operators

**Summary:**
- Use custom operators to have `torch.compile` treat a function as opaque. `torch.compile` will never trace into the function and Inductor (the backend) will run the function as-is.

You may wish to use a custom operator in any of the following situations:
- Your code calls some C/C++/CUDA code. Dynamo is a Python bytecode interpreter and generally does not know how to handle calls to C/C++/CUDA functions that are bound to Python.
- Dynamo and non-strict tracing have trouble tracing through a function and you want it to be ignored by `torch.compile`.

Please see [the Python custom ops tutorial](https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial)for more details on how to wrap a Python function into a `torch.compile`-understood custom operator.

For more advanced use cases, you may wish to use our C++ Custom Operator API; please see [here](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html) for more information.
