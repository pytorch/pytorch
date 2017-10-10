
def _check_inplace(trace):
    """Checks that all PythonOps that were not translated into JIT format are out of place.

    Should be run after the ONNX pass.
    """
    graph = trace.graph()
    for node in graph.nodes():
        if node.kind() == 'PythonOp':
            if node.i('__inplace'):
                raise RuntimeError("inplace {} not supported in the JIT".format(node.pyname()))
