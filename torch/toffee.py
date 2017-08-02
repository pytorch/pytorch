def op(op_name, *args, **kwargs):
    """A primitive operator

    TODO: better docs here

    TODO: This doesn't actually do an operation, eventually we want it to (and
    trace correctly).  DO NOT rely on this returning a dictionary!!!
    """
    return { "name": op_name, "inputs": args, "attrs": kwargs }
