try:
    import z3  # type: ignore[import]

    HAS_Z3 = True
    # dynamic type
    dyn = z3.DeclareSort("Dyn")
    dyn_type = z3.Const("dyn", dyn)

    # dimension
    dim = z3.Datatype("dim")
    dim.declare("dim", ("0", z3.IntSort()), ("1", z3.IntSort()))
    dim = dim.create()

    # tensors
    tensor_type = z3.Datatype("TensorType")
    tensor_type.declare("Dyn", ("dyn", dyn))
    tensor_type.declare("tensor1", ("0", dim))
    tensor_type.declare("tensor2", ("0", dim), ("1", dim))
    tensor_type.declare("tensor3", ("0", dim), ("1", dim), ("2", dim))
    tensor_type.declare("tensor4", ("0", dim), ("1", dim), ("2", dim), ("3", dim))
    tensor_type = tensor_type.create()

    # create dimension
    D = dim.dim

    z3_dyn = tensor_type.Dyn(dyn_type)


except ImportError:
    HAS_Z3 = False
