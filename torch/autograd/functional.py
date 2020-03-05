import torch

# Utility functions

def _as_tuple(inp, arg_name, fn_name):
    # Ensures that the inputs is a tuple of Tensors
    # Returns whether or not the original inp was a tuple and the tupled version of the input
    tuple_inp = True
    if not isinstance(inp, tuple):
        inp = (inp,)
        tuple_inp = False

    for i, el in enumerate(inp):
        if not torch.is_tensor(el):
            raise TypeError("The {} given to {} must be either a Tensor or a tuple of Tensors but the"
                            " value at index {} has type {}.".format(arg_name, fn_name, i, type(el)))

    return tuple_inp, inp

def _tuple_postprocess(inp, tuple_inp):
    # Unpacks a potentially nested tuple of Tensors
    # It is used to invert _as_tuple before returning to the user
    if isinstance(tuple_inp, tuple):
        assert len(tuple_inp) == 2
        if not tuple_inp[1]:
            inp = tuple(el[0] for el in inp)
        if not tuple_inp[0]:
            inp = inp[0]
    else:
        if not tuple_inp:
            inp = inp[0]
    return inp

def _grad_preprocess(inputs, create_graph, need_graph):
    # Preprocess the inputs to make sure they require gradient
    # inputs is a tuple of Tensor to preprocess
    # create_graph specifies if the user wants gradients to flow back to the Tensors in inputs
    # need_graph specifies if we internally want gradients to flow back to the Tensors in res
    # Note that we *always* create a new Tensor object to be able to see the difference between
    # inputs given as arguments and the same Tensors automatically captured by the user function.
    # Check this issue for more details on how that can happen: https://github.com/pytorch/pytorch/issues/32576
    res = []
    for inp in inputs:
        if create_graph and inp.requires_grad:
            # Use .view_as() to get a new Tensor in a differentiable way
            res.append(inp.view_as(inp))
        else:
            res.append(inp.detach().requires_grad_(need_graph))
    return tuple(res)


def _grad_postprocess(inputs, create_graph):
    # Postprocess the generated Tensors to avoid returning Tensors with history when the user does not
    # requested it.
    if torch.is_tensor(inputs[0]):
        if not create_graph:
            return tuple(inp.detach() for inp in inputs)
        else:
            return inputs
    else:
        return tuple(_grad_postprocess(inp, create_graph) for inp in inputs)

def _validate_v(v, other):
    # This assumes that other is the correct shape, and v should match
    # Both are assumed to be tuples of Tensors
    if len(other) != len(v):
        raise RuntimeError("v is a tuple of invalid length: should be {} but got {}.".format(len(other), len(v)))

    for idx, (el_v, el_other) in enumerate(zip(v, other)):
        if el_v.size() != el_other.size():
            raise RuntimeError("Entry {} in v has invalid size: should be {} but got {}.".format(
                               idx, el_other.size(), el_v.size()))

def _check_requires_grad(inputs, input_type, strict):
    # Used to make all the necessary checks to raise nice errors in strict mode.
    if not strict:
        return

    if input_type not in ["outputs", "grad_inputs", "jacobian", "hessian"]:
        raise RuntimeError("Invalid input_type to _check_requires_grad")
    for i, inp in enumerate(inputs):
        if inp is None:
            # This can only be reached for grad_inputs.
            raise RuntimeError("The output of the user-provided function is independent of input {}."
                               " This is not allowed in strict mode.".format(i))
        if not inp.requires_grad:
            if input_type == "hessian":
                raise RuntimeError("The hessian of the user-provided function with respect to input {}"
                                   " is independent of the input. This is not allowed in strict mode."
                                   " You should ensure that your function is thrice differentiable and that"
                                   " the hessian depends on the inputs.".format(i))
            elif input_type == "jacobian":
                raise RuntimeError("The jacobian of the user-provided function with respect to input {}"
                                   " is independent of the input. This is not allowed in strict mode."
                                   " You should ensure that your function is twice differentiable and that"
                                   " the jacobian depends on the inputs (this would be violated by a linear"
                                   " function).".format(i))
            elif input_type == "grad_inputs":
                raise RuntimeError("The gradient with respect to input {} is independent of the inputs of the"
                                   " user-provided function. This is not allowed in strict mode.".format(i))
            else:
                raise RuntimeError("Output {} of the user-provided function does not require gradients."
                                   " The outputs must be computed in a differentiable manner from the input"
                                   " when running in strict mode.".format(i))

def _autograd_grad(outputs, inputs, grad_outputs=None, create_graph=False, retain_graph=None):
    # Version of autograd.grad that accepts `None` in outputs and do not compute gradients for them.
    # This has the extra constraint that inputs has to be a tuple
    if torch.is_tensor(outputs):
        outputs = (outputs,)
        grad_outputs = (grad_outputs,)
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)

    new_outputs = tuple()
    new_grad_outputs = tuple()
    for out, grad_out in zip(outputs, grad_outputs):
        if out is not None and out.requires_grad:
            new_outputs += (out,)
            new_grad_outputs += (grad_out,)

    if len(new_outputs) == 0:
        # No differentiable output, we don't need to call the autograd engine
        return (None,) * len(inputs)
    else:
        return torch.autograd.grad(new_outputs, inputs, new_grad_outputs, allow_unused=True,
                                   create_graph=create_graph, retain_graph=retain_graph)

# Public API

def vjp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Function that computes the dot product between a vector ``v`` and the Jacobian of
    the given function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensor or a Tensor.
        inputs (tuple of Tensor or Tensor): inputs to the function ``func``.
        v (tuple of Tensor or Tensor): The vector that will multiply the Jacobian. Must be the
            same size as the output of ``func``. This argument is optional when
            ``func``'s output contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when strict is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return zeros as the vjp for
            said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple of Tensor or Tensor): output of ``func(inputs)``
        result (tuple of Tensor or Tensor): result of the dot product with the same shape
            as the inputs.

    Example::

        >>> def exp_reducer(x):
        ...   return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4)
        >>> vjp(exp_reducer, inputs, v)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782]),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]]))

        >>> vjp(exp_reducer, inputs, v, create_graph=True)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782], grad_fn=<SumBackward1>),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]], grad_fn=<MulBackward0>))

        >>> def adder(x, y):
        ...   return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = torch.ones(2)
        >>> vjp(adder, inputs, v)
        (tensor([2.4225, 2.3340]),
         (tensor([2., 2.]), tensor([3., 3.])))
    """

    tuple_inputs, inputs = _as_tuple(inputs, "inputs", "vjp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    outputs = func(*inputs)
    tuple_outputs, outputs = _as_tuple(outputs, "outputs of the user-provided function", "vjp")
    _check_requires_grad(outputs, "outputs", strict=strict)

    if v is not None:
        tuple_v, v = _as_tuple(v, "v", "vjp")
        v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
        _validate_v(v, outputs)
    else:
        if len(outputs) != 1 or outputs[0].nelement() != 1:
            raise RuntimeError("The vector v can only be None if the user-provided function returns "
                               "a single Tensor with a single element.")

    grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_graph)

    vjp = tuple()
    for i, vjp_i in enumerate(grad_res):
        if vjp_i is None:
            if strict:
                raise RuntimeError("The output of the user-provided function is independent of "
                                   "input {}. This is not allowed in strict mode.".format(i))
            vjp_i = torch.zeros_like(inputs[i])
        else:
            if strict and create_graph and not vjp_i.requires_grad:
                raise RuntimeError("The jacobian of the user-provided function is independent of "
                                   "input {}. This is not allowed in strict mode when create_graph=True.".format(i))
        vjp += (vjp_i,)

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    vjp = _grad_postprocess(vjp, create_graph)

    return _tuple_postprocess(outputs, tuple_outputs), _tuple_postprocess(vjp, tuple_inputs)


def jvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Function that computes the dot product between  the Jacobian of
    the given function at the point given by the inputs and a vector ``v``.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensor or a Tensor.
        inputs (tuple of Tensor or Tensor): inputs to the function ``func``.
        v (tuple of Tensor or Tensor): The vector that will multiply the Jacobian. Must be the
            same size as the input of ``func``. This argument is optional when
            ``func``'s input contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when strict is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return zeros as the jvp for
            said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple of Tensor or Tensor): output of ``func(inputs)``
        result (tuple of Tensor or Tensor): result of the dot product with the same shape
            as the output.

    Example::

        >>> def exp_reducer(x):
        ...   return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4, 4)
        >>> jvp(exp_reducer, inputs, v)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106]),
         tensor([6.3090, 4.6742, 7.9114, 8.2106]))

        >>> jvp(exp_reducer, inputs, v, create_graph=True)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SumBackward1>),
         tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SqueezeBackward1>))

        >>> def adder(x, y):
        ...   return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.ones(2), torch.ones(2))
        >>> jvp(adder, inputs, v)
        (tensor([2.2399, 2.5005]),
         tensor([5., 5.]))

    Note::

        The jvp is currently computed using the double backward trick as we don't have support for
        forward mode AD in pytorch at the moment.
    """

    tuple_inputs, inputs = _as_tuple(inputs, "inputs", "jvp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    if v is not None:
        tuple_v, v = _as_tuple(v, "v", "jvp")
        v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
        _validate_v(v, inputs)
    else:
        if len(inputs) != 1 or inputs[0].nelement() != 1:
            raise RuntimeError("The vector v can only be None if the input to the user-provided function "
                               "is a single Tensor with a single element.")

    outputs = func(*inputs)
    tuple_outputs, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jvp")
    _check_requires_grad(outputs, "outputs", strict=strict)
    # The backward is linear so the value of grad_outputs is not important as it won't appear in the double
    # backward graph. We only need to ensure that it does not contain inf or nan.
    grad_outputs = tuple(torch.zeros_like(out, requires_grad=True) for out in outputs)

    grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
    _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

    grad_res = _autograd_grad(grad_inputs, grad_outputs, v, create_graph=create_graph)

    jvp = tuple()
    for i, jvp_i in enumerate(grad_res):
        if jvp_i is None:
            if strict:
                raise RuntimeError("The gradient with respect to the input is independent of entry {}"
                                   " in the grad_outputs when using the double backward trick to compute"
                                   " forward mode gradients. This is not allowed in strict mode.".format(i))
            jvp_i = torch.zeros_like(outputs[i])
        else:
            if strict and create_graph and not jvp_i.requires_grad:
                raise RuntimeError("The jacobian of the user-provided function is independent of "
                                   "input {}. This is not allowed in strict mode when create_graph=True.".format(i))
        jvp += (jvp_i,)

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    jvp = _grad_postprocess(jvp, create_graph)

    return _tuple_postprocess(outputs, tuple_outputs), _tuple_postprocess(jvp, tuple_outputs)


def jacobian(func, inputs, create_graph=False, strict=False):
    r"""Function that computes the Jacobian of a given function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensor or a Tensor.
        inputs (tuple of Tensor or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Jacobian will be computed in
            a differentiable manner. Note that when strict is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return zeros as the jacobian for
            said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensor) if there are a single input
            and output, this will be a single Tensor containing the Jacobian for the
            linearized inputs and output. If one of the two is a tuple, then the Jacobian
            will be a tuple of Tensors. If both of them are tuples, then the Jacobian will
            be a tuple of tuple of Tensors where ``Jacobian[i][j]`` will contain the Jacobian
            of the ``i``th output and ``j``th input and will have as size the concatenation of the
            sizes of the corresponding output and the corresponding input.

    Example::

        >>> def exp_reducer(x):
        ...   return x.exp().sum(dim=1)
        >>> inputs = torch.rand(2, 2)
        >>> jacobian(exp_reducer, inputs)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],

                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]])

        >>> jacobian(exp_reducer, inputs, create_graph=True)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],

                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]], grad_fn=<ViewBackward>)

        >>> def exp_adder(x, y):
        ...   return 2 * x.exp() + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> jacobian(exp_adder, inputs)
        (tensor([[2.8052, 0.0000],
                [0.0000, 3.3963]]),
         tensor([[3., 0.],
                 [0., 3.]]))
    """

    tuple_inputs, inputs = _as_tuple(inputs, "inputs", "jacobian")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    outputs = func(*inputs)
    tuple_outputs, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jacobian")
    _check_requires_grad(outputs, "outputs", strict=strict)

    jacobian = tuple()
    for i, out in enumerate(outputs):

        jac_i = tuple([] for _ in range(len(inputs)))
        for j in range(out.nelement()):
            # Do this instead of the grad_output trick in `gradcheck` to avoid version counter issues
            vj = _autograd_grad(out.reshape(-1)[j], inputs, retain_graph=True, create_graph=create_graph)

            for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(zip(jac_i, vj, inputs)):
                if vj_el is not None:
                    if strict and create_graph and not vj_el.requires_grad:
                        raise RuntimeError("The jacobian of the user-provided function is independent of "
                                           "input {}. This is not allowed in strict mode when create_graph=True.".format(i))
                    jac_i_el.append(vj_el)
                else:
                    if strict:
                        raise RuntimeError("Output {} of the user-provided function is independent of "
                                           "input {}. This is not allowed in strict mode.".format(i, el_idx))
                    jac_i_el.append(torch.zeros_like(inp_el))

        jacobian += (tuple(torch.stack(jac_i_el, dim=0).view(out.size()
                     + inputs[el_idx].size()) for (el_idx, jac_i_el) in enumerate(jac_i)), )

    jacobian = _grad_postprocess(jacobian, create_graph)

    return _tuple_postprocess(jacobian, (tuple_outputs, tuple_inputs))


def hessian(func, inputs, create_graph=False, strict=False):
    r"""Function that computes the Hessian of a given scalar function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensor or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Hessian will be computed in
            a differentiable manner. Note that when strict is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return zeros as the hessian for
            said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        Hessian (Tensor or a tuple of tuple of Tensor) if there are a single input,
            this will be a single Tensor containing the Hessian for the input.
            If it is a tuple, then the Hessian will be a tuple of tuples where
            ``Hessian[i][j]`` will contain the Hessian of the ``i``th input
            and ``j``th input with size the sum of the size of the ``i``th input plus
            the size of the ``j``th input.

    Example::

        >>> def pow_reducer(x):
        ...   return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> hessian(pow_reducer, inputs)
        tensor([[[[5.2265, 0.0000],
                  [0.0000, 0.0000]],

                 [[0.0000, 4.8221],
                  [0.0000, 0.0000]]],


                [[[0.0000, 0.0000],
                  [1.9456, 0.0000]],

                 [[0.0000, 0.0000],
                  [0.0000, 3.2550]]]])

        >>> hessian(pow_reducer, inputs, create_graph=True)
        tensor([[[[5.2265, 0.0000],
                  [0.0000, 0.0000]],

                 [[0.0000, 4.8221],
                  [0.0000, 0.0000]]],


                [[[0.0000, 0.0000],
                  [1.9456, 0.0000]],

                 [[0.0000, 0.0000],
                  [0.0000, 3.2550]]]], grad_fn=<ViewBackward>)


        >>> def pow_adder_reducer(x, y):
        ...   return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> hessian(pow_adder_reducer, inputs)
        ((tensor([[4., 0.],
                  [0., 4.]]),
          tensor([[0., 0.],
                  [0., 0.]])),
         (tensor([[0., 0.],
                  [0., 0.]]),
          tensor([[6., 0.],
                  [0., 6.]])))
    """

    tuple_inputs, inputs = _as_tuple(inputs, "inputs", "hessian")

    def ensure_single_output_function(*inp):
        out = func(*inp)
        tuple_out, t_out = _as_tuple(out, "outputs of the user-provided function", "hessian")
        _check_requires_grad(t_out, "outputs", strict=strict)

        if tuple_out or not torch.is_tensor(out):
            raise RuntimeError("The function given to hessian should return a single Tensor")

        if out.nelement() != 1:
            raise RuntimeError("The Tensor returned by the function given to hessian should contain a single element")

        return out.squeeze()

    def jac_func(*inp):
        jac = jacobian(ensure_single_output_function, inp, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)
        return jac

    res = jacobian(jac_func, inputs, create_graph=create_graph, strict=strict)
    return _tuple_postprocess(res, (tuple_inputs, tuple_inputs))


def vhp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Function that computes the dot product between a vector ``v`` and the
    Hessian of a given scalar function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensor or Tensor): inputs to the function ``func``.
        v (tuple of Tensor or Tensor): The vector that will multiply the Hessian. Must be the
            same size as the input of ``func``. This argument is optional when
            ``func``'s input contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when strict is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return zeros as the vhp for
            said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple of Tensor or Tensor): output of ``func(inputs)``
        result (tuple of Tensor or Tensor): result of the dot product with the same shape
            as the inputs.

    Example::

        >>> def pow_reducer(x):
        ...   return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> vhp(pow_reducer, inputs, v)
       (tensor(0.5591),
        tensor([[1.0689, 1.2431],
                [3.0989, 4.4456]]))

        >>> vhp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.5591, grad_fn=<SumBackward0>),
         tensor([[1.0689, 1.2431],
                 [3.0989, 4.4456]], grad_fn=<MulBackward0>))


        >>> def pow_adder_reducer(x, y):
        ...   return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> vhp(pow_adder_reducer, inputs, v)
        (tensor(4.8053),
         (tensor([0., 0.]),
          tensor([6., 6.])))

    """

    tuple_inputs, inputs = _as_tuple(inputs, "inputs", "vhp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    if v is not None:
        tuple_v, v = _as_tuple(v, "v", "vhp")
        v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
        _validate_v(v, inputs)
    else:
        if len(inputs) != 1 or inputs[0].nelement() != 1:
            raise RuntimeError("The vector v can only be None if the input to the user-provided function "
                               "is a single Tensor with a single element.")

    outputs = func(*inputs)
    tuple_outputs, outputs = _as_tuple(outputs, "outputs of the user-provided function", "vhp")
    _check_requires_grad(outputs, "outputs", strict=strict)

    if tuple_outputs or not torch.is_tensor(outputs[0]):
        raise RuntimeError("The function given to vhp should return a single Tensor")

    if outputs[0].nelement() != 1:
        raise RuntimeError("The Tensor returned by the function given to vhp should contain a single element")

    jac = _autograd_grad(outputs, inputs, create_graph=True)
    _check_requires_grad(jac, "jacobian", strict=strict)

    grad_res = _autograd_grad(jac, inputs, v, create_graph=create_graph)

    vhp = tuple()
    for i, vhp_i in enumerate(grad_res):
        if vhp_i is None:
            if strict:
                raise RuntimeError("The jacobian of the user-provided function is independent of "
                                   "input {}. This is not allowed in strict mode.".format(i))
            vhp_i = torch.zeros_like(inputs[i])
        else:
            if strict and create_graph and not vhp_i.requires_grad:
                raise RuntimeError("The hessian of the user-provided function is independent of "
                                   "input {}. This is not allowed in strict mode when create_graph=True.".format(i))
        vhp += (vhp_i,)

    outputs = _grad_postprocess(outputs, create_graph)
    vhp = _grad_postprocess(vhp, create_graph)

    return _tuple_postprocess(outputs, tuple_outputs), _tuple_postprocess(vhp, tuple_inputs)


def hvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Function that computes the dot product between the Hessian of a given scalar
    function and a vector ``v`` at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensor or Tensor): inputs to the function ``func``.
        v (tuple of Tensor or Tensor): The vector that will multiply the Hessian. Must be the
            same size as the input of ``func``. This argument is optional when
            ``func``'s input contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when strict is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return zeros as the hvp for
            said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple of Tensor or Tensor): output of ``func(inputs)``
        result (tuple of Tensor or Tensor): result of the dot product with the same shape
            as the inputs.

    Example::

        >>> def pow_reducer(x):
        ...   return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> hvp(pow_reducer, inputs, v)
        (tensor(0.1448),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]]))

        >>> hvp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.1448, grad_fn=<SumBackward0>),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]], grad_fn=<MulBackward0>))


        >>> def pow_adder_reducer(x, y):
        ...   return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> hvp(pow_adder_reducer, inputs, v)
        (tensor(2.3030),
         (tensor([0., 0.]),
          tensor([6., 6.])))

    Note::

        This function is significantly slower than `vhp` due to backward mode AD constraints.
        If your functions is twice continuously differentiable, then hvp = vhp.t(). So if you
        know that your function verifies this condition, you should use vhp instead that is
        much faster with the current implementation.

    """

    tuple_inputs, inputs = _as_tuple(inputs, "inputs", "hvp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    if v is not None:
        tuple_v, v = _as_tuple(v, "v", "hvp")
        v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
        _validate_v(v, inputs)
    else:
        if len(inputs) != 1 or inputs[0].nelement() != 1:
            raise RuntimeError("The vector v can only be None if the input to the user-provided function "
                               "is a single Tensor with a single element.")

    outputs = func(*inputs)
    tuple_outputs, outputs = _as_tuple(outputs, "outputs of the user-provided function", "hvp")
    _check_requires_grad(outputs, "outputs", strict=strict)

    if tuple_outputs or not torch.is_tensor(outputs[0]):
        raise RuntimeError("The function given to hvp should return a single Tensor")

    if outputs[0].nelement() != 1:
        raise RuntimeError("The Tensor returned by the function given to hvp should contain a single element")

    jac = _autograd_grad(outputs, inputs, create_graph=True)
    _check_requires_grad(jac, "jacobian", strict=strict)

    grad_jac = tuple(torch.zeros_like(inp, requires_grad=True) for inp in inputs)

    double_back = _autograd_grad(jac, inputs, grad_jac, create_graph=True)
    _check_requires_grad(jac, "hessian", strict=strict)

    grad_res = _autograd_grad(double_back, grad_jac, v, create_graph=create_graph)

    hvp = tuple()
    for i, hvp_i in enumerate(grad_res):
        if hvp_i is None:
            if strict:
                raise RuntimeError("The hessian of the user-provided function is independent of "
                                   "entry {} in the grad_jacobian. This is not allowed in strict "
                                   "mode as it prevents form using the double backward trick to "
                                   "replace forward mode AD.".format(i))
            hvp_i = torch.zeros_like(inputs[i])
        else:
            if strict and create_graph and not hvp_i.requires_grad:
                raise RuntimeError("The hessian of the user-provided function is independent of "
                                   "input {}. This is not allowed in strict mode when create_graph=True.".format(i))
        hvp += (hvp_i,)

    outputs = _grad_postprocess(outputs, create_graph)
    hvp = _grad_postprocess(hvp, create_graph)

    return _tuple_postprocess(outputs, tuple_outputs), _tuple_postprocess(hvp, tuple_inputs)
