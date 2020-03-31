import torch

# Utility functions

def _as_tuple(inp, arg_name, fn_name):
    # Ensures that inp is a tuple of Tensors
    # Returns whether or not the original inp was a tuple and the tupled version of the input
    is_inp_tuple = True
    if not isinstance(inp, tuple):
        inp = (inp,)
        is_inp_tuple = False

    for i, el in enumerate(inp):
        if not torch.is_tensor(el):
            if is_inp_tuple:
                raise TypeError("The {} given to {} must be either a Tensor or a tuple of Tensors but the"
                                " value at index {} has type {}.".format(arg_name, fn_name, i, type(el)))
            else:
                raise TypeError("The {} given to {} must be either a Tensor or a tuple of Tensors but the"
                                " given {} has type {}.".format(arg_name, fn_name, arg_name, type(el)))

    return is_inp_tuple, inp

def _tuple_postprocess(res, to_unpack):
    # Unpacks a potentially nested tuple of Tensors
    # to_unpack should be a single boolean or a tuple of two booleans.
    # It is used to:
    # - invert _as_tuple when res should match the inp given to _as_tuple
    # - optionally remove nesting of two tuples created by multiple calls to _as_tuple
    if isinstance(to_unpack, tuple):
        assert len(to_unpack) == 2
        if not to_unpack[1]:
            res = tuple(el[0] for el in res)
        if not to_unpack[0]:
            res = res[0]
    else:
        if not to_unpack:
            res = res[0]
    return res

def _grad_preprocess(inputs, create_graph, need_graph):
    # Preprocess the inputs to make sure they require gradient
    # inputs is a tuple of Tensors to preprocess
    # create_graph specifies if the user wants gradients to flow back to the Tensors in inputs
    # need_graph specifies if we internally want gradients to flow back to the Tensors in res
    # Note that we *always* create a new Tensor object to be able to see the difference between
    # inputs given as arguments and the same Tensors automatically captured by the user function.
    # Check this issue for more details on how that can happen: https://github.com/pytorch/pytorch/issues/32576
    res = []
    for inp in inputs:
        if create_graph and inp.requires_grad:
            # Create at least a new Tensor object in a differentiable way
            if not inp.is_sparse:
                # Use .view_as() to get a shallow copy
                res.append(inp.view_as(inp))
            else:
                # We cannot use view for sparse Tensors so we clone
                res.append(inp.clone())
        else:
            res.append(inp.detach().requires_grad_(need_graph))
    return tuple(res)


def _grad_postprocess(inputs, create_graph):
    # Postprocess the generated Tensors to avoid returning Tensors with history when the user did not
    # request it.
    if torch.is_tensor(inputs[0]):
        if not create_graph:
            return tuple(inp.detach() for inp in inputs)
        else:
            return inputs
    else:
        return tuple(_grad_postprocess(inp, create_graph) for inp in inputs)

def _validate_v(v, other, is_other_tuple):
    # This assumes that other is the correct shape, and v should match
    # Both are assumed to be tuples of Tensors
    if len(other) != len(v):
        if is_other_tuple:
            raise RuntimeError("v is a tuple of invalid length: should be {} but got {}.".format(len(other), len(v)))
        else:
            raise RuntimeError("The given v should contain a single Tensor.")

    for idx, (el_v, el_other) in enumerate(zip(v, other)):
        if el_v.size() != el_other.size():
            prepend = ""
            if is_other_tuple:
                prepend = "Entry {} in ".format(idx)
            raise RuntimeError("{}v has invalid size: should be {} but got {}.".format(
                               prepend, el_other.size(), el_v.size()))


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
                raise RuntimeError("While computing the hessian, found that the jacobian of the user-provided"
                                   " function with respect to input {} is independent of the input. This is not"
                                   " allowed in strict mode. You should ensure that your function is twice"
                                   " differentiable and that the jacobian depends on the inputs (this would be"
                                   " violated by a linear function for example).".format(i))
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
    assert isinstance(outputs, tuple)
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)
    assert isinstance(grad_outputs, tuple)
    assert len(outputs) == len(grad_outputs)

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

def _fill_in_zeros(grads, refs, strict, create_graph, stage):
    # Used to detect None in the grads and depending on the flags, either replace them
    # with Tensors full of 0s of the appropriate size based on the refs or raise an error.
    # strict and create graph allow us to detect when it is appropriate to raise an error
    # stage gives us information of which backward call we consider to give good error message
    if stage not in ["back", "back_trick", "double_back", "double_back_trick"]:
        raise RuntimeError("Invalid stage argument '{}' to _fill_in_zeros".format(stage))

    res = tuple()
    for i, grads_i in enumerate(grads):
        if grads_i is None:
            if strict:
                if stage == "back":
                    raise RuntimeError("The output of the user-provided function is independent of "
                                       "input {}. This is not allowed in strict mode.".format(i))
                elif stage == "back_trick":
                    raise RuntimeError("The gradient with respect to the input is independent of entry {}"
                                       " in the grad_outputs when using the double backward trick to compute"
                                       " forward mode gradients. This is not allowed in strict mode.".format(i))
                elif stage == "double_back":
                    raise RuntimeError("The jacobian of the user-provided function is independent of "
                                       "input {}. This is not allowed in strict mode.".format(i))
                else:
                    raise RuntimeError("The hessian of the user-provided function is independent of "
                                       "entry {} in the grad_jacobian. This is not allowed in strict "
                                       "mode as it prevents from using the double backward trick to "
                                       "replace forward mode AD.".format(i))

            grads_i = torch.zeros_like(refs[i])
        else:
            if strict and create_graph and not grads_i.requires_grad:
                if "double" not in stage:
                    raise RuntimeError("The jacobian of the user-provided function is independent of "
                                       "input {}. This is not allowed in strict mode when create_graph=True.".format(i))
                else:
                    raise RuntimeError("The hessian of the user-provided function is independent of "
                                       "input {}. This is not allowed in strict mode when create_graph=True.".format(i))

        res += (grads_i,)

    return res

# Public API

def vjp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Function that computes the dot product between a vector ``v`` and the Jacobian of
    the given function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector Jacobian product is computed.
            Must be the same size as the output of ``func``. This argument is optional when
            ``func``'s output contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            vjp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        func_output (tuple of Tensors or Tensor): output of ``func(inputs)``
        vjp (tuple of Tensors or Tensor): result of the dot product with the same shape
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

    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vjp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    outputs = func(*inputs)
    is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "vjp")
    _check_requires_grad(outputs, "outputs", strict=strict)

    if v is not None:
        _, v = _as_tuple(v, "v", "vjp")
        v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
        _validate_v(v, outputs, is_outputs_tuple)
    else:
        if len(outputs) != 1 or outputs[0].nelement() != 1:
            raise RuntimeError("The vector v can only be None if the user-provided function returns "
                               "a single Tensor with a single element.")

    grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_graph)

    vjp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "back")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    vjp = _grad_postprocess(vjp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(vjp, is_inputs_tuple)


def jvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Function that computes the dot product between  the Jacobian of
    the given function at the point given by the inputs and a vector ``v``.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Jacobian vector product is computed. Must be the
            same size as the input of ``func``. This argument is optional when
            ``func``'s input contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            jvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        func_output (tuple of Tensors or Tensor): output of ``func(inputs)``
        jvp (tuple of Tensors or Tensor): result of the dot product with the same shape
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

        The jvp is currently computed by using the backward of the backward (sometimes called the double
        backwards trick) as we don't have support for forward mode AD in PyTorch at the moment.
    """

    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jvp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    if v is not None:
        _, v = _as_tuple(v, "v", "jvp")
        v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
        _validate_v(v, inputs, is_inputs_tuple)
    else:
        if len(inputs) != 1 or inputs[0].nelement() != 1:
            raise RuntimeError("The vector v can only be None if the input to the user-provided function "
                               "is a single Tensor with a single element.")

    outputs = func(*inputs)
    is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jvp")
    _check_requires_grad(outputs, "outputs", strict=strict)
    # The backward is linear so the value of grad_outputs is not important as it won't appear in the double
    # backward graph. We only need to ensure that it does not contain inf or nan.
    grad_outputs = tuple(torch.zeros_like(out, requires_grad=True) for out in outputs)

    grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
    _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

    grad_res = _autograd_grad(grad_inputs, grad_outputs, v, create_graph=create_graph)

    jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    jvp = _grad_postprocess(jvp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(jvp, is_outputs_tuple)


def jacobian(func, inputs, create_graph=False, strict=False):
    r"""Function that computes the Jacobian of a given function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Jacobian will be computed in
            a differentiable manner. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors) if there are a single input
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

    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    outputs = func(*inputs)
    is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jacobian")
    _check_requires_grad(outputs, "outputs", strict=strict)

    jacobian = tuple()
    for i, out in enumerate(outputs):

        jac_i = tuple([] for _ in range(len(inputs)))
        for j in range(out.nelement()):
            vj = _autograd_grad((out.reshape(-1)[j],), inputs, retain_graph=True, create_graph=create_graph)

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

    return _tuple_postprocess(jacobian, (is_outputs_tuple, is_inputs_tuple))


def hessian(func, inputs, create_graph=False, strict=False):
    r"""Function that computes the Hessian of a given scalar function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Hessian will be computed in
            a differentiable manner. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            hessian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        Hessian (Tensor or a tuple of tuple of Tensors) if there are a single input,
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

    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hessian")

    def ensure_single_output_function(*inp):
        out = func(*inp)
        is_out_tuple, t_out = _as_tuple(out, "outputs of the user-provided function", "hessian")
        _check_requires_grad(t_out, "outputs", strict=strict)

        if is_out_tuple or not torch.is_tensor(out):
            raise RuntimeError("The function given to hessian should return a single Tensor")

        if out.nelement() != 1:
            raise RuntimeError("The Tensor returned by the function given to hessian should contain a single element")

        return out.squeeze()

    def jac_func(*inp):
        jac = jacobian(ensure_single_output_function, inp, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)
        return jac

    res = jacobian(jac_func, inputs, create_graph=create_graph, strict=strict)
    return _tuple_postprocess(res, (is_inputs_tuple, is_inputs_tuple))


def vhp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Function that computes the dot product between a vector ``v`` and the
    Hessian of a given scalar function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector Hessian product is computed. Must be the
            same size as the input of ``func``. This argument is optional when
            ``func``'s input contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            vhp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        func_output (tuple of Tensors or Tensor): output of ``func(inputs)``
        vhp (tuple of Tensors or Tensor): result of the dot product with the same shape
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

    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vhp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    if v is not None:
        _, v = _as_tuple(v, "v", "vhp")
        v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
        _validate_v(v, inputs, is_inputs_tuple)
    else:
        if len(inputs) != 1 or inputs[0].nelement() != 1:
            raise RuntimeError("The vector v can only be None if the input to the user-provided function "
                               "is a single Tensor with a single element.")

    outputs = func(*inputs)
    is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "vhp")
    _check_requires_grad(outputs, "outputs", strict=strict)

    if is_outputs_tuple or not torch.is_tensor(outputs[0]):
        raise RuntimeError("The function given to vhp should return a single Tensor")

    if outputs[0].nelement() != 1:
        raise RuntimeError("The Tensor returned by the function given to vhp should contain a single element")

    jac = _autograd_grad(outputs, inputs, create_graph=True)
    _check_requires_grad(jac, "jacobian", strict=strict)

    grad_res = _autograd_grad(jac, inputs, v, create_graph=create_graph)

    vhp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "double_back")

    outputs = _grad_postprocess(outputs, create_graph)
    vhp = _grad_postprocess(vhp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(vhp, is_inputs_tuple)


def hvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Function that computes the dot product between the Hessian of a given scalar
    function and a vector ``v`` at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Hessian vector product is computed. Must be the
            same size as the input of ``func``. This argument is optional when
            ``func``'s input contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            hvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        func_output (tuple of Tensors or Tensor): output of ``func(inputs)``
        hvp (tuple of Tensors or Tensor): result of the dot product with the same shape
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
        know that your function satisfies this condition, you should use vhp instead that is
        much faster with the current implementation.

    """

    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hvp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    if v is not None:
        _, v = _as_tuple(v, "v", "hvp")
        v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
        _validate_v(v, inputs, is_inputs_tuple)
    else:
        if len(inputs) != 1 or inputs[0].nelement() != 1:
            raise RuntimeError("The vector v can only be None if the input to the user-provided function "
                               "is a single Tensor with a single element.")

    outputs = func(*inputs)
    is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "hvp")
    _check_requires_grad(outputs, "outputs", strict=strict)

    if is_outputs_tuple or not torch.is_tensor(outputs[0]):
        raise RuntimeError("The function given to hvp should return a single Tensor")

    if outputs[0].nelement() != 1:
        raise RuntimeError("The Tensor returned by the function given to hvp should contain a single element")

    jac = _autograd_grad(outputs, inputs, create_graph=True)
    _check_requires_grad(jac, "jacobian", strict=strict)

    grad_jac = tuple(torch.zeros_like(inp, requires_grad=True) for inp in inputs)

    double_back = _autograd_grad(jac, inputs, grad_jac, create_graph=True)
    _check_requires_grad(jac, "hessian", strict=strict)

    grad_res = _autograd_grad(double_back, grad_jac, v, create_graph=create_graph)

    hvp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "double_back_trick")

    outputs = _grad_postprocess(outputs, create_graph)
    hvp = _grad_postprocess(hvp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(hvp, is_inputs_tuple)
