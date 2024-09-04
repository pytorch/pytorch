# mypy: allow-untyped-defs
from typing import List, Tuple

import torch
from torch._vmap_internals import _vmap

from . import forward_ad as fwAD


__all__ = ["vjp", "jvp", "jacobian", "hessian", "hvp", "vhp"]

# Utility functions


def _as_tuple_nocheck(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def _as_tuple(inp, arg_name=None, fn_name=None):
    # Ensures that inp is a tuple of Tensors
    # Returns whether or not the original inp was a tuple and the tupled version of the input
    if arg_name is None and fn_name is None:
        return _as_tuple_nocheck(inp)

    is_inp_tuple = True
    if not isinstance(inp, tuple):
        inp = (inp,)
        is_inp_tuple = False

    for i, el in enumerate(inp):
        if not isinstance(el, torch.Tensor):
            if is_inp_tuple:
                raise TypeError(
                    f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the"
                    f" value at index {i} has type {type(el)}."
                )
            else:
                raise TypeError(
                    f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the"
                    f" given {arg_name} has type {type(el)}."
                )

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
    if isinstance(inputs[0], torch.Tensor):
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
            raise RuntimeError(
                f"v is a tuple of invalid length: should be {len(other)} but got {len(v)}."
            )
        else:
            raise RuntimeError("The given v should contain a single Tensor.")

    for idx, (el_v, el_other) in enumerate(zip(v, other)):
        if el_v.size() != el_other.size():
            prepend = ""
            if is_other_tuple:
                prepend = f"Entry {idx} in "
            raise RuntimeError(
                f"{prepend}v has invalid size: should be {el_other.size()} but got {el_v.size()}."
            )


def _check_requires_grad(inputs, input_type, strict):
    # Used to make all the necessary checks to raise nice errors in strict mode.
    if not strict:
        return

    if input_type not in ["outputs", "grad_inputs", "jacobian", "hessian"]:
        raise RuntimeError("Invalid input_type to _check_requires_grad")
    for i, inp in enumerate(inputs):
        if inp is None:
            # This can only be reached for grad_inputs.
            raise RuntimeError(
                f"The output of the user-provided function is independent of input {i}."
                " This is not allowed in strict mode."
            )
        if not inp.requires_grad:
            if input_type == "hessian":
                raise RuntimeError(
                    f"The hessian of the user-provided function with respect to input {i}"
                    " is independent of the input. This is not allowed in strict mode."
                    " You should ensure that your function is thrice differentiable and that"
                    " the hessian depends on the inputs."
                )
            elif input_type == "jacobian":
                raise RuntimeError(
                    "While computing the hessian, found that the jacobian of the user-provided"
                    f" function with respect to input {i} is independent of the input. This is not"
                    " allowed in strict mode. You should ensure that your function is twice"
                    " differentiable and that the jacobian depends on the inputs (this would be"
                    " violated by a linear function for example)."
                )
            elif input_type == "grad_inputs":
                raise RuntimeError(
                    f"The gradient with respect to input {i} is independent of the inputs of the"
                    " user-provided function. This is not allowed in strict mode."
                )
            else:
                raise RuntimeError(
                    f"Output {i} of the user-provided function does not require gradients."
                    " The outputs must be computed in a differentiable manner from the input"
                    " when running in strict mode."
                )


def _autograd_grad(
    outputs,
    inputs,
    grad_outputs=None,
    create_graph=False,
    retain_graph=None,
    is_grads_batched=False,
):
    # Version of autograd.grad that accepts `None` in outputs and do not compute gradients for them.
    # This has the extra constraint that inputs has to be a tuple
    assert isinstance(outputs, tuple)
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)
    assert isinstance(grad_outputs, tuple)
    assert len(outputs) == len(grad_outputs)

    new_outputs: Tuple[torch.Tensor, ...] = ()
    new_grad_outputs: Tuple[torch.Tensor, ...] = ()
    for out, grad_out in zip(outputs, grad_outputs):
        if out is not None and out.requires_grad:
            new_outputs += (out,)
            new_grad_outputs += (grad_out,)

    if len(new_outputs) == 0:
        # No differentiable output, we don't need to call the autograd engine
        return (None,) * len(inputs)
    else:
        return torch.autograd.grad(
            new_outputs,
            inputs,
            new_grad_outputs,
            allow_unused=True,
            create_graph=create_graph,
            retain_graph=retain_graph,
            is_grads_batched=is_grads_batched,
        )


def _fill_in_zeros(grads, refs, strict, create_graph, stage):
    # Used to detect None in the grads and depending on the flags, either replace them
    # with Tensors full of 0s of the appropriate size based on the refs or raise an error.
    # strict and create graph allow us to detect when it is appropriate to raise an error
    # stage gives us information of which backward call we consider to give good error message
    if stage not in ["back", "back_trick", "double_back", "double_back_trick"]:
        raise RuntimeError(f"Invalid stage argument '{stage}' to _fill_in_zeros")

    res: Tuple[torch.Tensor, ...] = ()
    for i, grads_i in enumerate(grads):
        if grads_i is None:
            if strict:
                if stage == "back":
                    raise RuntimeError(
                        "The output of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode."
                    )
                elif stage == "back_trick":
                    raise RuntimeError(
                        f"The gradient with respect to the input is independent of entry {i}"
                        " in the grad_outputs when using the double backward trick to compute"
                        " forward mode gradients. This is not allowed in strict mode."
                    )
                elif stage == "double_back":
                    raise RuntimeError(
                        "The jacobian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode."
                    )
                else:
                    raise RuntimeError(
                        "The hessian of the user-provided function is independent of "
                        f"entry {i} in the grad_jacobian. This is not allowed in strict "
                        "mode as it prevents from using the double backward trick to "
                        "replace forward mode AD."
                    )

            grads_i = torch.zeros_like(refs[i])
        else:
            if strict and create_graph and not grads_i.requires_grad:
                if "double" not in stage:
                    raise RuntimeError(
                        "The jacobian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode when create_graph=True."
                    )
                else:
                    raise RuntimeError(
                        "The hessian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode when create_graph=True."
                    )

        res += (grads_i,)

    return res


# Public API


def vjp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between a vector ``v`` and the Jacobian of the given function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector
            Jacobian product is computed.  Must be the same size as the output
            of ``func``. This argument is optional when the output of ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vjp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            vjp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
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
        ...     return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = torch.ones(2)
        >>> vjp(adder, inputs, v)
        (tensor([2.4225, 2.3340]),
         (tensor([2., 2.]), tensor([3., 3.])))
    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vjp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "vjp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if v is not None:
            _, v = _as_tuple(v, "v", "vjp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, outputs, is_outputs_tuple)
        else:
            if len(outputs) != 1 or outputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the "
                    "user-provided function returns "
                    "a single Tensor with a single element."
                )

    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_graph)
        vjp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "back")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    vjp = _grad_postprocess(vjp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        vjp, is_inputs_tuple
    )


def jvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between the Jacobian of the given function at the point given by the inputs and a vector ``v``.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Jacobian
            vector product is computed. Must be the same size as the input of
            ``func``. This argument is optional when the input to ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            jvp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the output.

    Note:
        ``autograd.functional.jvp`` computes the jvp by using the backward of
        the backward (sometimes called the double backwards trick). This is not
        the most performant way of computing the jvp. Please consider using
        :func:`torch.func.jvp` or the
        :ref:`low-level forward-mode AD API <forward-mode-ad>` instead.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4, 4)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> jvp(exp_reducer, inputs, v)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106]),
         tensor([6.3090, 4.6742, 7.9114, 8.2106]))

        >>> jvp(exp_reducer, inputs, v, create_graph=True)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SumBackward1>),
         tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SqueezeBackward1>))

        >>> def adder(x, y):
        ...     return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.ones(2), torch.ones(2))
        >>> jvp(adder, inputs, v)
        (tensor([2.2399, 2.5005]),
         tensor([5., 5.]))

    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jvp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if v is not None:
            _, v = _as_tuple(v, "v", "jvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the input to "
                    "the user-provided function is a single Tensor "
                    "with a single element."
                )

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "jvp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)
        # The backward is linear so the value of grad_outputs is not important as
        # it won't appear in the double backward graph. We only need to ensure that
        # it does not contain inf or nan.
        grad_outputs = tuple(
            torch.zeros_like(out, requires_grad=True) for out in outputs
        )

        grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
        _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

    if create_graph:
        with torch.enable_grad():
            grad_res = _autograd_grad(
                grad_inputs, grad_outputs, v, create_graph=create_graph
            )
            jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")
    else:
        grad_res = _autograd_grad(
            grad_inputs, grad_outputs, v, create_graph=create_graph
        )
        jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    jvp = _grad_postprocess(jvp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        jvp, is_outputs_tuple
    )


def _construct_standard_basis_for(
    tensors: Tuple[torch.Tensor, ...], tensor_numels: Tuple[int, ...]
) -> Tuple[torch.Tensor, ...]:
    # This function:
    # - constructs a N=sum(tensor_numels) standard basis. i.e. an NxN identity matrix.
    # - Splits the identity matrix into chunks with each chunk size determined by `tensor_numels`.
    # - Each chunk corresponds to one tensor. The chunk has the same dtype and
    #   device as the tensor
    #
    # For example, with tensor_numels = [1, 2, 1], this function returns:
    # ( tensor([[1],     tensor([[0, 0],      tensor([[0],
    #           [0],             [1, 0],              [0],
    #           [0],             [0, 1],              [0],
    #           [0]])  ,         [0, 0]])  ,          [1]])  )
    #
    # Precondition: tensor_numels == tuple(tensor.numel() for tensor in tensors)
    # Precondition: tensors always has at least one element.
    #
    # See NOTE: [Computing jacobian with vmap and grad for multiple tensors]
    # for context behind this function. All the pre-conditions are guarded for
    # in torch.autograd.functional.jacobian.
    assert len(tensors) == len(tensor_numels)
    assert len(tensors) > 0
    total_numel = sum(tensor_numels)
    chunks = tuple(
        tensor.new_zeros(total_numel, tensor_numel)
        for tensor, tensor_numel in zip(tensors, tensor_numels)
    )
    diag_start_idx = 0
    for chunk, numel in zip(chunks, tensor_numels):
        chunk.diagonal(diag_start_idx).fill_(1)
        diag_start_idx -= numel
    return chunks


def _jacfwd(func, inputs, strict=False, vectorize=False):
    if strict:
        raise RuntimeError(
            "torch.autograd.functional.jacobian: `strict=True` "
            'and `strategy="forward-mode"` are not supported together (yet). '
            "Please either set `strict=False` or "
            '`strategy="reverse-mode"`.'
        )
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
    output_info = []

    if vectorize:
        # See NOTE: [Computing jacobian with vmap and grad for multiple outputs]
        input_numels = tuple(input.numel() for input in inputs)

        # Step 1: Prepare tangents
        tangents = _construct_standard_basis_for(inputs, input_numels)

        # Step 2: Compute vmap over computation with dual tensors
        def jvp(tangents):
            with fwAD.dual_level():
                dual_inputs = tuple(
                    fwAD.make_dual(input, tangent.view_as(input))
                    for input, tangent in zip(inputs, tangents)
                )
                _is_outputs_tuple, dual_outputs = _as_tuple(
                    func(*dual_inputs), "outputs"
                )
                output_info.append(_is_outputs_tuple)
                jv = []
                primal_outs = []
                for dual_out in dual_outputs:
                    primal, tangent = fwAD.unpack_dual(dual_out)
                    primal_outs.append(primal)
                    if tangent is not None:
                        jv.append(tangent)
                    else:
                        jv.append(torch.zeros_like(primal))
                output_info.append(primal_outs)
                return tuple(jv)

        outputs_before_split = _vmap(jvp)(tangents)
        is_outputs_tuple, outputs = output_info
        # Step 3: for each of the output tangents, split along dim 0
        jacobian_input_output = []
        for jac_output_i, output_i in zip(outputs_before_split, outputs):
            jacobian_output_i_output = []
            for jac, input_j in zip(jac_output_i.split(input_numels, dim=0), inputs):
                # We need to transpose the Jacobian because in forward AD, the
                # batch dimension represents that of the inputs
                jacobian_input_i_output_j = jac.permute(*range(1, jac.ndim), 0).reshape(
                    (*output_i.shape, *input_j.shape)
                )  # noqa: C409

                jacobian_output_i_output.append(jacobian_input_i_output_j)
            jacobian_input_output.append(jacobian_output_i_output)

        # Omit [Step 4] because everything is already transposed w/ forward AD
        return _tuple_postprocess(
            jacobian_input_output, (is_outputs_tuple, is_inputs_tuple)
        )
    else:
        raise NotImplementedError(
            "Computing Jacobian using forward-AD or forward-over-reverse Hessian is"
            "only implemented for `vectorize=True`."
        )


def jacobian(
    func,
    inputs,
    create_graph=False,
    strict=False,
    vectorize=False,
    strategy="reverse-mode",
):
    r"""Compute the Jacobian of a given function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Jacobian will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
        vectorize (bool, optional): This feature is experimental.
            Please consider using :func:`torch.func.jacrev` or
            :func:`torch.func.jacfwd` instead if you are looking for something
            less experimental and more performant.
            When computing the jacobian, usually we invoke
            ``autograd.grad`` once per row of the jacobian. If this flag is
            ``True``, we perform only a single ``autograd.grad`` call with
            ``batched_grad=True`` which uses the vmap prototype feature.
            Though this should lead to performance improvements in many cases,
            because this feature is still experimental, there may be performance
            cliffs. See :func:`torch.autograd.grad`'s ``batched_grad`` parameter for
            more information.
        strategy (str, optional): Set to ``"forward-mode"`` or ``"reverse-mode"`` to
            determine whether the Jacobian will be computed with forward or reverse
            mode AD. Currently, ``"forward-mode"`` requires ``vectorized=True``.
            Defaults to ``"reverse-mode"``. If ``func`` has more outputs than
            inputs, ``"forward-mode"`` tends to be more performant. Otherwise,
            prefer to use ``"reverse-mode"``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if there is a single
        input and output, this will be a single Tensor containing the
        Jacobian for the linearized inputs and output. If one of the two is
        a tuple, then the Jacobian will be a tuple of Tensors. If both of
        them are tuples, then the Jacobian will be a tuple of tuple of
        Tensors where ``Jacobian[i][j]`` will contain the Jacobian of the
        ``i``\th output and ``j``\th input and will have as size the
        concatenation of the sizes of the corresponding output and the
        corresponding input and will have same dtype and device as the
        corresponding input. If strategy is ``forward-mode``, the dtype will be
        that of the output; otherwise, the input.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = torch.rand(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
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
        ...     return 2 * x.exp() + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> jacobian(exp_adder, inputs)
        (tensor([[2.8052, 0.0000],
                [0.0000, 3.3963]]),
         tensor([[3., 0.],
                 [0., 3.]]))
    """
    assert strategy in ("forward-mode", "reverse-mode"), (
        'Expected strategy to be either "forward-mode" or "reverse-mode". Hint: If your '
        'function has more outputs than inputs, "forward-mode" tends to be more performant. '
        'Otherwise, prefer to use "reverse-mode".'
    )
    if strategy == "forward-mode":
        if create_graph:
            raise NotImplementedError(
                "torch.autograd.functional.jacobian: `create_graph=True` "
                'and `strategy="forward-mode"` are not supported together (yet). '
                "Please either set `create_graph=False` or "
                '`strategy="reverse-mode"`.'
            )
        return _jacfwd(func, inputs, strict, vectorize)

    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "jacobian"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if vectorize:
            if strict:
                raise RuntimeError(
                    "torch.autograd.functional.jacobian: `strict=True` "
                    "and `vectorized=True` are not supported together. "
                    "Please either set `strict=False` or "
                    "`vectorize=False`."
                )
            # NOTE: [Computing jacobian with vmap and grad for multiple outputs]
            #
            # Let's consider f(x) = (x**2, x.sum()) and let x = torch.randn(3).
            # It turns out we can compute the jacobian of this function with a single
            # call to autograd.grad by using vmap over the correct grad_outputs.
            #
            # Firstly, one way to compute the jacobian is to stack x**2 and x.sum()
            # into a 4D vector. E.g., use g(x) = torch.stack([x**2, x.sum()])
            #
            # To get the first row of the jacobian, we call
            # >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([1, 0, 0, 0]))
            # To get the 2nd row of the jacobian, we call
            # >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([0, 1, 0, 0]))
            # and so on.
            #
            # Using vmap, we can vectorize all 4 of these computations into one by
            # passing the standard basis for R^4 as the grad_output.
            # vmap(partial(autograd.grad, g(x), x))(torch.eye(4)).
            #
            # Now, how do we compute the jacobian *without stacking the output*?
            # We can just split the standard basis across the outputs. So to
            # compute the jacobian of f(x), we'd use
            # >>> autograd.grad(f(x), x, grad_outputs=_construct_standard_basis_for(...))
            # The grad_outputs looks like the following:
            # ( torch.tensor([[1, 0, 0],
            #                 [0, 1, 0],
            #                 [0, 0, 1],
            #                 [0, 0, 0]]),
            #   torch.tensor([[0],
            #                 [0],
            #                 [0],
            #                 [1]]) )
            #
            # But we're not done yet!
            # >>> vmap(partial(autograd.grad(f(x), x, grad_outputs=...)))
            # returns a Tensor of shape [4, 3]. We have to remember to split the
            # jacobian of shape [4, 3] into two:
            # - one of shape [3, 3] for the first output
            # - one of shape [   3] for the second output

            # Step 1: Construct grad_outputs by splitting the standard basis
            output_numels = tuple(output.numel() for output in outputs)
            grad_outputs = _construct_standard_basis_for(outputs, output_numels)
            flat_outputs = tuple(output.reshape(-1) for output in outputs)

            # Step 2: Call vmap + autograd.grad
            def vjp(grad_output):
                vj = list(
                    _autograd_grad(
                        flat_outputs,
                        inputs,
                        grad_output,
                        create_graph=create_graph,
                        is_grads_batched=True,
                    )
                )
                for el_idx, vj_el in enumerate(vj):
                    if vj_el is not None:
                        continue
                    vj[el_idx] = torch.zeros_like(inputs[el_idx]).expand(
                        (sum(output_numels),) + inputs[el_idx].shape
                    )
                return tuple(vj)

            jacobians_of_flat_output = vjp(grad_outputs)

            # Step 3: The returned jacobian is one big tensor per input. In this step,
            # we split each Tensor by output.
            jacobian_input_output = []
            for jac_input_i, input_i in zip(jacobians_of_flat_output, inputs):
                jacobian_input_i_output = []
                for jac, output_j in zip(
                    jac_input_i.split(output_numels, dim=0), outputs
                ):
                    jacobian_input_i_output_j = jac.view(output_j.shape + input_i.shape)
                    jacobian_input_i_output.append(jacobian_input_i_output_j)
                jacobian_input_output.append(jacobian_input_i_output)

            # Step 4: Right now, `jacobian` is a List[List[Tensor]].
            # The outer List corresponds to the number of inputs,
            # the inner List corresponds to the number of outputs.
            # We need to exchange the order of these and convert to tuples
            # before returning.
            jacobian_output_input = tuple(zip(*jacobian_input_output))

            jacobian_output_input = _grad_postprocess(
                jacobian_output_input, create_graph
            )
            return _tuple_postprocess(
                jacobian_output_input, (is_outputs_tuple, is_inputs_tuple)
            )

        jacobian: Tuple[torch.Tensor, ...] = ()

        for i, out in enumerate(outputs):
            # mypy complains that expression and variable have different types due to the empty list
            jac_i: Tuple[List[torch.Tensor]] = tuple([] for _ in range(len(inputs)))  # type: ignore[assignment]
            for j in range(out.nelement()):
                vj = _autograd_grad(
                    (out.reshape(-1)[j],),
                    inputs,
                    retain_graph=True,
                    create_graph=create_graph,
                )

                for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(
                    zip(jac_i, vj, inputs)
                ):
                    if vj_el is not None:
                        if strict and create_graph and not vj_el.requires_grad:
                            msg = (
                                "The jacobian of the user-provided function is "
                                f"independent of input {i}. This is not allowed in "
                                "strict mode when create_graph=True."
                            )
                            raise RuntimeError(msg)
                        jac_i_el.append(vj_el)
                    else:
                        if strict:
                            msg = (
                                f"Output {i} of the user-provided function is "
                                f"independent of input {el_idx}. This is not allowed in "
                                "strict mode."
                            )
                            raise RuntimeError(msg)
                        jac_i_el.append(torch.zeros_like(inp_el))

            jacobian += (
                tuple(
                    torch.stack(jac_i_el, dim=0).view(
                        out.size() + inputs[el_idx].size()  # type: ignore[operator]
                    )
                    for (el_idx, jac_i_el) in enumerate(jac_i)
                ),
            )

        jacobian = _grad_postprocess(jacobian, create_graph)

        return _tuple_postprocess(jacobian, (is_outputs_tuple, is_inputs_tuple))


def hessian(
    func,
    inputs,
    create_graph=False,
    strict=False,
    vectorize=False,
    outer_jacobian_strategy="reverse-mode",
):
    r"""Compute the Hessian of a given scalar function.

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
        vectorize (bool, optional): This feature is experimental.
            Please consider using :func:`torch.func.hessian`
            instead if you are looking for something less experimental and more performant.
            When computing the hessian, usually we invoke
            ``autograd.grad`` once per row of the hessian. If this flag is
            ``True``, we use the vmap prototype feature as the backend to
            vectorize calls to ``autograd.grad`` so we only invoke it once
            instead of once per row. This should lead to performance
            improvements in many use cases, however, due to this feature
            being incomplete, there may be performance cliffs. Please
            use `torch._C._debug_only_display_vmap_fallback_warnings(True)`
            to show any performance warnings and file us issues if
            warnings exist for your use case. Defaults to ``False``.
        outer_jacobian_strategy (str, optional): The Hessian is computed by
            computing the Jacobian of a Jacobian. The inner Jacobian is always
            computed in reverse-mode AD. Setting strategy to ``"forward-mode"``
            or ``"reverse-mode"`` determines whether the outer Jacobian will be
            computed with forward or reverse mode AD. Currently, computing the outer
            Jacobian in ``"forward-mode"`` requires ``vectorized=True``. Defaults
            to ``"reverse-mode"``.

    Returns:
        Hessian (Tensor or a tuple of tuple of Tensors): if there is a single input,
        this will be a single Tensor containing the Hessian for the input.
        If it is a tuple, then the Hessian will be a tuple of tuples where
        ``Hessian[i][j]`` will contain the Hessian of the ``i``\th input
        and ``j``\th input with size the sum of the size of the ``i``\th input plus
        the size of the ``j``\th input. ``Hessian[i][j]`` will have the same
        dtype and device as the corresponding ``i``\th input.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
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
        ...     return (2 * x.pow(2) + 3 * y.pow(2)).sum()
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
    assert outer_jacobian_strategy in (
        "forward-mode",
        "reverse-mode",
    ), 'Expected strategy to be either "forward-mode" or "reverse-mode".'

    def ensure_single_output_function(*inp):
        out = func(*inp)
        is_out_tuple, t_out = _as_tuple(
            out, "outputs of the user-provided function", "hessian"
        )
        _check_requires_grad(t_out, "outputs", strict=strict)

        if is_out_tuple or not isinstance(out, torch.Tensor):
            raise RuntimeError(
                "The function given to hessian should return a single Tensor"
            )

        if out.nelement() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to hessian should contain a single element"
            )

        return out.squeeze()

    def jac_func(*inp):
        if outer_jacobian_strategy == "forward-mode":
            # _grad_preprocess requires create_graph=True and input to require_grad
            # or else the input will be detached
            inp = tuple(t.requires_grad_(True) for t in inp)
        jac = jacobian(ensure_single_output_function, inp, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)
        return jac

    res = jacobian(
        jac_func,
        inputs,
        create_graph=create_graph,
        strict=strict,
        vectorize=vectorize,
        strategy=outer_jacobian_strategy,
    )
    return _tuple_postprocess(res, (is_inputs_tuple, is_inputs_tuple))


def vhp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between vector ``v`` and Hessian of a  given scalar function at a specified point.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector Hessian
            product is computed. Must be the same size as the input of
            ``func``. This argument is optional when ``func``'s input contains
            a single element and (if it is not provided) will be set as a
            Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vhp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            vhp (tuple of Tensors or Tensor): result of the dot product with the
            same shape as the inputs.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> vhp(pow_reducer, inputs, v)
        (tensor(0.5591),
         tensor([[1.0689, 1.2431],
                 [3.0989, 4.4456]]))
        >>> vhp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.5591, grad_fn=<SumBackward0>),
         tensor([[1.0689, 1.2431],
                 [3.0989, 4.4456]], grad_fn=<MulBackward0>))
        >>> def pow_adder_reducer(x, y):
        ...     return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> vhp(pow_adder_reducer, inputs, v)
        (tensor(4.8053),
         (tensor([0., 0.]),
          tensor([6., 6.])))
    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vhp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if v is not None:
            _, v = _as_tuple(v, "v", "vhp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the input to the user-provided function "
                    "is a single Tensor with a single element."
                )
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "vhp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if is_outputs_tuple or not isinstance(outputs[0], torch.Tensor):
            raise RuntimeError(
                "The function given to vhp should return a single Tensor"
            )

        if outputs[0].nelement() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to vhp should contain a single element"
            )

        jac = _autograd_grad(outputs, inputs, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)

    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(jac, inputs, v, create_graph=create_graph)
        vhp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "double_back")

    outputs = _grad_postprocess(outputs, create_graph)
    vhp = _grad_postprocess(vhp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        vhp, is_inputs_tuple
    )


def hvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between the scalar function's Hessian and a vector ``v`` at a specified point.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Hessian vector
            product is computed. Must be the same size as the input of
            ``func``. This argument is optional when ``func``'s input contains
            a single element and (if it is not provided) will be set as a
            Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            hvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            hvp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> hvp(pow_reducer, inputs, v)
        (tensor(0.1448),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]]))

        >>> hvp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.1448, grad_fn=<SumBackward0>),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]], grad_fn=<MulBackward0>))


        >>> def pow_adder_reducer(x, y):
        ...     return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> hvp(pow_adder_reducer, inputs, v)
        (tensor(2.3030),
         (tensor([0., 0.]),
          tensor([6., 6.])))

    Note:

        This function is significantly slower than `vhp` due to backward mode AD constraints.
        If your functions is twice continuously differentiable, then hvp = vhp.t(). So if you
        know that your function satisfies this condition, you should use vhp instead that is
        much faster with the current implementation.

    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hvp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if v is not None:
            _, v = _as_tuple(v, "v", "hvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the input to the user-provided function "
                    "is a single Tensor with a single element."
                )
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "hvp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if is_outputs_tuple or not isinstance(outputs[0], torch.Tensor):
            raise RuntimeError(
                "The function given to hvp should return a single Tensor"
            )

        if outputs[0].nelement() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to hvp should contain a single element"
            )

        jac = _autograd_grad(outputs, inputs, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)

        grad_jac = tuple(torch.zeros_like(inp, requires_grad=True) for inp in inputs)

        double_back = _autograd_grad(jac, inputs, grad_jac, create_graph=True)
        _check_requires_grad(jac, "hessian", strict=strict)

    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(double_back, grad_jac, v, create_graph=create_graph)
        hvp = _fill_in_zeros(
            grad_res, inputs, strict, create_graph, "double_back_trick"
        )

    outputs = _grad_postprocess(outputs, create_graph)
    hvp = _grad_postprocess(hvp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        hvp, is_inputs_tuple
    )
