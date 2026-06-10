# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Inductor lowerings for torchcomms collective operations."""

import logging

import torch


__all__ = ["register_torchcomms_lowerings"]

logger = logging.getLogger(__name__)

try:
    from torch._inductor import ir
    from torch._inductor.lowering import register_lowering

    def register_torchcomms_lowerings():
        from torch.comms.functional import collectives

        """Register all torchcomms collective lowerings with inductor."""
        if collectives is None:
            logger.warning("torch.comms.functional.collectives not available")
            return

        try:
            from torch.comms.functional.registry import _REGISTERED_COLLECTIVES

            # Register ops with the reinplace pass and create lowerings
            for base_op_name, info in _REGISTERED_COLLECTIVES.items():
                schema = info["param_schema"]
                if len(schema.mutable_params) > 0:
                    _register_with_reinplace_pass(base_op_name, schema)
                    _register_inplace_lowering(base_op_name, schema)
                    _register_functional_lowering(base_op_name, schema)

            # Register lowering for functional wait_tensors
            _register_wait_tensors_lowering()

            logger.info("Registered torchcomms lowerings for torch.compile")
        except AttributeError as e:
            logger.warning("Failed to register torchcomms lowerings: %s", e)

    def _register_with_reinplace_pass(base_op_name: str, schema) -> None:
        """Register functional/inplace op pair with the reinplace pass.

        This allows the reinplace pass to convert functional ops to inplace
        when ALL input tensors have no other uses.
        """
        try:
            from torch._inductor.fx_passes.reinplace import (
                inplaceable_ops,
                InplaceableOp,
            )

            functional_op = getattr(torch.ops.torchcomms, base_op_name, None)
            inplace_op = getattr(torch.ops.torchcomms, f"{base_op_name}_", None)

            if functional_op is None or inplace_op is None:
                return

            # Find all mutable tensor arg indices
            mutable_tensor_indices = []
            for i, p in enumerate(schema.input_params):
                if p.mutable and (
                    p.torch_type == "Tensor" or p.torch_type == "Tensor[]"
                ):
                    mutable_tensor_indices.append(
                        i + 1
                    )  # +1 for comm object at index 0

            if not mutable_tensor_indices:
                return

            inplaceable_ops[functional_op.default] = InplaceableOp(
                inplace_op.default,
                (
                    tuple(mutable_tensor_indices)
                    if len(mutable_tensor_indices) > 1
                    else mutable_tensor_indices[0]
                ),
            )
            logger.info(
                "Registered reinplace: %s -> %s_ (mutated_args=%s)",
                base_op_name,
                base_op_name,
                mutable_tensor_indices[0],
            )
        except ImportError:
            logger.debug("reinplace pass not available")

    def _register_inplace_lowering(base_op_name: str, schema) -> None:
        """Register lowering for the inplace op.

        The inplace op mutates tensors directly - no cloning needed.
        (Reinplace only converts to inplace when all tensors can be inplaced.)
        """
        from torch._inductor.virtualized import V

        inplace_op = getattr(torch.ops.torchcomms, f"{base_op_name}_", None)
        if inplace_op is None:
            return

        # Get indices of mutable tensor args (offset by 1 for the comm object)
        mutable_indices = []
        for i, p in enumerate(schema.input_params):
            if p.mutable and (p.torch_type == "Tensor" or p.torch_type == "Tensor[]"):
                mutable_indices.append(i + 1)

        def _inplace_lowering(*args):
            logger.debug("Lowering inplace %s_", base_op_name)

            # Get the mutable tensors from args
            mutable_tensors = []
            for idx in mutable_indices:
                if idx < len(args):
                    tensor_arg = args[idx]
                    if isinstance(tensor_arg, ir.TensorBox):
                        mutable_tensors.append(tensor_arg)
                    elif isinstance(tensor_arg, (list, tuple)):
                        mutable_tensors.extend(
                            t for t in tensor_arg if isinstance(t, ir.TensorBox)
                        )

            if not mutable_tensors:
                logger.warning("No mutable tensors for %s_", base_op_name)
                return None

            # Realize and mark tensors as mutated
            device = None
            for tensor in mutable_tensors:
                tensor.realize()
                V.graph.mark_buffer_mutated(tensor.get_name())
                if device is None:
                    device = tensor.get_device()

            # Process kernel args
            with V.graph.fake_mode:
                (
                    _example_output,
                    tensor_args,
                    non_tensor_args,
                    unflatten_args,
                    unbacked_bindings,
                ) = ir._CollectiveKernel.process_kernel(inplace_op.default, *args)
            assert not unbacked_bindings, f"{inplace_op} {unbacked_bindings}"

            # Create the collective kernel
            packed = ir._CollectiveKernel(
                ir.NoneLayout(device=device),
                inplace_op.default,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )

            # Set up mutation_outputs
            packed.mutation_outputs.extend(
                [
                    ir.MutationOutput(ir.NoneLayout(device=device), buf, packed)
                    for buf in mutable_tensors
                ]
            )
            packed.alias_names.extend([t.get_name() for t in mutable_tensors])

            # Return the mutable tensors
            if len(mutable_tensors) == 1:
                return mutable_tensors[0]
            return mutable_tensors

        register_lowering(inplace_op.default)(_inplace_lowering)
        logger.info("Registered inplace lowering: %s_", base_op_name)

    def _register_functional_lowering(base_op_name: str, schema) -> None:
        """Register lowering for the functional op.

        Uses FallbackKernel with the functional op directly.
        The functional op is non-mutable (returns new tensors), so FallbackKernel can handle it.
        """
        import torch.utils._pytree as pytree

        functional_op = getattr(torch.ops.torchcomms, base_op_name, None)

        if functional_op is None:
            return

        def _functional_lowering(*args):
            logger.debug("Lowering functional %s with %s args", base_op_name, len(args))

            # Use FallbackKernel with the functional op
            # The functional op returns new tensors, so it's not mutable
            def wrap_tensors(x):
                return ir.TensorBox.create(x) if isinstance(x, ir.IRNode) else x

            result = pytree.tree_map(
                wrap_tensors,
                ir._CollectiveKernel.create_out_of_place(functional_op.default, *args),
            )

            return result

        register_lowering(functional_op.default)(_functional_lowering)
        logger.info("Registered functional lowering: %s", base_op_name)

    def _register_wait_tensors_lowering() -> None:
        """Register lowerings for both functional and inplace wait_tensors."""
        from torch._inductor.virtualized import V

        # === REINPLACE PASS REGISTRATION ===
        # Allow reinplace to convert functional -> inplace when tensors have no other uses.
        try:
            from torch._inductor.fx_passes.reinplace import (
                inplaceable_ops,
                InplaceableOp,
            )

            functional_op = torch.ops.torchcomms.torchcomm_wait_tensors
            inplace_op = torch.ops.torchcomms.torchcomm_wait_tensors_

            # wait_tensors takes a list of tensors at index 0
            # All tensors in the list are mutated
            inplaceable_ops[functional_op.default] = InplaceableOp(
                inplace_op.default,
                0,  # arg index 0 is the tensor list
            )
            logger.info(
                "Registered reinplace: torchcomm_wait_tensors -> torchcomm_wait_tensors_"
            )
        except ImportError:
            logger.debug("reinplace pass not available for wait_tensors")

        # === FUNCTIONAL LOWERING ===
        # Use FallbackKernel to create new output tensors that depend on the wait.
        # FallbackKernel properly handles the op call and creates output buffers.
        def _wait_tensors_functional_lowering(*args):
            import torch.utils._pytree as pytree

            logger.debug(
                "Lowering functional torch.comms.torchcomm_wait_tensors with %s args",
                len(args),
            )

            # The op takes a list of tensors as the first argument
            # Inductor may pass this as a list or as individual tensors
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                inputs = list(args[0])
            else:
                inputs = list(args)

            if not inputs:
                return []

            # Flatten to get individual TensorBox objects
            flat_inputs = []
            for inp in inputs:
                if isinstance(inp, ir.TensorBox):
                    flat_inputs.append(inp)
                elif isinstance(inp, (list, tuple)):
                    flat_inputs.extend(t for t in inp if isinstance(t, ir.TensorBox))

            if not flat_inputs:
                return []

            logger.info("  - Processing %s TensorBox inputs", len(flat_inputs))

            # Use FallbackKernel to create new output tensors
            # Pass flat_inputs as a list since the op signature is Tensor[] -> Tensor[]
            def wrap_tensors(x):
                return ir.TensorBox.create(x) if isinstance(x, ir.IRNode) else x

            result = pytree.tree_map(
                wrap_tensors,
                ir.FallbackKernel.create(
                    torch.ops.torchcomms.torchcomm_wait_tensors.default,
                    flat_inputs,
                ),
            )

            logger.debug("  - Created FallbackKernel result: %s", type(result))
            return result

        register_lowering(torch.ops.torchcomms.torchcomm_wait_tensors.default)(
            _wait_tensors_functional_lowering
        )
        logger.info("Registered functional wait_tensors lowering")

        # === INPLACE LOWERING ===
        # Used when reinplace pass converts functional -> inplace.
        # Uses _WaitKernel for proper wait semantics.
        # Now returns the input tensors to match the updated op signature.
        def _wait_tensors_inplace_lowering(*args):
            # The op takes a list of tensors as the first argument
            # Inductor may pass this as a list or as individual tensors
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                inputs = list(args[0])
            else:
                inputs = list(args)

            logger.debug(
                "Lowering inplace torch.comms.torchcomm_wait_tensors_ with %s tensors",
                len(inputs),
            )

            if not inputs:
                return []

            # Flatten to get individual TensorBox objects
            flat_inputs = []
            for inp in inputs:
                if isinstance(inp, ir.TensorBox):
                    flat_inputs.append(inp)
                elif isinstance(inp, (list, tuple)):
                    flat_inputs.extend(t for t in inp if isinstance(t, ir.TensorBox))

            if not flat_inputs:
                return []

            # Realize all inputs and mark as mutated
            device = None
            for inp in flat_inputs:
                inp.realize()
                V.graph.mark_buffer_mutated(inp.get_name())
                if device is None:
                    device = inp.get_device()

            # Create wait kernel using the inplace op
            with V.graph.fake_mode:
                (
                    _example_output,
                    tensor_args,
                    non_tensor_args,
                    unflatten_args,
                    unbacked_bindings,
                ) = ir._WaitKernel.process_kernel(
                    torch.ops.torchcomms.torchcomm_wait_tensors_.default,
                    flat_inputs,
                )
            assert not unbacked_bindings

            packed = ir._WaitKernel(
                ir.NoneLayout(device=device),
                torch.ops.torchcomms.torchcomm_wait_tensors_.default,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )

            # Add MutationOutput for each input tensor to register with graph
            for inp in flat_inputs:
                packed.mutation_outputs.append(
                    ir.MutationOutput(ir.NoneLayout(device=device), inp, packed)
                )
            packed.alias_names.extend([t.get_name() for t in flat_inputs])

            # Return the input tensors (they are mutated in-place)
            return flat_inputs

        register_lowering(torch.ops.torchcomms.torchcomm_wait_tensors_.default)(
            _wait_tensors_inplace_lowering
        )
        logger.info(
            "Registered inplace wait_tensors lowering for %s",
            torch.ops.torchcomms.torchcomm_wait_tensors_.default,
        )

except ImportError:
    logger.info("torch._inductor not available, skipping torchcomms lowerings")

    def register_torchcomms_lowerings():
        pass
