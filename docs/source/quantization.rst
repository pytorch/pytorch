.. _quantization-doc:

Quantization
============

.. automodule:: torch.ao.quantization
.. automodule:: torch.ao.quantization.fx

We are cetralizing all quantization related development to `torchao <https://github.com/pytorch/ao>`__, please checkout our new doc page: https://docs.pytorch.org/ao/stable/index.html

Plan for the existing quantization flows:
1. Eager mode quantization (torch.ao.quantization.quantize,
torch.ao.quantization.quantize_dynamic), please migrate to use torchao eager mode
`quantize_ <https://docs.pytorch.org/ao/main/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_>`__ API instead

2. FX graph mode quantization (torch.ao.quantization.quantize_fx.prepare_fx
torch.ao.quantization.quantize_fx.convert_fx, please migrate to use torchao pt2e quantization
API instead (`torchao.quantization.pt2e.quantize_pt2e.prepare_pt2e`, `torchao.quantization.pt2e.quantize_pt2e.convert_pt2e`)

3. pt2e quantization has been migrated to torchao (https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e)
see https://github.com/pytorch/ao/issues/2259 for more details

We plan to delete `torch.ao.quantization` in 2.10 if there are no blockers, or in the earliest PyTorch version until all the blockers are cleared.


Quantization API Reference (Kept since APIs are still public)
-----------------------------------------------------------------

The :doc:`Quantization API Reference <quantization-support>` contains documentation
of quantization APIs, such as quantization passes, quantized tensor operations,
and supported quantized modules and functions.

.. toctree::
    :hidden:

    quantization-support

.. torch.ao is missing documentation. Since part of it is mentioned here, adding them here for now.
.. They are here for tracking purposes until they are more permanently fixed.
.. py:module:: torch.ao
.. py:module:: torch.ao.nn
.. py:module:: torch.ao.nn.quantizable
.. py:module:: torch.ao.nn.quantizable.modules
.. py:module:: torch.ao.nn.quantized
.. py:module:: torch.ao.nn.quantized.reference
.. py:module:: torch.ao.nn.quantized.reference.modules
.. py:module:: torch.ao.nn.sparse
.. py:module:: torch.ao.nn.sparse.quantized
.. py:module:: torch.ao.nn.sparse.quantized.dynamic
.. py:module:: torch.ao.ns
.. py:module:: torch.ao.ns.fx
.. py:module:: torch.ao.quantization.backend_config
.. py:module:: torch.ao.pruning
.. py:module:: torch.ao.pruning.scheduler
.. py:module:: torch.ao.pruning.sparsifier
.. py:module:: torch.ao.nn.intrinsic.modules.fused
.. py:module:: torch.ao.nn.intrinsic.qat.modules.linear_fused
.. py:module:: torch.ao.nn.intrinsic.qat.modules.linear_relu
.. py:module:: torch.ao.nn.intrinsic.quantized.dynamic.modules.linear_relu
.. py:module:: torch.ao.nn.intrinsic.quantized.modules.bn_relu
.. py:module:: torch.ao.nn.intrinsic.quantized.modules.conv_add
.. py:module:: torch.ao.nn.intrinsic.quantized.modules.linear_relu
.. py:module:: torch.ao.nn.qat.dynamic.modules.linear
.. py:module:: torch.ao.nn.qat.modules.conv
.. py:module:: torch.ao.nn.qat.modules.embedding_ops
.. py:module:: torch.ao.nn.qat.modules.linear
.. py:module:: torch.ao.nn.quantizable.modules.activation
.. py:module:: torch.ao.nn.quantizable.modules.rnn
.. py:module:: torch.ao.nn.quantized.dynamic.modules.conv
.. py:module:: torch.ao.nn.quantized.dynamic.modules.linear
.. py:module:: torch.ao.nn.quantized.dynamic.modules.rnn
.. py:module:: torch.ao.nn.quantized.modules.activation
.. py:module:: torch.ao.nn.quantized.modules.batchnorm
.. py:module:: torch.ao.nn.quantized.modules.conv
.. py:module:: torch.ao.nn.quantized.modules.dropout
.. py:module:: torch.ao.nn.quantized.modules.embedding_ops
.. py:module:: torch.ao.nn.quantized.modules.functional_modules
.. py:module:: torch.ao.nn.quantized.modules.linear
.. py:module:: torch.ao.nn.quantized.modules.normalization
.. py:module:: torch.ao.nn.quantized.modules.rnn
.. py:module:: torch.ao.nn.quantized.modules.utils
.. py:module:: torch.ao.nn.quantized.reference.modules.conv
.. py:module:: torch.ao.nn.quantized.reference.modules.linear
.. py:module:: torch.ao.nn.quantized.reference.modules.rnn
.. py:module:: torch.ao.nn.quantized.reference.modules.sparse
.. py:module:: torch.ao.nn.quantized.reference.modules.utils
.. py:module:: torch.ao.nn.sparse.quantized.dynamic.linear
.. py:module:: torch.ao.nn.sparse.quantized.linear
.. py:module:: torch.ao.nn.sparse.quantized.utils
.. py:module:: torch.ao.ns.fx.graph_matcher
.. py:module:: torch.ao.ns.fx.graph_passes
.. py:module:: torch.ao.ns.fx.mappings
.. py:module:: torch.ao.ns.fx.n_shadows_utils
.. py:module:: torch.ao.ns.fx.ns_types
.. py:module:: torch.ao.ns.fx.pattern_utils
.. py:module:: torch.ao.ns.fx.qconfig_multi_mapping
.. py:module:: torch.ao.ns.fx.weight_utils
.. py:module:: torch.ao.ns.fx.utils
.. py:module:: torch.ao.pruning.scheduler.base_scheduler
.. py:module:: torch.ao.pruning.scheduler.cubic_scheduler
.. py:module:: torch.ao.pruning.scheduler.lambda_scheduler
.. py:module:: torch.ao.pruning.sparsifier.base_sparsifier
.. py:module:: torch.ao.pruning.sparsifier.nearly_diagonal_sparsifier
.. py:module:: torch.ao.pruning.sparsifier.utils
.. py:module:: torch.ao.pruning.sparsifier.weight_norm_sparsifier
.. py:module:: torch.ao.quantization.backend_config.backend_config
.. py:module:: torch.ao.quantization.backend_config.executorch
.. py:module:: torch.ao.quantization.backend_config.fbgemm
.. py:module:: torch.ao.quantization.backend_config.native
.. py:module:: torch.ao.quantization.backend_config.onednn
.. py:module:: torch.ao.quantization.backend_config.qnnpack
.. py:module:: torch.ao.quantization.backend_config.tensorrt
.. py:module:: torch.ao.quantization.backend_config.utils
.. py:module:: torch.ao.quantization.backend_config.x86
.. py:module:: torch.ao.quantization.fake_quantize
.. py:module:: torch.ao.quantization.fuser_method_mappings
.. py:module:: torch.ao.quantization.fuse_modules
.. py:module:: torch.ao.quantization.fx.convert
.. py:module:: torch.ao.quantization.fx.custom_config
.. py:module:: torch.ao.quantization.fx.fuse
.. py:module:: torch.ao.quantization.fx.fuse_handler
.. py:module:: torch.ao.quantization.fx.graph_module
.. py:module:: torch.ao.quantization.fx.lower_to_fbgemm
.. py:module:: torch.ao.quantization.fx.lower_to_qnnpack
.. py:module:: torch.ao.quantization.fx.lstm_utils
.. py:module:: torch.ao.quantization.fx.match_utils
.. py:module:: torch.ao.quantization.fx.pattern_utils
.. py:module:: torch.ao.quantization.fx.prepare
.. py:module:: torch.ao.quantization.fx.qconfig_mapping_utils
.. py:module:: torch.ao.quantization.fx.quantize_handler
.. py:module:: torch.ao.quantization.fx.tracer
.. py:module:: torch.ao.quantization.fx.utils
.. py:module:: torch.ao.quantization.observer
.. py:module:: torch.ao.quantization.qconfig
.. py:module:: torch.ao.quantization.qconfig_mapping
.. py:module:: torch.ao.quantization.quant_type
.. py:module:: torch.ao.quantization.quantization_mappings
.. py:module:: torch.ao.quantization.quantize_fx
.. py:module:: torch.ao.quantization.quantize_jit
.. py:module:: torch.ao.quantization.stubs
.. py:module:: torch.nn.intrinsic.modules.fused
.. py:module:: torch.nn.intrinsic.qat.modules.conv_fused
.. py:module:: torch.nn.intrinsic.qat.modules.linear_fused
.. py:module:: torch.nn.intrinsic.qat.modules.linear_relu
.. py:module:: torch.nn.intrinsic.quantized.dynamic.modules.linear_relu
.. py:module:: torch.nn.intrinsic.quantized.modules.bn_relu
.. py:module:: torch.nn.intrinsic.quantized.modules.conv_relu
.. py:module:: torch.nn.intrinsic.quantized.modules.linear_relu
.. py:module:: torch.nn.qat.dynamic.modules.linear
.. py:module:: torch.nn.qat.modules.conv
.. py:module:: torch.nn.qat.modules.embedding_ops
.. py:module:: torch.nn.qat.modules.linear
.. py:module:: torch.nn.quantizable.modules.activation
.. py:module:: torch.nn.quantizable.modules.rnn
.. py:module:: torch.nn.quantized.dynamic.modules.conv
.. py:module:: torch.nn.quantized.dynamic.modules.linear
.. py:module:: torch.nn.quantized.dynamic.modules.rnn
.. py:module:: torch.nn.quantized.functional
.. py:module:: torch.nn.quantized.modules.activation
.. py:module:: torch.nn.quantized.modules.batchnorm
.. py:module:: torch.nn.quantized.modules.conv
.. py:module:: torch.nn.quantized.modules.dropout
.. py:module:: torch.nn.quantized.modules.embedding_ops
.. py:module:: torch.nn.quantized.modules.functional_modules
.. py:module:: torch.nn.quantized.modules.linear
.. py:module:: torch.nn.quantized.modules.normalization
.. py:module:: torch.nn.quantized.modules.rnn
.. py:module:: torch.nn.quantized.modules.utils
.. py:module:: torch.quantization.fake_quantize
.. py:module:: torch.quantization.fuse_modules
.. py:module:: torch.quantization.fuser_method_mappings
.. py:module:: torch.quantization.fx.convert
.. py:module:: torch.quantization.fx.fuse
.. py:module:: torch.quantization.fx.fusion_patterns
.. py:module:: torch.quantization.fx.graph_module
.. py:module:: torch.quantization.fx.match_utils
.. py:module:: torch.quantization.fx.pattern_utils
.. py:module:: torch.quantization.fx.prepare
.. py:module:: torch.quantization.fx.quantization_patterns
.. py:module:: torch.quantization.fx.quantization_types
.. py:module:: torch.quantization.fx.utils
.. py:module:: torch.quantization.observer
.. py:module:: torch.quantization.qconfig
.. py:module:: torch.quantization.quant_type
.. py:module:: torch.quantization.quantization_mappings
.. py:module:: torch.quantization.quantize
.. py:module:: torch.quantization.quantize_fx
.. py:module:: torch.quantization.quantize_jit
.. py:module:: torch.quantization.stubs
.. py:module:: torch.quantization.utils


.. currentmodule:: torch.ao.ns.fx.utils
.. autofunction:: torch.ao.ns.fx.utils.compute_sqnr(x, y)
.. autofunction:: torch.ao.ns.fx.utils.compute_normalized_l2_error(x, y)
.. autofunction:: torch.ao.ns.fx.utils.compute_cosine_similarity(x, y)
