torch.utils.mobile_optimizer
===================================

.. warning::
   PyTorch Mobile is no longer actively supported. Please check out
   `ExecuTorch <https://pytorch.org/executorch-overview>`__, PyTorch's
   all-new on-device inference library. You can also review
   documentation on `XNNPACK <https://pytorch.org/executorch/stable/native-delegates-executorch-xnnpack-delegate.html>`__
   and `Vulkan <https://pytorch.org/executorch/stable/native-delegates-executorch-vulkan-delegate.html>`__ delegates.

Torch mobile supports ``torch.utils.mobile_optimizer.optimize_for_mobile`` utility to run a list of optimization pass with modules in eval mode.
The method takes the following parameters: a torch.jit.ScriptModule object, a blocklisting optimization set, a preserved method list, and a backend.

For CPU Backend, by default, if optimization blocklist is None or empty, ``optimize_for_mobile`` will run the following optimizations:
    - **Conv2D + BatchNorm fusion** (blocklisting option `mobile_optimizer.MobileOptimizerType.CONV_BN_FUSION`):  This optimization pass folds ``Conv2d-BatchNorm2d`` into ``Conv2d`` in ``forward`` method of this module and all its submodules. The weight and bias of the ``Conv2d`` are correspondingly updated.
    - **Insert and Fold prepacked ops** (blocklisting option `mobile_optimizer.MobileOptimizerType.INSERT_FOLD_PREPACK_OPS`): This optimization pass rewrites the graph to replace 2D convolutions and linear ops with their prepacked counterparts. Prepacked ops are stateful ops in that, they require some state to be created, such as weight prepacking and use this state, i.e. prepacked weights, during op execution. XNNPACK is one such backend that provides prepacked ops, with kernels optimized for mobile platforms (such as ARM CPUs). Prepacking of weight enables efficient memory access and thus faster kernel execution. At the moment ``optimize_for_mobile`` pass rewrites the graph to replace ``Conv2D/Linear`` with 1) op that pre-packs weight for XNNPACK conv2d/linear ops and 2) op that takes pre-packed weight and activation as input and generates output activations. Since 1 needs to be done only once, we fold the weight pre-packing such that it is done only once at model load time. This pass of the ``optimize_for_mobile`` does 1 and 2 and then folds, i.e. removes, weight pre-packing ops.
    - **ReLU/Hardtanh fusion**: XNNPACK ops support fusion of clamping. That is clamping of output activation is done as part of the kernel, including for 2D convolution and linear op kernels. Thus clamping effectively comes for free. Thus any op that can be expressed as clamping op, such as ``ReLU`` or ``hardtanh``, can be fused with previous ``Conv2D`` or ``linear`` op in XNNPACK. This pass rewrites graph by finding ``ReLU/hardtanh`` ops that follow XNNPACK ``Conv2D/linear`` ops, written by the previous pass, and fuses them together.
    - **Dropout removal** (blocklisting option `mobile_optimizer.MobileOptimizerType.REMOVE_DROPOUT`): This optimization pass removes ``dropout`` and ``dropout_`` nodes from this module when training is false.
    - **Conv packed params hoisting** (blocklisting option `mobile_optimizer.MobileOptimizerType.HOIST_CONV_PACKED_PARAMS`): This optimization pass moves convolution packed params to the root module, so that the convolution structs can be deleted. This decreases model size without impacting numerics.
    - **Add/ReLU fusion** (blocklisting option `mobile_optimizer.MobileOptimizerType.FUSE_ADD_RELU`): This pass finds instances of ``relu`` ops that follow ``add`` ops and fuses them into a single ``add_relu``.

for Vulkan Backend, by default, if optimization blocklist is None or empty, ``optimize_for_mobile`` will run the following optimization:
    - **Automatic GPU Transfer** (blocklisting option `mobile_optimizer.MobileOptimizerType.VULKAN_AUTOMATIC_GPU_TRANSFER`): This optimization pass rewrites the graph so that moving input and output data to and from the GPU becomes part of the model.

``optimize_for_mobile`` will also invoke freeze_module pass which only preserves ``forward`` method. If you have other method to that needed to be preserved, add them into the preserved method list and pass into the method.


.. currentmodule:: torch.utils.mobile_optimizer
.. autofunction:: optimize_for_mobile
