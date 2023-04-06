Quantization Accuracy Debugging
-------------------------------

This document provides high level strategies for improving quantization
accuracy. If a quantized model has error compared to the original model,
we can categorize the error into:

1. **data insensitive error** - caused by intrinsic model quantization error,
   large portion of input data has large error
2. **data sensitive error** - caused by outlier input data, small
   portion of input data has large error
3. **implementation error** - quantized kernel is not matching reference implementation

Data insensitive error
~~~~~~~~~~~~~~~~~~~~~~

General tips
^^^^^^^^^^^^

1. For PTQ, ensure that the data you are calibrating with is representative
   of your dataset. For example, for a classification problem a general
   guideline is to have multiple samples in every category, and the overall
   number of samples should be at least 100. There is no penalty for
   calibrating with more data other than calibration time.
2. If your model has Conv-BN or Linear-BN patterns, consider fusing them.
   If you are using FX graph mode quantization, this is done automatically
   by the workflow. If you are using Eager mode quantization, you can do
   this manually with the ``torch.ao.quantization.fuse_modules`` API.
3. Increase the precision of dtype of the problematic ops. Usually, fp32
   will have the highest accuracy, followed by fp16, followed by dynamically
   quantized int8, followed by statically quantized int8.

   1. Note: this is trading off performance for accuracy.
   2. Note: availability of kernels per dtype per op can vary by backend.
   3. Note: dtype conversions add an additional performance cost. For example,
      ``fp32_op -> quant -> int8_op -> dequant -> fp32_op -> quant -> int8_op -> dequant``
      will have a performance penalty compared to
      ``fp32_op -> fp32_op -> quant -> int8_op -> int8_op -> dequant``
      because of a higher number of required dtype conversions.

4. If you are using PTQ, consider using QAT to recover some of the accuracy loss
   from quantization.

Int8 quantization tips
^^^^^^^^^^^^^^^^^^^^^^

1. If you are using per-tensor weight quantization, consider using per-channel
   weight quantization.
2. If you are doing inference on `fbgemm`, ensure that you set the `reduce_range`
   argument to `False` if your CPU is Cooperlake or newer, and to `True` otherwise.
3. Audit the input activation distribution variation across different samples.
   If this variation is high, the layer may be suitable for dynamic quantization
   but not static quantization.

Data sensitive error
~~~~~~~~~~~~~~~~~~~~

If you are using static quantization and a small portion of your input data is
resulting in high quantization error, you can try:

1. Adjust your calibration dataset to make it more representative of your
   inference dataset.
2. Manually inspect (using Numeric Suite) which layers have high quantization
   error. For these layers, consider leaving them in floating point or adjusting
   the observer settings to choose a better scale and zero_point.


Implementation error
~~~~~~~~~~~~~~~~~~~~

If you are using PyTorch quantization with your own backend
you may see differences between the reference implementation of an
operation (such as ``dequant -> op_fp32 -> quant``) and the quantized implementation
(such as `op_int8`) of the op on the target hardware. This could mean one of two things:

1. the differences (usually small) are expected due to specific behavior of
   the target kernel on the target hardware compared to fp32/cpu. An example of this
   is accumulating in an integer dtype. Unless the kernel guarantees bitwise
   equivalency with the reference implementation, this is expected.
2. the kernel on the target hardware has an accuracy issue. In this case, reach
   out to the kernel developer.

Numerical Debugging Tooling (prototype)
---------------------------------------

.. toctree::
    :hidden:

    torch.ao.ns._numeric_suite
    torch.ao.ns._numeric_suite_fx

.. warning ::
     Numerical debugging tooling is early prototype and subject to change.

* :ref:`torch_ao_ns_numeric_suite`
  Eager mode numeric suite
* :ref:`torch_ao_ns_numeric_suite_fx`
  FX numeric suite
