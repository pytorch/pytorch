.. meta::
   :description: A guide to torch.backends.mkldnn, a PyTorch backend to run MKLDNN operations
   :keywords: optimize PyTorch, MKLDNN

.. _mkldnn_backend:

MKLDNN backend
---------------------------------------------------

MKLDNN is an open-source cross-platform performance library of basic building blocks
for deep learning applications.

.. code:: python

  # The flag below controls whether enable MKLDNN backend in Pytorch.
  torch.backends.mkldnn.enabled = True

Users can disable MKLDNN backend by:

.. code:: python

  torch.backends.mkldnn.enabled = False

.. _bf16_on_mkldnn:

Bfloat16 (BF16) on MKLDNN backend
---------------------------------------------------

Starting in PyTorch 2.4, there is a set of APIs to control the internal computation precision
for `float32` operators.

.. code:: python

  # The flag below controls the internal computation precision for mkldnn matmul. Default is float32.
  torch.backends.mkldnn.matmul.fp32_precision = "default"

  # The flag below controls the internal computation precision for mkldnn conv. Default is float32.
  torch.backends.mkldnn.conv.fp32_precision = "default"

  # The flag below controls the internal computation precision for mkldnn rnn. Default is float32.
  torch.backends.mkldnn.rnn.fp32_precision = "default"

Note that besides matmuls and convolutions themselves, functions and nn modules that internally uses
matmuls or convolutions are also affected. These include `nn.Linear`, `nn.Conv*`, cdist, tensordot,
affine grid and grid sample, adaptive log softmax, GRU and LSTM.

To get an idea of the precision and speed, see the example code and benchmark data (on SPR) below:

.. code:: python

  torch.manual_seed(0)
  a_full = torch.randn(10240, 10240, dtype=torch.double)
  b_full = torch.randn(10240, 10240, dtype=torch.double)
  ab_full = a_full @ b_full
  mean = ab_full.abs().mean()  # 80.7451

  a = a_full.float()
  b = b_full.float()

  # Do matmul at BF16 mode.
  torch.backends.mkldnn.matmul.fp32_precision = 'bf16'
  ab_bf16 = a @ b  # expected speedup with BF16 dot-product acceleration
  error = (ab_bf16 - ab_full).abs().max()  # 1.3704
  relative_error = error / mean  # 0.0170
  print(error, relative_error)

  # Do matmul FP32 mode.
  torch.backends.mkldnn.matmul.fp32_precision = 'default'
  ab_fp32 = a @ b
  error = (ab_fp32 - ab_full).abs().max()  # 0.0003
  relative_error = error / mean  # 0.00000317
  print(error, relative_error)

From the above example, we can see that with BF16, the speed is ~7x faster on SPR, and that
relative error compared to double precision is approximately 2 orders of magnitude larger.
If full FP32 precision is needed, users can disable BF16 by:

.. code:: python

  torch.backends.mkldnn.matmul.fp32_precision = 'default'
  torch.backends.mkldnn.conv.fp32_precision = 'default'
  torch.backends.mkldnn.rnn.fp32_precision = 'default'

To toggle the TF32 flags off in C++, you can do

.. code:: C++

  at::globalContext().setFloat32Precision("default", "mkldnn", "matmul");
  at::globalContext().setFloat32Precision("default", "mkldnn", "conv");
  at::globalContext().setFloat32Precision("default", "mkldnn", "rnn");

We can override a generic setting for a specific operator or backend if the fp32_precision is set to `default`.

.. code:: python

  torch.backends.fp32_precision = "bf16"
  torch.backends.mkldnn.fp32_precision = "default"
  torch.backends.mkldnn.matmul.fp32_precision = "default"

For such case, both `torch.backends.mkldnn.fp32_precision` and `torch.backends.mkldnn.matmul.fp32_precision`
is overrided to bf16.
