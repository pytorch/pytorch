.. warning::
    There are known non-determinism issues for RNN functions on some versions of cuDNN and CUDA.
    You can enforce deterministic behavior by setting the following environment variables:

    On CUDA 10.1, set environment variable ``CUDA_LAUNCH_BLOCKING=1``.
    This may affect performance.

    On CUDA 10.2 or later, set environment variable
    (note the leading colon symbol)
    ``CUBLAS_WORKSPACE_CONFIG=:16:8``
    or
    ``CUBLAS_WORKSPACE_CONFIG=:4096:2``.

    See the `cuDNN 8 Release Notes`_ for more information.

.. _cuDNN 8 Release Notes: https://docs.nvidia.com/deeplearning/sdk/cudnn-release-notes/rel_8.html
