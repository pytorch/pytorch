.. _mps_environment_variables:

MPS Environment Variables
==========================

**PyTorch Environment Variables**

.. list-table::
  :header-rows: 1

  * - Variable
    - Description
  * - ``PYTORCH_DEBUG_MPS_ALLOCATOR``
    - If set to ``1``, set allocator logging level to verbose.
  * - ``PYTORCH_MPS_HIGH_WATERMARK_RATIO``
    - High watermark ratio for MPS allocator. By default, it is set to 1.7.
  * - ``PYTORCH_MPS_LOW_WATERMARK_RATIO``
    - Low watermark ratio for MPS allocator. By default, it is set to 1.4 if the memory is unified and set to 1.0 if the memory is discrete.
  * - ``PYTORCH_MPS_FAST_MATH``
    - If set to ``1``, enable fast math for MPS metal kernels. See section 1.6.3 in https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf for precision implications.
  * - ``PYTORCH_MPS_PREFER_METAL``
    - If set to ``1``, force using metal kernels instead of using MPS Graph APIs. For now this is only used for matmul op.
  * - ``PYTORCH_ENABLE_MPS_FALLBACK``
    - If set to ``1``, full back operations to CPU when MPS does not support them.

.. note::

    **high watermark ratio** is a hard limit for the total allowed allocations

    - `0.0` : disables high watermark limit (may cause system failure if system-wide OOM occurs)
    - `1.0` : recommended maximum allocation size (i.e., device.recommendedMaxWorkingSetSize)
    - `>1.0`: allows limits beyond the device.recommendedMaxWorkingSetSize

    e.g., value 0.95 means we allocate up to 95% of recommended maximum
    allocation size; beyond that, the allocations would fail with OOM error.

    **low watermark ratio** is a soft limit to attempt limiting memory allocations up to the lower watermark
    level by garbage collection or committing command buffers more frequently (a.k.a, adaptive commit).
    Value between 0 to m_high_watermark_ratio (setting 0.0 disables adaptive commit and garbage collection)
    e.g., value 0.9 means we 'attempt' to limit allocations up to 90% of recommended maximum
    allocation size.