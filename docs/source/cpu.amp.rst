.. role:: hidden
    :class: hidden-section

Automatic Mixed Precision package - torch.cpu.amp
==================================================

.. automodule:: torch.cpu.amp
.. currentmodule:: torch.cpu.amp

:class:`torch.cpu.amp` provides convenience methods for mixed precision,
where some operations use the ``torch.float32`` (``float``) datatype and other operations
use ``torch.bfloat16``. Some ops, like linear layers and convolutions,
are much faster in ``bfloat16``. Other ops, like reductions, often require the dynamic
range of ``float32``.  Mixed precision tries to match each op to its appropriate datatype.

.. contents:: :local:

Autocasting
^^^^^^^^^^^
.. currentmodule:: torch.cpu.amp

.. autoclass:: autocast
    :members:

Autocast Op Reference
^^^^^^^^^^^^^^^^^^^^^

Op Eligibility
--------------
Ops that run in ``float64`` or non-floating-point dtypes are not eligible, and will
run in these types whether or not autocast is enabled.

Only out-of-place ops and Tensor methods are eligible.
In-place variants and calls that explicitly supply an ``out=...`` Tensor
are allowed in autocast-enabled regions, but won't go through autocasting.
For example, in an autocast-enabled region ``a.addmm(b, c)`` can autocast,
but ``a.addmm_(b, c)`` and ``a.addmm(b, c, out=d)`` cannot.
For best performance and stability, prefer out-of-place ops in autocast-enabled
regions.

Ops called with an explicit ``dtype=...`` argument are not eligible,
and will produce output that respects the ``dtype`` argument.

Op-Specific Behavior
--------------------
The following lists describe the behavior of eligible ops in autocast-enabled regions.
These ops always go through autocasting whether they are invoked as part of a :class:`torch.nn.Module`,
as a function, or as a :class:`torch.Tensor` method. If functions are exposed in multiple namespaces,
they go through autocasting regardless of the namespace.

Ops not listed below do not go through autocasting.  They run in the type
defined by their inputs.  However, autocasting may still change the type
in which unlisted ops run if they're downstream from autocasted ops.

If an op is unlisted, we assume it's numerically stable in ``bfloat16``.
If you believe an unlisted op is numerically unstable in ``bfloat16``,
please file an issue.

Ops that can autocast to ``bfloat16``
"""""""""""""""""""""""""""""""""""""

``conv1d``,
``conv2d``,
``conv3d``,
``bmm``,
``mm``,
``baddbmm``,
``addmm``,
``addbmm``,
``linear``,
``_convolution``

Ops that can autocast to ``float32``
""""""""""""""""""""""""""""""""""""

``conv_transpose1d``,
``conv_transpose2d``,
``conv_transpose3d``,
``batch_norm``,
``dropout``,
``avg_pool1d``,
``avg_pool2d``,
``avg_pool3d``,
``gelu``,
``upsample_nearest1d``,
``_upsample_nearest_exact1d``,
``upsample_nearest2d``,
``_upsample_nearest_exact2d``,
``upsample_nearest3d``,
``_upsample_nearest_exact3d``,
``upsample_linear1d``,
``upsample_bilinear2d``,
``upsample_trilinear3d``,
``binary_cross_entropy``,
``binary_cross_entropy_with_logits``,
``instance_norm``,
``grid_sampler``,
``polar``,
``multinomial``,
``poisson``,
``fmod``,
``prod``,
``quantile``,
``nanquantile``,
``stft``,
``cdist``,
``cross``,
``cumprod``,
``cumsum``,
``diag``,
``diagflat``,
``histc``,
``logcumsumexp``,
``searchsorted``,
``trace``,
``tril``,
``triu``,
``vander``,
``view_as_complex``,
``cholesky``,
``cholesky_inverse``,
``cholesky_solve``,
``dot``,
``inverse``,
``lu_solve``,
``matrix_rank``,
``orgqr``,
``inverse``,
``ormqr``,
``pinverse``,
``vdot``,
``im2col``,
``col2im``,
``max_pool3d``,
``max_unpool2d``,
``max_unpool3d``,
``adaptive_avg_pool3d``,
``reflection_pad1d``,
``reflection_pad2d``,
``replication_pad1d``,
``replication_pad2d``,
``replication_pad3d``,
``elu``,
``hardshrink``,
``hardsigmoid``,
``hardswish``,
``log_sigmoid``,
``prelu``,
``selu``,
``celu``,
``softplus``,
``softshrink``,
``group_norm``,
``smooth_l1_loss``,
``mse_loss``,
``ctc_loss``,
``kl_div``,
``multilabel_margin_loss``,
``fft_fft``,
``fft_ifft``,
``fft_fft2``,
``fft_ifft2``,
``fft_fftn``,
``fft_ifftn``,
``fft_rfft``,
``fft_irfft``,
``fft_rfft2``,
``fft_irfft2``,
``fft_rfftn``,
``fft_irfftn``,
``fft_hfft``,
``fft_ihfft``,
``conv_tbc``,
``linalg_matrix_norm``,
``linalg_cond``,
``linalg_matrix_rank``,
``linalg_solve``,
``linalg_cholesky``,
``linalg_svdvals``,
``linalg_eigvals``,
``linalg_eigvalsh``,
``linalg_inv``,
``linalg_householder_product``,
``linalg_tensorinv``,
``linalg_tensorsolve``,
``fake_quantize_per_tensor_affine``,
``glu``,
``cummax``,
``cummin``,
``eig``,
``geqrf``,
``lstsq``,
``_lu_with_info``,
``lu_unpack``,
``qr``,
``solve``,
``svd``,
``symeig``,
``triangular_solve``,
``fractional_max_pool2d``,
``fractional_max_pool3d``,
``adaptive_max_pool1d``,
``adaptive_max_pool2d``,
``adaptive_max_pool3d``,
``multilabel_margin_loss_forward``,
``linalg_qr``,
``linalg_cholesky_ex``,
``linalg_svd``,
``linalg_eig``,
``linalg_eigh``,
``linalg_lstsq``,
``linalg_inv_ex``

Ops that promote to the widest input type
"""""""""""""""""""""""""""""""""""""""""
These ops don't require a particular dtype for stability, but take multiple inputs
and require that the inputs' dtypes match.  If all of the inputs are
``bfloat16``, the op runs in ``bfloat16``.  If any of the inputs is ``float32``,
autocast casts all inputs to ``float32`` and runs the op in ``float32``.

``cat``,
``stack``,
``index_copy``

Some ops not listed here (e.g., binary ops like ``add``) natively promote
inputs without autocasting's intervention.  If inputs are a mixture of ``bfloat16``
and ``float32``, these ops run in ``float32`` and produce ``float32`` output,
regardless of whether autocast is enabled.
