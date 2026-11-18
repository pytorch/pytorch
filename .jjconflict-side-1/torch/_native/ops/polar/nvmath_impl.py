"""cuSOLVER Xpolar (QDWH) polar decomposition via nvmath bindings."""

import ctypes

import torch

from ... import nvmath_utils


# cuBLAS fill mode passed to Xpolar. The general polar decomposition reads the
# entire input A (it is not assumed symmetric), so FULL is required -- LOWER is
# rejected and UPPER silently reads only a triangle, yielding wrong factors.
_CUBLAS_FILL_MODE_FULL = 2


def _cuda_dtype(dtype: torch.dtype):
    # cuSOLVER Xpolar supports only real single/double precision. Complex inputs
    # are routed to the SVD path by the caller and never reach here.
    from cuda.bindings.runtime import cudaDataType  # pyrefly: ignore[missing-import]

    return {
        torch.float32: cudaDataType.CUDA_R_32F,
        torch.float64: cudaDataType.CUDA_R_64F,
    }[dtype]


def _empty_colmajor(m: int, n: int, A: torch.Tensor) -> torch.Tensor:
    # Column-major (m, n) buffer for cuSOLVER (leading dimension m). The view
    # owns its storage and presents logical shape (m, n) so values read back
    # directly without a transpose.
    return A.new_empty((n, m)).mT


def _polar_2d(
    A: torch.Tensor, *, return_residual: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    from nvmath.bindings import cusolverDn as cs  # pyrefly: ignore[missing-import]

    m, n = A.shape
    dtype = A.dtype
    device = A.device

    # cuSOLVER handles are device-bound; run everything (handle lookup, workspace
    # allocation, and the xpolar call) with A's device current so a non-default
    # device (e.g. cuda:1) doesn't trigger an illegal memory access.
    with torch.cuda.device(device):
        handle = nvmath_utils.get_cusolver_handle(device)
        params = cs.create_params()
        return _polar_2d_impl(A, m, n, dtype, device, handle, params, return_residual)


def _polar_2d_impl(A, m, n, dtype, device, handle, params, return_residual):
    from nvmath.bindings import cusolverDn as cs  # pyrefly: ignore[missing-import]

    try:
        cs.set_stream(handle, torch.cuda.current_stream(device).cuda_stream)

        # cuSOLVER is column-major and overwrites the input (m x n, ld=m) with U.
        # Use column-major buffers so the input is untouched and both factors
        # read back with the correct (non-transposed/non-conjugated) values.
        U_col = _empty_colmajor(m, n, A)
        U_col.copy_(A)
        H_col = _empty_colmajor(n, n, A)

        # Real single/double: input, output, and compute share one data type.
        cdt = _cuda_dtype(dtype)
        dtype_a = dtype_h = compute_type = cdt

        ws_dev, ws_host = cs.xpolar_buffer_size(
            handle,
            params,
            _CUBLAS_FILL_MODE_FULL,
            m,
            n,
            dtype_a,
            U_col.data_ptr(),
            m,
            dtype_h,
            H_col.data_ptr(),
            n,
            compute_type,
        )

        # Allocate exactly the queried sizes. When a size is 0, cuSOLVER does
        # not dereference the corresponding pointer (an empty CUDA tensor's
        # data_ptr is null, which is the expected "no workspace" convention).
        buf_dev = torch.empty(ws_dev, dtype=torch.uint8, device=device)
        buf_host = bytearray(ws_host)
        host_ptr = (ctypes.c_char * ws_host).from_buffer(buf_host)

        # Xpolar reports convergence diagnostics into device scalars: three
        # double* (res_nrm, A_nrmF, rcond) and one int* (info). Pack all four
        # into a single byte buffer (8-byte aligned slots) so a single kernel
        # zero-initializes them.
        #
        # IMPORTANT: these MUST be zero-initialized. cuSOLVER 12.2.0.1's Xpolar
        # reads res_nrm/A_nrmF/rcond as accumulators during the QDWH iteration
        # (not write-only outputs); passing uninitialized memory (e.g. a block
        # the caching allocator just recycled from prior GPU work) makes the
        # iteration diverge and silently return a wrong factorization with
        # info=0. Zeroing them makes the result deterministic and correct.
        diag = torch.zeros(28, dtype=torch.uint8, device=device)
        diag_ptr = diag.data_ptr()
        d_res_nrm = diag_ptr  # double
        d_a_nrm_f = diag_ptr + 8  # double, == ||A||_F
        d_rcond = diag_ptr + 16  # double
        d_info = diag_ptr + 24  # int32

        cs.xpolar(
            handle,
            params,
            _CUBLAS_FILL_MODE_FULL,
            m,
            n,
            dtype_a,
            U_col.data_ptr(),
            m,
            dtype_h,
            H_col.data_ptr(),
            n,
            compute_type,
            buf_dev.data_ptr(),
            ws_dev,
            ctypes.addressof(host_ptr),
            ws_host,
            d_res_nrm,
            d_a_nrm_f,
            d_rcond,
            d_info,
        )

        # H is symmetric/Hermitian up to round-off; symmetrize to make it exact.
        H = 0.5 * (H_col + H_col.mH)

        # Optionally expose the relative residual ||A - U*H||_F / ||A||_F that
        # Xpolar reports, as a device scalar. We never sync on it ourselves;
        # callers who want it can copy it to host (taking the sync hit) for
        # debugging or to validate convergence.
        residual = None
        if return_residual:
            diag_f64 = diag[:24].view(torch.float64)
            a_nrm = diag_f64[1]
            residual = diag_f64[0] / a_nrm.clamp_min(torch.finfo(torch.float64).tiny)
        return U_col, H, residual
    finally:
        cs.destroy_params(params)


def polar_xpolar(
    A: torch.Tensor, *, return_residual: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Polar decomposition A = U @ H of a single 2-D matrix via cuSOLVER Xpolar.

    Fully asynchronous: issues work on the current CUDA stream and never
    synchronizes. When ``return_residual`` is True, also returns the relative
    residual ``||A - U*H||_F / ||A||_F`` as a device scalar for callers that
    want to validate convergence.

    cuSOLVER Xpolar has no batched API, so only 2-D inputs are supported; the
    dispatch override declines batched inputs (they take the batched SVD
    kernel instead of a slower per-matrix loop).
    """
    torch._check(A.dim() == 2, lambda: "polar_xpolar expects a 2-D matrix")
    return _polar_2d(A, return_residual=return_residual)
