"""Cached cublasLt grouped GEMM via nvmath bindings."""

import ctypes

import numpy as np
from cuda.bindings.runtime import cudaDataType
from nvmath.bindings import cublasLt
from nvmath.bindings.cublas import ComputeType, Operation

import torch


# cublasLt scratch space for kernel execution. Shared across all cached
# ForeachMMCublasLt instances. Larger values may enable better algorithms
# for large shapes; 32 MB matches the PyTorch C++ cublasLt path.
CUBLASLT_WORKSPACE_BYTES = 32 * 1024 * 1024

_cublaslt_handle = None
_cublaslt_workspace = None


def _get_cublaslt_handle():
    global _cublaslt_handle
    if _cublaslt_handle is None:
        _cublaslt_handle = cublasLt.create()
    return _cublaslt_handle


def _get_cublaslt_workspace(device="cuda"):
    global _cublaslt_workspace
    if _cublaslt_workspace is None or _cublaslt_workspace.device != torch.device(
        device
    ):
        _cublaslt_workspace = torch.empty(
            CUBLASLT_WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
    return _cublaslt_workspace


def _set_attr(setter, handle, attr, val, ctype):
    buf = (ctype * 1)(val)
    setter(handle, attr, ctypes.addressof(buf), ctypes.sizeof(buf))


class ForeachMMCublasLt:
    """Cached cublasLt grouped GEMM for repeated same-shape calls."""

    def __init__(
        self,
        M,
        N,
        K,
        G,
        a_row_major=True,
        b_row_major=True,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        if dtype != torch.bfloat16:
            raise ValueError(f"ForeachMMCublasLt only supports bf16, got {dtype}")

        self.G = G
        self.M, self.N = M, N
        elem_size = 2  # bf16
        alignment = 16 // elem_size
        self.N_padded = (N + alignment - 1) // alignment * alignment
        self._out_stride_bytes = M * self.N_padded * elem_size

        # cuBLAS col-major: we compute C^T = mat2^T * self^T
        # cuBLAS-A = mat2, cuBLAS-B = self
        # Row-major matrix seen as col-major without transpose -> OP_N
        # Col-major matrix seen as col-major transposed -> OP_T
        opa = Operation.N if b_row_major else Operation.T
        opb = Operation.N if a_row_major else Operation.T
        lda = N if b_row_major else K
        ldb = K if a_row_major else M

        # Device-side dim arrays (constant across calls)
        # cublas_m=N, cublas_n=M, cublas_k=K
        dims = (
            torch.tensor(
                [N, M, K, lda, ldb, self.N_padded] * G,
                dtype=torch.int32,
                device=device,
            )
            .reshape(G, 6)
            .T.contiguous()
        )
        d_m, d_n, d_k, d_lda, d_ldb, d_ldd = [dims[i] for i in range(6)]

        # Pinned host + device buffers for pointer arrays [Aptr|Bptr|Dptr]
        self._pinned = torch.empty(3 * G, dtype=torch.int64, pin_memory=True)
        self._pinned_raw = ctypes.cast(
            self._pinned.data_ptr(), ctypes.POINTER(ctypes.c_int64)
        )
        self._dev_ptrs = torch.empty(3 * G, dtype=torch.int64, device=device)

        # alpha=1.0, beta=0.0 as device float32
        scalars = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
        self._alpha_ptr = scalars.data_ptr()
        self._beta_ptr = scalars.data_ptr() + 4
        self._scalars = scalars  # prevent GC

        self._workspace = _get_cublaslt_workspace(device)
        self._ws_ptr = self._workspace.data_ptr()
        self._dims = dims  # prevent GC

        # cublasLt matmul descriptor
        handle = _get_cublaslt_handle()
        self._handle = handle
        desc = cublasLt.matmul_desc_create(
            ComputeType.COMPUTE_32F, cudaDataType.CUDA_R_32F
        )

        def _set_desc(attr, val):
            _set_attr(
                cublasLt.matmul_desc_set_attribute, desc, attr, val, ctypes.c_int32
            )

        _set_desc(cublasLt.MatmulDescAttribute.TRANSA, opa)
        _set_desc(cublasLt.MatmulDescAttribute.TRANSB, opb)
        _set_desc(
            cublasLt.MatmulDescAttribute.POINTER_MODE, cublasLt.PointerMode.DEVICE
        )
        self._desc = desc

        # Grouped matrix layouts: A=mat2, B=self, D=output (C reuses D since beta=0)
        # Layout rows/cols are the physical dimensions seen by cuBLAS (before op)
        cdt = cudaDataType.CUDA_R_16BF
        _gl = cublasLt.grouped_matrix_layout_create
        a_rows = d_m if opa == Operation.N else d_k
        a_cols = d_k if opa == Operation.N else d_m
        b_rows = d_k if opb == Operation.N else d_n
        b_cols = d_n if opb == Operation.N else d_k
        self._Ad = _gl(cdt, G, a_rows.data_ptr(), a_cols.data_ptr(), d_lda.data_ptr())
        self._Bd = _gl(cdt, G, b_rows.data_ptr(), b_cols.data_ptr(), d_ldb.data_ptr())
        self._Dd = _gl(cdt, G, d_m.data_ptr(), d_n.data_ptr(), d_ldd.data_ptr())

        # Heuristic search (run once, algo reused for all calls)
        pref = cublasLt.matmul_preference_create()
        Attr = cublasLt.MatmulPreferenceAttribute
        _set_attr(
            cublasLt.matmul_preference_set_attribute,
            pref,
            Attr.MAX_WORKSPACE_BYTES,
            CUBLASLT_WORKSPACE_BYTES,
            ctypes.c_uint64,
        )
        for attr, val in [
            (Attr.GROUPED_DESC_D_AVERAGE_ROWS, N),
            (Attr.GROUPED_DESC_D_AVERAGE_COLS, M),
            (Attr.GROUPED_AVERAGE_REDUCTION_DIM, K),
        ]:
            _set_attr(
                cublasLt.matmul_preference_set_attribute,
                pref,
                attr,
                val,
                ctypes.c_int64,
            )

        heur = np.zeros(1, dtype=cublasLt.matmul_heuristic_result_dtype)
        ret = (ctypes.c_int32 * 1)(0)
        cublasLt.matmul_algo_get_heuristic(
            handle,
            desc,
            self._Ad,
            self._Bd,
            self._Dd,
            self._Dd,
            pref,
            1,
            heur.ctypes.data,
            ctypes.addressof(ret),
        )
        if ret[0] == 0:
            raise RuntimeError("cublasLt grouped GEMM: no algorithm found")
        cublasLt.matmul_preference_destroy(pref)

        self._algo_ptr = heur["algo"].ctypes.data
        self._heur = heur  # prevent GC

        # Cached pointer offsets into _dev_ptrs
        base = self._dev_ptrs.data_ptr()
        self._dev_Aptr = base
        self._dev_Bptr = base + G * 8
        self._dev_Dptr = base + 2 * G * 8

    def __call__(self, self_list, mat2_list):
        G = self.G
        out_buf = torch.empty(
            G, self.M, self.N_padded, dtype=torch.bfloat16, device=self._dev_ptrs.device
        )

        # Fill pinned host buffer with device pointers, async copy to GPU
        h = self._pinned_raw
        out_base = out_buf.data_ptr()
        out_stride = self._out_stride_bytes
        for i in range(G):
            h[i] = mat2_list[i].data_ptr()
            h[G + i] = self_list[i].data_ptr()
            h[2 * G + i] = out_base + i * out_stride
        self._dev_ptrs.copy_(self._pinned, non_blocking=True)

        cublasLt.matmul(
            self._handle,
            self._desc,
            self._alpha_ptr,
            self._dev_Aptr,
            self._Ad,
            self._dev_Bptr,
            self._Bd,
            self._beta_ptr,
            self._dev_Dptr,
            self._Dd,  # C (unused, beta=0)
            self._dev_Dptr,
            self._Dd,  # D (output)
            self._algo_ptr,
            self._ws_ptr,
            CUBLASLT_WORKSPACE_BYTES,
            torch.cuda.current_stream().cuda_stream,
        )

        results = out_buf.unbind(0)
        if self.N != self.N_padded:
            return [r.narrow(1, 0, self.N) for r in results]
        return list(results)

    def __del__(self):
        for attr in ("_Dd", "_Bd", "_Ad"):
            d = getattr(self, attr, None)
            if d is not None:
                try:
                    cublasLt.matrix_layout_destroy(d)
                except Exception:
                    pass
        desc = getattr(self, "_desc", None)
        if desc is not None:
            try:
                cublasLt.matmul_desc_destroy(desc)
            except Exception:
                pass
