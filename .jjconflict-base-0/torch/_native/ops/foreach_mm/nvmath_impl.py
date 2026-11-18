"""Cached cublasLt grouped GEMM via nvmath bindings."""

import ctypes

from cuda.bindings.runtime import cudaDataType  # pyrefly: ignore[missing-import]

from nvmath.bindings import cublasLt  # pyrefly: ignore[missing-import]
from nvmath.bindings.cublas import (  # pyrefly: ignore[missing-import]
    ComputeType,
    Operation,
)

import torch


_cublaslt_workspaces: dict[tuple, torch.Tensor] = {}


def _get_cublaslt_workspace(device="cuda"):
    ws_size = torch.backends.cuda.cublaslt_workspace_size()
    key = (torch.device(device), ws_size)
    if key not in _cublaslt_workspaces:
        _cublaslt_workspaces[key] = torch.empty(
            ws_size, dtype=torch.uint8, device=device
        )
    return _cublaslt_workspaces[key], ws_size


def _set_attr(setter, handle, attr, val, ctype):
    buf = (ctype * 1)(val)
    setter(handle, attr, ctypes.addressof(buf), ctypes.sizeof(buf))


class ForeachMMCublasLt:
    """Cached cublasLt grouped GEMM. Supports uniform or mixed shapes per group."""

    def __init__(
        self,
        shapes,
        G,
        a_row_major=True,
        b_row_major=True,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        """
        Args:
            shapes: tuple of (M, N, K) triples, one per group
            G: number of groups (must equal len(shapes))
        """
        if dtype != torch.bfloat16:
            raise ValueError(f"ForeachMMCublasLt only supports bf16, got {dtype}")

        self.G = G
        self._dtype = dtype
        elem_size = dtype.itemsize
        alignment = 16 // elem_size

        # cuBLAS col-major: C^T = mat2^T * self^T
        # cuBLAS-A = mat2, cuBLAS-B = self
        opa = Operation.N if b_row_major else Operation.T
        opb = Operation.N if a_row_major else Operation.T

        # Per-group dimensions
        Ms, Ns, Ks = zip(*shapes)
        ldas = [n if b_row_major else k for n, k in zip(Ns, Ks)]
        ldbs = [k if a_row_major else m for m, k in zip(Ms, Ks)]
        N_paddeds = [(n + alignment - 1) // alignment * alignment for n in Ns]

        for vals in (Ms, Ns, Ks, ldas, ldbs, N_paddeds):
            for v in vals:
                if v > 2**31 - 1:
                    raise ValueError(
                        f"ForeachMMCublasLt: dimension {v} exceeds int32 range"
                    )

        # Per-group output layout: allocate one contiguous buffer with
        # max(M) x max(N_padded) per group, then narrow each slice to actual (M, N)
        self._out_Ns = Ns
        self._out_Ms = Ms
        self._out_N_paddeds = N_paddeds
        self._max_M = max(Ms)
        self._max_N_padded = max(N_paddeds)
        self._out_stride_bytes = self._max_M * self._max_N_padded * elem_size
        self._uniform = len(set(Ms)) == 1 and len(set(Ns)) == 1
        # ldd must match the physical stride in the contiguous output buffer
        N_paddeds = [self._max_N_padded] * G

        # Device-side dim arrays (6 x G)
        # cublas_m=N, cublas_n=M, cublas_k=K
        cublas_ms = list(Ns)
        cublas_ns = list(Ms)
        cublas_ks = list(Ks)
        dims = torch.tensor(
            [cublas_ms, cublas_ns, cublas_ks, ldas, ldbs, N_paddeds],
            dtype=torch.int32,
            device=device,
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

        self._workspace, self._ws_bytes = _get_cublaslt_workspace(device)
        self._ws_ptr = self._workspace.data_ptr()
        self._dims = dims  # prevent GC

        # cublasLt matmul descriptor
        # Reuse PyTorch's cuBLAS handle (cublasHandle_t is valid as cublasLtHandle_t)
        handle = torch.cuda.current_blas_handle()
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
            self._ws_bytes,
            ctypes.c_uint64,
        )
        avg_N = sum(Ns) // G
        avg_M = sum(Ms) // G
        avg_K = sum(Ks) // G
        for attr, val in [
            (Attr.GROUPED_DESC_D_AVERAGE_ROWS, avg_N),
            (Attr.GROUPED_DESC_D_AVERAGE_COLS, avg_M),
            (Attr.GROUPED_AVERAGE_REDUCTION_DIM, avg_K),
        ]:
            _set_attr(
                cublasLt.matmul_preference_set_attribute,
                pref,
                attr,
                val,
                ctypes.c_int64,
            )

        # cublasLtMatmulHeuristicResult_t is 96 bytes; algo field is at offset 0
        heur_buf = (ctypes.c_byte * 96)()
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
            ctypes.addressof(heur_buf),
            ctypes.addressof(ret),
        )
        if ret[0] == 0:
            raise RuntimeError("cublasLt grouped GEMM: no algorithm found")
        cublasLt.matmul_preference_destroy(pref)

        self._algo_ptr = ctypes.addressof(heur_buf)
        self._heur_buf = heur_buf  # prevent GC

        # Cached pointer offsets into _dev_ptrs
        base = self._dev_ptrs.data_ptr()
        ptr_size = torch.int64.itemsize
        self._dev_Aptr = base
        self._dev_Bptr = base + G * ptr_size
        self._dev_Dptr = base + 2 * G * ptr_size

    def __call__(self, self_list, mat2_list):
        G = self.G
        max_M = self._max_M
        max_Np = self._max_N_padded

        # Single contiguous allocation, sliced per group
        out_buf = torch.empty(
            G, max_M, max_Np, dtype=self._dtype, device=self._dev_ptrs.device
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
            self._ws_bytes,
            torch.cuda.current_stream().cuda_stream,
        )

        # unbind is a single C++ call creating all G views at once.
        # Narrow only when output shape differs from the buffer slot size.
        results = out_buf.unbind(0)
        if self._uniform:
            M, N = self._out_Ms[0], self._out_Ns[0]
            if M != self._max_M or N != self._max_N_padded:
                return [r[:M, :N] for r in results]
            return list(results)
        return [r[: self._out_Ms[i], : self._out_Ns[i]] for i, r in enumerate(results)]

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
