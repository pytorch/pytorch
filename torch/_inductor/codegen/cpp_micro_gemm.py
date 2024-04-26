from collections import namedtuple

import torch

from .. import ir
from ..utils import parallel_num_threads
from .common import KernelTemplate
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import DTYPE_TO_CPP, GemmBlocking, value_to_cpp


class CppMicroGemm:
    DECLARE_KERNEL = r"""
template <bool accum>
inline void {{kernel_name}}(
    const {{input_t}}* __restrict__ A,
    const {{input_t}}* __restrict__ B,
    {{output_t}}* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
)
"""

    def __init__(
        self,
        name,
        input_dtype,
        output_dtype,
        compute_dtype,
        register_blocking,
        alpha=1,
    ):
        self.name = name
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.compute_dtype = compute_dtype
        self.register_blocking = register_blocking
        self.alpha = alpha

    def get_common_options(self):
        return {
            "kernel_name": self.name,
            "input_t": DTYPE_TO_CPP[self.input_dtype],
            "output_t": DTYPE_TO_CPP[self.output_dtype],
            "compute_t": DTYPE_TO_CPP[self.compute_dtype],
            "alpha": self.alpha,
        }

    def get_kernel_declaration(self):
        options = self.get_common_options()
        return KernelTemplate._template_from_string(self.DECLARE_KERNEL).render(options)

    def codegen_define(self) -> str:
        raise NotImplementedError

    def codegen_call(
        self,
        kernel: CppTemplateKernel,
        A: ir.Buffer,
        B: ir.Buffer,
        C: ir.Buffer,
        accum: bool,
    ) -> str:
        A_ptr = f"&({kernel.index(A, [0, 0])})"
        B_ptr = f"&({kernel.index(B, [0, 0])})"
        C_ptr = f"&({kernel.index(C, [0, 0])})"
        M = kernel.size(C, 0)
        N = kernel.size(C, 1)
        K = kernel.size(A, 1)
        lda = kernel.stride(A, 0)
        ldb = kernel.stride(B, 0)
        ldc = kernel.stride(C, 0)
        return f"{self.name}<{value_to_cpp(accum, 'bool')}>({A_ptr}, {B_ptr}, {C_ptr}, {M}, {N}, {K}, {lda}, {ldb}, {ldc});"


class CppMicroGemmRef(CppMicroGemm):
    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < K; ++k) {
                C[m * ldc + n] =
                    ({{compute_t}})C[m * ldc + n] * accum
                        + ({{compute_t}})A[m * lda + k] * ({{compute_t}})B[k * ldb + n] * {{alpha}};
            }
        }
    }
}
"""

    def __init__(self, name, input_dtype, output_dtype, compute_dtype, alpha):
        super().__init__(
            name, input_dtype, output_dtype, compute_dtype, GemmBlocking(1, 1, 1), alpha
        )

    def codegen_define(self) -> str:
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            **self.get_common_options(),
        }
        return KernelTemplate._template_from_string(self.TEMPLATE_ENTRY).render(options)


class CppMicroGemmFP32AVX(CppMicroGemm):
    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    TORCH_CHECK(N % {{block_n}} == 0, "N dimension must be multiple of {{block_n}}");
    TORCH_CHECK(K % {{block_k}} == 0, "K dimension must be multiple of {{block_k}}");
    // TODO: loop unroll
    for (int64_t m = 0; m < M; m += {{block_m}}) {
        int64_t block_m = std::min<int64_t>(M - m, {{block_m}});
        for (int64_t n = 0; n < N; n += {{block_n}}) {
            switch (block_m) {
            {% for b in range(block_m, 0, -1) %}
            case {{b}}:
                {{kernel_name}}_kernel<{{b}}, {{block_n}}, accum>(
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    {{block_k}},
                    lda,
                    ldb,
                    ldc
                );
            {% endfor %}
            default:
                TORCH_CHECK(false, "Unsupported block_m");
            }
        }
    }
}
"""

    TEMPLATE_KERNEL = r"""
template <int64_t BLOCK_M, int64_t BLOCK_N, bool accum>
inline void {{kernel_name}}_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
) {
    using Vectorized = at::vec::Vectorized<float>;
    constexpr auto VLEN = Vectorized::size();
    constexpr auto ROWS = BLOCK_M;
    constexpr auto COLS = BLOCK_N / VLEN;

    Vectorized va;
    at::vec::VectorizedN<float, COLS> vb;
    at::vec::VectorizedN<float, ROWS*COLS> vc;

    auto loadc = [&](auto i) {
        if constexpr (accum) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            vc[i] = Vectorized::loadu(C + row * ldc + col * VLEN);
        } else {
            vc[i] = Vectorized(0.0f);
        }
    };
    c10::ForcedUnroll<ROWS * COLS>{}(loadc);

    auto compute = [&, COLS](auto i, int k) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;

        if constexpr (col == 0) {
            {% if alpha != 1 %}
            va = Vectorized(A[row * lda + k] * {{alpha}});
            {% else %}
            va = Vectorized(A[row * lda + k]);
            {% endif %}
        }

        if constexpr (row == 0) {
            vb[col] = Vectorized::loadu(B + k * ldb + col * VLEN);
        }

        constexpr int idx = row * COLS + col;
        vc[idx] = at::vec::fmadd(va, vb[col], vc[idx]);
    };

    // TODO: unroll k
    for (int k = 0; k < K; ++k) {
        c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
    }

    // store to C
    auto storec = [&](auto i) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        vc[i].store(C + row * ldc + col * VLEN);
    };
    c10::ForcedUnroll<ROWS * COLS>{}(storec);
}
"""

    def codegen_define(self) -> str:
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            "block_m": self.register_blocking.block_m,
            "block_n": self.register_blocking.block_n,
            "block_k": self.register_blocking.block_k,
            **self.get_common_options(),
        }
        result = KernelTemplate._template_from_string(self.TEMPLATE_KERNEL).render(
            options
        )
        result += KernelTemplate._template_from_string(self.TEMPLATE_ENTRY).render(
            options
        )
        return result


CppMicroGemmConfig = namedtuple(
    "CppMicroGemmConfig",
    ["cls", "input_dtype", "output_dtype", "compute_dtype", "register_blocking"],
)


micro_gemm_configs = [
    # TODO: decide register_blocking per cpu arch, assume avx512 now
    CppMicroGemmConfig(
        CppMicroGemmFP32AVX,
        torch.float32,
        torch.float32,
        torch.float32,
        GemmBlocking(8, 32, 1),
    ),
    CppMicroGemmConfig(
        CppMicroGemmFP32AVX,
        torch.float32,
        torch.float32,
        torch.float32,
        GemmBlocking(16, 16, 1),
    ),
]


def create_micro_gemm(
    name,
    m,
    n,
    k,
    input_dtype,
    output_dtype=None,
    compute_dtype=None,
    alpha=1,
    num_threads=-1,
    use_ref=True,
) -> CppMicroGemm:
    def create_from_config(config: CppMicroGemmConfig):
        return config.cls(
            name,
            config.input_dtype,
            config.output_dtype,
            config.compute_dtype,
            config.register_blocking,
            alpha,
        )

    assert isinstance(n, int) or n.is_number
    assert isinstance(k, int) or k.is_number
    if output_dtype is None:
        output_dtype = input_dtype
    if compute_dtype is None:
        compute_dtype = input_dtype
    if num_threads < 0:
        num_threads = parallel_num_threads()
    matched_configs = []
    for config in micro_gemm_configs:
        if (
            config.input_dtype == input_dtype
            and config.output_dtype == output_dtype
            and config.compute_dtype == compute_dtype
        ):
            score = 0
            block_m, block_n, block_k = config.register_blocking
            if n % block_n == 0:
                score += 1
            if k % block_k == 0:
                score += 1
            if m % block_m == 0:
                score += 1
            n_blocks = (n + block_n - 1) // block_n
            if n_blocks >= num_threads:
                score += 1
            matched_configs.append((score, config))
    if len(matched_configs) == 0 or use_ref:
        return CppMicroGemmRef(name, input_dtype, output_dtype, compute_dtype, alpha)
    return create_from_config(max(matched_configs, key=lambda x: x[0])[1])
