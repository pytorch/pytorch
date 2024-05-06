from dataclasses import fields, replace
from functools import lru_cache
import logging
import os
import random
import subprocess
import sympy
from typing import List, Optional

from torch._inductor import config
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateKernel
from torch._inductor.codegen.rocm.ck_template import CKTemplate
from torch._inductor.codegen.rocm.ck_universal_gemm_op import CKGemmOperation
from torch._inductor.ir import Buffer, Layout

log = logging.getLogger(__name__)

from ...utils import IndentedBuffer

def is_static_int(number):
    return isinstance(number, (int, sympy.Integer))

class CKGemmTemplate(CKTemplate):
    # the JINJA template for rendering CK Universal GEMMs
    gemm_template = r"""
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    {{kernel_definition}} {
        auto gemm = {{instance_type}} {};
        auto invoker = gemm.MakeInvoker();

        constexpr auto M = {{M}};
        constexpr auto N = {{N}};
        constexpr auto K = {{K}};
        constexpr auto StrideA = std::is_same_v<{{a_layout}}, Row> ? K : M;
        constexpr auto StrideB = std::is_same_v<{{b_layout}}, Row> ? N : K;
        constexpr auto StrideC = std::is_same_v<{{c_layout}}, Row> ? N : M;
        constexpr auto KBatch = 1; // split k into batches

        auto argument = gemm.MakeArgument(
            reinterpret_cast<const {{a_element_dtype}}*>(X),
            reinterpret_cast<const {{b_element_dtype}}*>(W),
            reinterpret_cast<{{c_element_dtype}}*>(Y),
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC,
            KBatch,
            {{a_elementwise_op}} {},
            {{b_elementwise_op}} {},
            {{c_elementwise_op}} {}
        );
        if (!gemm.IsSupportedArgument(argument)) {
            // magic spell to assign +inf to benchmarking time in select_algorithm.py:1052 (1/2)
            std::cerr << "invalid argument for gemm instance " << gemm.GetTypeString() << std::endl;
            argument.Print();
            return -23;
        }
        // run the kernel
        float elapsed_time = invoker.Run(argument, StreamConfig{stream, /* time kernel */ false, /* log level */ kDEBUG_LOG});
        return 0;
    } // kernel definition
    } // extern C
    """

    # manually selected (through benchmarking) F16/F16/F16 Row/Col/Row instances
    preselected_instances = r"""
    # Compute-friendly, 7 instances
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 256, 224, 256, 64, 8, 8, 16, 16, 7, 8, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 2, S<1, 32, 1, 8>, 8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 256, 128, 128, 64, 8, 8, 32, 32, 2, 2, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 8>, 8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 256, 128, 128, 64, 8, 8, 32, 32, 2, 2, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 8>, 8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 256, 128, 128, 64, 8, 8, 32, 32, 2, 2, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 8>, 8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v5>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::Default,    256, 128, 128, 64, 8, 8, 32, 32, 2, 2, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 8>, 8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::Default,    256, 128, 128, 64, 8, 8, 32, 32, 2, 2, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 8>, 8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::Default,    256, 128, 128, 64, 8, 8, 32, 32, 2, 2, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 8>, 8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v5>,
    # Memory-friendly, 10 instances
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::Default,    128, 16,  32,  64, 8, 8, 16, 16, 1, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 16, 1, 8>, 4, BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v2>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 128, 16,  32,  64, 8, 8, 16, 16, 1, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 16, 1, 8>, 4, BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v2>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 128, 16,  64,  64, 8, 8, 16, 16, 1, 2, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 2, S<1, 16, 1, 8>, 8, BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v2>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 128, 32,  64,  64, 8, 8, 32, 32, 1, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 16, 1, 8>, 8, BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v2>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 128, 32,  128, 64, 8, 8, 32, 32, 1, 2, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 16, 1, 8>, 8, BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v2>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::Default,    128, 32,  16,  64, 8, 8, 16, 16, 1, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 4>, 4, BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v2>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 128, 32,  16,  64, 8, 8, 16, 16, 1, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 4>, 4, BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v2>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 128, 64,  16,  64, 8, 8, 16, 16, 2, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 2, 1, S<1, 64, 1, 2>, 8, BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v2>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 128, 64,  32,  64, 8, 8, 32, 32, 1, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 4>, 8, BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v2>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::MNKPadding, 128, 128, 32,  64, 8, 8, 32, 32, 2, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 2, 1, S<1, 32, 1, 4>, 8, BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v2>,
    # Latency-friendly, 2 instances
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::Default,    128, 16, 32,   64, 8, 8, 16, 16, 1, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 16, 1, 8>, 4, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1>,
    DeviceGemm_Xdl_CShuffleV3<Row, Col, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmSpecialization::Default,    128, 32, 16,   64, 8, 8, 16, 16, 1, 1, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 4>, 4, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1>,
    """

    def __init__(
        self,
        input_nodes: List[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[List[int]] = None,
    ):
        super().__init__(
            "ck_gemm_template",
            input_nodes=input_nodes,
            layout=layout,
            input_reorder=input_reorder,
        )
        self.alpha = alpha
        self.beta = beta

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK GEMM header(s)

                #include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK GEMM globals

                using Row = ck::tensor_layout::gemm::RowMajor;
                using Col = ck::tensor_layout::gemm::ColumnMajor;

                using cudaStream_t = hipStream_t;

                using BlockGemmPipelineScheduler = ck::BlockGemmPipelineScheduler;
                using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;
                using BlockGemmPipelineVersion = ck::BlockGemmPipelineVersion;
            """
        )
        return res

    def filter_op(self, op: CKGemmOperation) -> Optional[CKGemmOperation]:
        """
        Determines whether a given op definition is suitable for the current
        input / output of the operation that this template implements.

        Filter is based on inputs' dtype, layout and statically inferred size.

        Returns None if the op is not suitable, otherwise returns the op to be used.
        """

        # TBD return None if alignment or layout or dtype is invalid
        def torch_layout_to_ck_layout(torch_layout):
            if torch_layout.stride[-1] == 1:
                return "Row"
            elif torch_layout.stride[-2] == 1:
                return "Col"
            else:
                return None

        X_meta, W_meta, Y_meta = map(
            lambda T: T.get_layout(), [*self.input_nodes, self.output_node]
        )
        # disable the instance if dtypes don't match
        if op.a_element_dtype != self._TORCH_DTYPE_TO_CK[X_meta.dtype]:
            return None
        if op.b_element_dtype != self._TORCH_DTYPE_TO_CK[W_meta.dtype]:
            return None
        if op.c_element_dtype != self._TORCH_DTYPE_TO_CK[Y_meta.dtype]:
            return None
        # disable the instance if layouts don't match
        if op.a_layout != torch_layout_to_ck_layout(X_meta):
            return None
        if op.b_layout != torch_layout_to_ck_layout(W_meta):
            return None
        if op.c_layout != torch_layout_to_ck_layout(Y_meta):
            return None
        # try to avoid launching the instance with invalid problem size
        # see GridwiseGemm_xdl_cshuffle_v3::CheckValidity

        M = X_meta.size[-2]
        K = X_meta.size[-1]
        N = W_meta.size[-1]

        if is_static_int(M):
            if not any(
                m_padding in op.gemm_specialization
                for m_padding in ["MPadding", "MNPadding", "MKPadding", "MNKPadding"]
            ):
                if M % op.m_per_block != 0:
                    return None
        if is_static_int(N):
            if not any(
                n_padding in op.gemm_specialization
                for n_padding in ["NPadding", "MNPadding", "NKPadding", "MNKPadding"]
            ):
                if N % op.n_per_block != 0:
                    return None
        if is_static_int(K):
            if not any(
                k_padding in op.gemm_specialization
                for k_padding in ["KPadding", "MKPadding", "NKPadding", "MNKPadding"]
            ):
                if K % op.k_per_block != 0:
                    return None

        a_contig_size = (
            K if op.a_layout == "Row" else M if op.a_layout == "Col" else None
        )
        if (
            is_static_int(a_contig_size)
            and a_contig_size % op.a_block_transfer_src_scalar_per_vector != 0
        ):
            return None
        b_contig_size = (
            N if op.b_layout == "Row" else K if op.b_layout == "Col" else None
        )
        if (
            is_static_int(b_contig_size)
            and b_contig_size % op.b_block_transfer_src_scalar_per_vector != 0
        ):
            return None
        c_contig_size = (
            N if op.c_layout == "Row" else M if op.c_layout == "Col" else None
        )
        if (
            is_static_int(c_contig_size)
            and c_contig_size
            % op.c_shuffle_block_transfer_scalar_per_vector_n_per_block
            != 0
        ):
            return None

        # TBD disable instances with invalid number of pipeline prefetch stages
        # It will avoid compiling a small percentage of unrunnable instances which fail the gemm argument check

        return op

    def emit_ck_instance(self, op: CKGemmOperation):
        # The Jinja template for generating a C++ type alias *definition* for a Universal GEMM instance
        template_definition = r"""
    // Gemm operator {{operation_name}}
    using Operation_{{operation_name}} =
        ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
            {{template_params}}>;

"""
        # The Jinja template for generating a C++ type alias *usage* for a Universal GEMM instance
        template_type = r"""
    Operation_{{operation_name}}
"""
        template_params = []
        for f in fields(op):
            field_value = getattr(op, f.name)
            if isinstance(field_value, tuple):
                template_params.append(
                    f"/* {f.name} */ S<{', '.join(map(str, iter(field_value)))}>"
                )
            else:
                if field_value is not None:
                    template_params.append(f"/* {f.name} */ {field_value}")
        return self._template_from_string(template_definition).render(
            operation_name=op.name(),
            template_params=(",\n" + 12 * " ").join(template_params),
        ), self._template_from_string(template_type).render(operation_name=op.name())

    def render(self, kernel: CUDATemplateKernel, op: CKGemmOperation, **kwargs) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        """
        epilogue_nodes = kwargs.get("epilogue_nodes", None)
        assert epilogue_nodes is None or 0 == len(epilogue_nodes)
        template_buffer_node = kwargs.get("template_buffer_node", None)
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        instance_definition, instance_type = self.emit_ck_instance(op)
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None  # TBD support gemm_bias

        return self._template_from_string(self.gemm_template).render(
            headers=self.header().getvalue(),
            globals=self.globals().getvalue(),
            instance_definition=instance_definition,
            kernel_definition=kernel.def_kernel(
                inputs=[X, W, Bias],
                outputs=[Y],
                names_str="X, W, Bias, Y",
                input_reorder=self.input_reorder,
            ),
            instance_type=instance_type,
            M=kernel.size(X, -2),
            K=kernel.size(X, -1),
            N=kernel.size(W, -1),
            a_elementwise_op=op.a_elementwise_op,
            b_elementwise_op=op.b_elementwise_op,
            c_elementwise_op=op.c_elementwise_op,
            a_element_dtype=op.a_element_dtype,
            b_element_dtype=op.b_element_dtype,
            c_element_dtype=op.c_element_dtype,
            a_layout=op.a_layout,
            b_layout=op.b_layout,
            c_layout=op.c_layout,
        )

    def _parse_instances(self, str_instances: List[str]) -> List[CKGemmOperation]:
        """
        Parse the lines containing Universal Gemm template instances into `CKGemmOperation` instances
        """

        def maybe_int(s):
            try:
                return int(s)
            except ValueError:
                return s

        op_instances = []
        for line in str_instances:
            s_template_args = line.split("DeviceGemm_Xdl_CShuffleV3")[-1].strip("<>, ")
            template_args = []
            i_current = 0
            while i_current < len(s_template_args):
                if s_template_args[i_current] == " ":
                    # skip whitespace
                    i_current += 1
                    continue
                elif s_template_args[i_current : i_current + 2] == "S<":
                    # parse template S<Index...>
                    i_next = s_template_args.find(">", i_current)
                    template_args.append(
                        tuple(
                            map(int, s_template_args[i_current + 2 : i_next].split(","))
                        )
                    )
                    i_current = i_next + 2
                else:
                    # all string attributes must be either type aliases or global constants in C++
                    i_next = s_template_args.find(",", i_current)
                    template_args.append(
                        maybe_int(
                            s_template_args[
                                i_current : i_next if i_next != -1 else None
                            ]
                        )
                    )
                    if i_next != -1:
                        i_current = i_next + 1
                if i_next == -1:
                    break
            # pad with `None`s for the fields which are not defined in the instance
            new_instance = CKGemmOperation(
                *template_args,
                *((None,) * (len(fields(CKGemmOperation)) - len(template_args))),
            )
            # the last 2 template parameters are optional
            # if they are absent, substitute them with default values from Universal Gemm C++ template declaration
            if new_instance.a_compute_dtype is None:
                new_instance.a_compute_dtype = new_instance.c_element_dtype
            if new_instance.b_compute_dtype is None:
                new_instance.b_compute_dtype = new_instance.c_element_dtype

            op_instances.append(new_instance)
        return op_instances

    def _default_instances(self) -> List[CKGemmOperation]:
        # fallback: known working op instance for problem size M=2240 K=256 N=2048
        # all string attributes must be either type aliases or global constants in C++

        return [
            CKGemmOperation(
                a_layout="Row",
                b_layout="Row",
                c_layout="Row",
                a_element_dtype="F16",
                b_element_dtype="F16",
                c_element_dtype="F16",
                a_compute_dtype="F16",
                b_compute_dtype="F16",
                acc_dtype="F32",
                c_shuffle_dtype="F16",
                a_elementwise_op="PassThrough",
                b_elementwise_op="PassThrough",
                c_elementwise_op="PassThrough",
                gemm_specialization="GemmSpecialization::Default",
                block_size=256,
                m_per_block=224,
                n_per_block=256,
                k_per_block=64,
                a_k1=8,
                b_k1=2,
                m_per_xdl=16,
                n_per_xdl=16,
                m_xdl_per_wave=7,
                n_xdl_per_wave=8,
                a_block_transfer_thread_cluster_lengths_ak0_m_ak1=(8, 32, 1),
                a_block_transfer_thread_cluster_arrange_order=(1, 0, 2),
                a_block_transfer_src_access_order=(1, 0, 2),
                a_block_transfer_src_vector_dim=2,
                a_block_transfer_src_scalar_per_vector=8,
                a_block_transfer_dst_scalar_per_vector_ak1=8,
                a_block_lds_extra_m=0,
                b_block_transfer_thread_cluster_lengths_bk0_n_bk1=(8, 32, 1),
                b_block_transfer_thread_cluster_arrange_order=(0, 2, 1),
                b_block_transfer_src_access_order=(0, 2, 1),
                b_block_transfer_src_vector_dim=1,
                b_block_transfer_src_scalar_per_vector=8,
                b_block_transfer_dst_scalar_per_vector_bk1=2,
                b_block_lds_extra_n=0,
                c_shuffle_m_xdl_per_wave_per_shuffle=1,
                c_shuffle_n_xdl_per_wave_per_shuffle=2,
                c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                    1,
                    32,
                    1,
                    8,
                ),
                c_shuffle_block_transfer_scalar_per_vector_n_per_block=8,
                block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
                block_gemm_pipeline_version="BlockGemmPipelineVersion::v3",
            )
        ]

    @lru_cache(None)
    def _gen_ops_library(self) -> List[CKGemmOperation]:
        """
        Parse the Universal Gemm instances defined in the composable kernel library folder.
        """
        grep_result = subprocess.run(
            [
                "grep",
                "-inR",
                "DeviceGemm_Xdl_CShuffleV3",
                os.path.join(config.rocm.ck_dir, "library"),
            ],
            capture_output=True,
            text=True,
        )

        op_instances = self._parse_instances(grep_result.stdout.strip().split("\n"))

        log.debug(f"ck instances from library: {len(op_instances)}")

        schedulers = [
            "BlockGemmPipelineScheduler::Intrawave",
            "BlockGemmPipelineScheduler::Interwave",
        ]
        gemm_specs = [
            "GemmSpecialization::Default",
            "GemmSpecialization::MPadding",
            "GemmSpecialization::NPadding",
            "GemmSpecialization::KPadding",
            "GemmSpecialization::MNPadding",
            "GemmSpecialization::MKPadding",
            "GemmSpecialization::NKPadding",
            "GemmSpecialization::MNKPadding",
        ]

        # substitute templated args by looping through their domains
        substitute_instances = []
        for instance in op_instances:
            sub_scheduler = instance.block_gemm_pipeline_scheduler == "BlkGemmPipeSched"
            sub_spec = instance.gemm_specialization == "GemmSpec"
            schedulers_range = (
                schedulers
                if sub_scheduler
                else [instance.block_gemm_pipeline_scheduler]
            )
            spec_range = gemm_specs if sub_spec else [instance.gemm_specialization]
            for scheduler in schedulers_range:
                for spec in spec_range:
                    substitute_instances.append(
                        replace(
                            instance,
                            block_gemm_pipeline_scheduler=scheduler,
                            gemm_specialization=spec,
                        )
                    )

        return substitute_instances

    @lru_cache(None)
    def _gen_ops_preselected(self) -> List[CKGemmOperation]:
        """
        Parse the preselected Universal Gemm instances
        """
        return self._parse_instances(self.preselected_instances.split("\n"))

    def gen_ops(self) -> List[CKGemmOperation]:
        """
        Creates a list of `CKGemmOperation` instances that match the GEMM operation this template represents.
        The instances are guaranteed to have the correct layout, dtype and dimension padding for the GEMM input arguments.

        An instance may invalidate the GEMM configuration at runtime; such instances will be assigned +inf runtime by the autotune process.
        """
        unfiltered_instances = (
            self._gen_ops_preselected()
            if config.rocm.use_preselected_instances
            else self._gen_ops_library()
        )
        filtered_instances = list(
            filter(lambda op: self.filter_op(op), unfiltered_instances)
        )
        # NB: when using a fixed list order, most likely we will pick the subset of instances
        # which are very similar to each other. Randomizing the choice seems to solve this.
        random.seed(-11)
        chosen_instances = (
            random.sample(filtered_instances, config.rocm.n_max_profiling_configs)
            if config.rocm.n_max_profiling_configs
            else filtered_instances
        )
        log.debug(f"generated {len(chosen_instances)} ck instances: {chosen_instances}")
        return chosen_instances

    @staticmethod
    def add_ck_gemm_choices(
        choices,
        layout,
        input_nodes,
        alpha=1,
        beta=0,
        input_reorder=None,
    ):
        """
        Add Composable Kernel Universal GEMM instance choices to the auto-tuning list.
        """
        template = CKGemmTemplate(
            input_nodes,
            layout,
            alpha=alpha,
            beta=beta,
            input_reorder=input_reorder,
        )
        ops = template.gen_ops()
        log.debug(f"ck instance choices: {ops}")
        for op in ops:
            template.maybe_append_choice(
                choices,
                op=op,
            )
