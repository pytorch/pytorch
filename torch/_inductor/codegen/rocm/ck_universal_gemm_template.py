# mypy: allow-untyped-defs, disable-error-code="attr-defined, valid-type"
import copy
import logging
import random
from typing import List, Optional

import sympy

import torch
from torch._inductor import config
from torch._inductor.codegen.cpp_utils import DTYPE_TO_CPP
from torch._inductor.codegen.rocm.ck_template import CKTemplate
from torch._inductor.codegen.rocm.compile_command import rocm_compile_command
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateKernel
from torch._inductor.ir import Buffer, Layout

from ...utils import IndentedBuffer, try_import_ck_lib


_, gen_ops_library, gen_ops_preselected, CKGemmOperation = try_import_ck_lib()


log = logging.getLogger(__name__)


def is_static_int(number):
    return isinstance(number, (int, sympy.Integer))


def torch_layout_to_ck_layout(torch_layout):
    if torch_layout.stride[-1] == 1:
        return "Row"
    elif torch_layout.stride[-2] == 1:
        return "Col"
    else:
        return None


class CKGemmTemplate(CKTemplate):
    # the JINJA template for rendering CK Universal GEMMs
    gemm_template = r"""{{version_comment}}
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    PT_EXPORT {{kernel_definition}} {
        auto gemm = {{instance_type}} {};
        auto invoker = gemm.MakeInvoker();

        auto argument = gemm.MakeArgument(
            reinterpret_cast<const {{a_element_dtype}}*>(X),
            reinterpret_cast<const {{b_element_dtype}}*>(W),
            std::array<const void*, {{ds_size}}>{ {{ds_names}} },
            reinterpret_cast<{{c_element_dtype}}*>(Y),
            M,
            N,
            K,
            LDA,
            LDB,
            std::array<ck::index_t, {{ds_size}}>{ {{ds_strides}} },
            LDC,
            1, // kBatch
            {{a_elementwise_op}},
            {{b_elementwise_op}},
            {{epilogue}} // c_elementwise_op
        );
        if (!gemm.IsSupportedArgument(argument)) {
            // we do our best to statically avoid this case in `filter_op`
            std::cerr << "invalid argument for gemm instance " << gemm.GetTypeString() << std::endl;
            argument.Print();
            return -23;
        }
        if (workspace_size) {
            *workspace_size = gemm.GetWorkSpaceSize(&argument);
            return 0;
        }
        // run the kernel
        #ifdef GENERATE_CK_STANDALONE_RUNNER
        const auto stream_config = StreamConfig{
            stream,
            /* time kernel */ 1,
            /* log level */ 1,
            /* n_cold_iter */ 100,
            /* n_hot_iter */ 100,
            /* flush_l2_cache */ 1,
            /* rotate_count */ 5};
        #else
        const auto stream_config = StreamConfig{stream, /* time kernel */ false, /* log level */ 0};
        #endif

        const float elapsed_time = invoker.Run(argument, stream_config);

        #ifdef GENERATE_CK_STANDALONE_RUNNER
        std::cout << "elapsed time: " << elapsed_time << " ms" << std::endl;
        #else
        (void)elapsed_time;
        #endif
        return 0;
    } // kernel definition
    } // extern C
    """

    standalone_runner_template = r"""
    #ifdef GENERATE_CK_STANDALONE_RUNNER
    // standalone runner for the generated CK GEMM kernel

    {{inline_utils}}

    extern "C" {
    int run_main(int argc, char** argv) {
        const int32_t M = {{M}};
        const int32_t N = {{N}};
        const int32_t K = {{K}};
        const int32_t LDA = {{LDA}};
        const int32_t LDB = {{LDB}};
        const int32_t LDC = {{LDC}};
        const int32_t LDD = {{LDD}};

        using AElementType = {{a_ck_dtype}};
        using BElementType = {{b_ck_dtype}};
        using CElementType = {{c_ck_dtype}};
        {% if has_bias %}
        using BiasElementType = {{bias_ck_dtype}};
        {% endif %}
        {% if has_scale %}
        using ScaleAElementType = {{scale_a_ck_dtype}};
        using ScaleBElementType = {{scale_b_ck_dtype}};
        {% endif %}

        using AArgType = {{a_torch_dtype}};
        using BArgType = {{b_torch_dtype}};
        using CArgType = {{c_torch_dtype}};
        {% if has_bias %}
        using BiasArgType = {{bias_torch_dtype}};
        {% endif %}
        {% if has_scale %}
        using ScaleAArgType = {{scale_a_torch_dtype}};
        using ScaleBArgType = {{scale_b_torch_dtype}};
        {% endif %}

        using ALayout = {{a_layout}};
        using BLayout = {{b_layout}};
        using CLayout = {{c_layout}};
        {% if has_bias %}
        using BiasLayout = {{bias_layout}};
        {% endif %}

        using strides_t = std::array<int32_t, 2>;

        auto get_strides = [](int32_t leading_dimension, auto layout) constexpr -> strides_t {
            if constexpr (std::is_same_v<decltype(layout), Row>) {
                return {leading_dimension, 1};
            }
            return {1, leading_dimension};
        };

        Tensor<AElementType> a_m_k ( HostTensorDescriptor ( strides_t{M, K}, get_strides(LDA, ALayout{}) ) );
        Tensor<BElementType> b_k_n ( HostTensorDescriptor ( strides_t{N, K}, get_strides(LDB, BLayout{}) ) );
        {% if has_bias %}
        Tensor<BiasElementType> d_m_n ( HostTensorDescriptor ( strides_t{M, N}, get_strides(LDD, BiasLayout{}) ) );
        {% endif %}
        {% if has_scale %}
        // NB: these are hardcoded
        Tensor<ScaleAElementType> s_a_m_n ( HostTensorDescriptor ( strides_t{M, N}, get_strides(0, Row{}) ));
        Tensor<ScaleAElementType> s_b_m_n ( HostTensorDescriptor ( strides_t{M, N}, get_strides(0, Col{}) ));
        {% endif %}

        Tensor<CElementType> c_m_n_host ( HostTensorDescriptor ( strides_t{M, N}, get_strides(LDC, CLayout{}) ) );
        Tensor<CElementType> c_m_n_device ( HostTensorDescriptor ( strides_t{M, N}, get_strides(LDC, CLayout{}) ) );

        a_m_k.GenerateTensorValue(GeneratorTensor_2<AElementType>());
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BElementType>());
        {% if has_bias %}
        d_m_n.GenerateTensorValue(GeneratorTensor_2<BiasElementType>());
        {% endif %}
        {% if has_scale %}
        s_a_m_n.GenerateTensorValue(GeneratorTensor_2<ScaleAElementType>());
        s_b_m_n.GenerateTensorValue(GeneratorTensor_2<ScaleBElementType>());
        {% endif %}
        DeviceMem a_m_k_device_buf(sizeof(AElementType) * a_m_k.mDesc.GetElementSpaceSize());
        DeviceMem b_k_n_device_buf(sizeof(BElementType) * b_k_n.mDesc.GetElementSpaceSize());
        {% if has_bias %}
        DeviceMem d_m_n_device_buf(sizeof(BiasElementType) * d_m_n.mDesc.GetElementSpaceSize());
        {% endif %}
        {% if has_scale %}
        DeviceMem s_a_m_n_device_buf(sizeof(ScaleAElementType) * s_a_m_n.mDesc.GetElementSpaceSize());
        DeviceMem s_b_m_n_device_buf(sizeof(ScaleBElementType) * s_b_m_n.mDesc.GetElementSpaceSize());
        {% endif %}
        DeviceMem c_m_n_device_buf(sizeof(CElementType) * c_m_n_device.mDesc.GetElementSpaceSize());

        a_m_k_device_buf.ToDevice(a_m_k.mData.data());
        b_k_n_device_buf.ToDevice(b_k_n.mData.data());
        {% if has_bias %}
        d_m_n_device_buf.ToDevice(d_m_n.mData.data());
        {% endif %}
        {% if has_scale %}
        s_a_m_n_device_buf.ToDevice(s_a_m_n.mData.data());
        s_b_m_n_device_buf.ToDevice(s_b_m_n.mData.data());
        {% endif %}

        {{kernel_name}}(
            static_cast<const AArgType*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<const BArgType*>(b_k_n_device_buf.GetDeviceBuffer()),
            {% if has_bias %}
            static_cast<const BiasArgType*>(d_m_n_device_buf.GetDeviceBuffer()),
            {% endif %}
            {% if has_scale %}
            static_cast<const ScaleAArgType*>(s_a_m_n_device_buf.GetDeviceBuffer()),
            static_cast<const ScaleBArgType*>(s_b_m_n_device_buf.GetDeviceBuffer()),
            {% endif %}
            static_cast<CArgType*>(c_m_n_device_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            LDA,
            LDB,
            LDC,
            LDD,
            nullptr, // workspace_size
            nullptr, // workspace
            nullptr); // stream

        hip_check_error(hipDeviceSynchronize());

        return 0;
    } // run_main
    } // extern C

    int main(int argc, char** argv) {
        return run_main(argc, argv);
    }
    // compile with: {{compile_cmd}}
    #endif // GENERATE_CK_STANDALONE_RUNNER
    """

    def __init__(
        self,
        input_nodes: List[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[List[int]] = None,
    ) -> None:
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

                #include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
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

                using BlockGemmPipelineScheduler = ck::BlockGemmPipelineScheduler;
                using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;
                using BlockGemmPipelineVersion = ck::BlockGemmPipelineVersion;
            """
        )
        return res

    def inline_utils(self):
        res = IndentedBuffer()
        res.splice(
            """
                #include "host_tensor.cpp"
                #include "device_memory.cpp"
            """
        )
        return res

    def filter_op(self, op: "CKGemmOperation"):
        """
        Determines whether a given op definition is suitable for the current
        input / output of the operation that this template implements.

        Filter is based on inputs' dtype, layout and statically inferred size.

        Returns None if the op is not suitable, otherwise returns the op to be used.
        """
        metas = [T.get_layout() for T in [*self.input_nodes, self.output_node]]
        X_meta = metas[0]
        W_meta = metas[1]
        Y_meta = metas[-1]
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

    def emit_ck_instance(self, op: "CKGemmOperation"):
        # The Jinja template for generating a C++ type alias *definition* for a Universal GEMM instance
        template_definition = r"""
    // Gemm operator {{operation_name}}
    using Operation_{{operation_name}} =
        ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
            {{template_params}}>;

"""
        # The Jinja template for generating a C++ type alias *usage* for a Universal GEMM instance
        template_type = r"""
    Operation_{{operation_name}}
"""
        template_params = []
        for field_name, field_value in op.dict_items():
            if isinstance(field_value, tuple):
                tuple_elements = ", ".join(map(str, iter(field_value)))
                if "ds" in field_name:  # element type and layout for bias
                    arg = f"/* {field_name} */ Tuple<{tuple_elements}>"
                else:  # tile shape
                    arg = f"/* {field_name} */ S<{tuple_elements}>"
                template_params.append(arg)
            else:
                if field_value is not None:
                    template_params.append(f"/* {field_name} */ {field_value}")
        return self._template_from_string(template_definition).render(
            operation_name=op.name(),
            template_params=(",\n" + 12 * " ").join(template_params),
        ), self._template_from_string(template_type).render(operation_name=op.name())

    def render(self, kernel: ROCmTemplateKernel, op: "CKGemmOperation", **kwargs) -> str:  # type: ignore[override]
        """
        The primary entry point for the code rendering process used in this template.
        """
        epilogue_nodes = kwargs.get("epilogue_nodes", None)
        assert epilogue_nodes is None or 0 == len(epilogue_nodes)
        template_buffer_node = kwargs.get("template_buffer_node", None)
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = self.input_nodes[2] if 3 == len(self.input_nodes) else None

        op = copy.deepcopy(op)

        # This parameter is converted into tuple because of change
        # from DeviceGemm_Xdl_CShuffleV3 to DeviceGemmMultiD_Xdl_CShuffle_V3.
        # The first tuple element corresponds to matmul result...
        op.c_shuffle_block_transfer_scalar_per_vector_n_per_block = (
            op.c_shuffle_block_transfer_scalar_per_vector_n_per_block,
        )

        if len(self.input_nodes) == 4:
            scale_x = self.input_nodes[2]
            scale_w = self.input_nodes[3]
            if 1 == scale_x.get_numel() and 1 == scale_w.get_numel():
                op.c_elementwise_op = "Scale"
            else:
                op.c_elementwise_op = "MultiplyMultiply"
                op.c_shuffle_dtype = "F32"
                op.ds_layouts = (
                    torch_layout_to_ck_layout(scale_x.get_layout()),
                    torch_layout_to_ck_layout(scale_w.get_layout()),
                )
                op.ds_element_dtypes = (
                    self._TORCH_DTYPE_TO_CK[scale_x.get_layout().dtype],
                    self._TORCH_DTYPE_TO_CK[scale_w.get_layout().dtype],
                )
                op.c_shuffle_block_transfer_scalar_per_vector_n_per_block += (1, 1)
        else:
            scale_x = None
            scale_w = None

        if Bias is not None:
            op.ds_layouts = (torch_layout_to_ck_layout(Bias.get_layout()),)
            op.ds_element_dtypes = ((self._TORCH_DTYPE_TO_CK[Bias.get_layout().dtype]),)
            op.c_elementwise_op = "Bilinear"
            # c_shuffle_dtype is also used for adding bias to matmul result
            # before converting down to the result dtype
            op.c_shuffle_dtype = op.acc_dtype
            # this parameter needs to be set accordingly to bias stride for correct accumulation
            if op.ds_layouts[0] == "Row":
                # bias has (N, ) shape
                bias_shuffle_block_transfer_scalar_per_vector_n_per_block = (
                    op.c_shuffle_block_transfer_scalar_per_vector_n_per_block
                )
            else:
                # bias has (M, 1) shape
                bias_shuffle_block_transfer_scalar_per_vector_n_per_block = (1,)
            # ...and the second tuple element corresponds to the bias
            op.c_shuffle_block_transfer_scalar_per_vector_n_per_block += (
                bias_shuffle_block_transfer_scalar_per_vector_n_per_block
            )

        instance_definition, instance_type = self.emit_ck_instance(op)

        version_comment = rf"""/**
* Generated code for CK inductor backend
* See {type(self).__module__}.{type(self).__qualname__}
*
* Template instance {op}
*
* {torch.__version__=}
* torch.version.git_version={getattr(torch.version, 'git_version', 'None')}
*/
"""
        epilogue = None

        if op.c_elementwise_op == "Bilinear":
            epilogue = f"Bilinear {{ {self.alpha}, {self.beta} }}"

        elif op.c_elementwise_op == "Scale":
            epilogue = "Scale { (inv_scale_w && inv_scale_x) ? (*inv_scale_w * *inv_scale_x) : 1.0f }"

        elif op.c_elementwise_op == "MultiplyMultiply":
            epilogue = "MultiplyMultiply {}"

        elif op.c_elementwise_op == "PassThrough":
            epilogue = "PassThrough {}"

        assert epilogue is not None, "CK GEMM epilogue is not set"

        res = self._template_from_string(self.gemm_template).render(
            inline_utils=self.inline_utils(),
            headers=self.header().getvalue(),
            globals=self.globals().getvalue(),
            instance_definition=instance_definition,
            kernel_definition=kernel.def_kernel(
                inputs=[X, W, scale_x, scale_w, Bias],  # type: ignore[list-item]
                outputs=[Y],
                names_str="X, W, inv_scale_x, inv_scale_w, Bias, Y",
                input_reorder=self.input_reorder,
                size_args=[
                    f"int32_t {arg}"
                    for arg in ["M", "N", "K", "LDA", "LDB", "LDC", "LDD"]
                ],
            ),
            instance_type=instance_type,
            a_element_dtype=op.a_element_dtype,
            b_element_dtype=op.b_element_dtype,
            c_element_dtype=op.c_element_dtype,
            bias_element_dtype=op.ds_element_dtypes[0] if Bias is not None else "",
            alpha=self.alpha,
            beta=self.beta,
            a_elementwise_op="PassThrough {}",
            b_elementwise_op="PassThrough {}",
            epilogue=epilogue,
            has_bias=Bias is not None,
            ds_size=1
            if Bias is not None
            else 2
            if op.c_elementwise_op == "MultiplyMultiply"
            else 0,
            ds_names=", ".join(
                ["Bias"]
                if Bias is not None
                else ["inv_scale_x", "inv_scale_w"]
                if op.c_elementwise_op == "MultiplyMultiply"
                else []
            ),
            ds_strides=", ".join(
                ["LDD"]
                if Bias is not None
                else ["0", "0"]
                if op.c_elementwise_op == "MultiplyMultiply"
                else []
            ),
            version_comment=version_comment,
        )

        if config.rocm.generate_test_runner:
            is_static_problem = all(is_static_int(arg) for arg in self.size_args())
            M, N, K, LDA, LDB, LDC, LDD = (
                self.size_args()
                if is_static_problem
                else (
                    f"std::stoi(argv[{k}])" for k, _ in enumerate(self.size_args(), 1)
                )
            )
            has_bias = Bias is not None
            has_scale = scale_x is not None and scale_w is not None
            runner_code = self._template_from_string(
                self.standalone_runner_template
            ).render(
                inline_utils=self.inline_utils().getvalue(),
                kernel_name=kernel.kernel_name,
                M=M,
                N=N,
                K=K,
                LDA=LDA,
                LDB=LDB,
                LDC=LDC,
                LDD=LDD,
                has_bias=has_bias,
                has_scale=has_scale,
                a_ck_dtype=op.a_element_dtype,
                b_ck_dtype=op.b_element_dtype,
                c_ck_dtype=op.c_element_dtype,
                bias_ck_dtype=op.ds_element_dtypes[0] if has_bias else "",
                scale_a_ck_dtype=op.ds_element_dtypes[0]
                if has_scale and 2 == len(op.ds_element_dtypes)
                else "BF16",
                scale_b_ck_dtype=op.ds_element_dtypes[1]
                if has_scale and 2 == len(op.ds_element_dtypes)
                else "BF16",
                a_torch_dtype=DTYPE_TO_CPP[X.get_layout().dtype],
                b_torch_dtype=DTYPE_TO_CPP[W.get_layout().dtype],
                c_torch_dtype=DTYPE_TO_CPP[Y.get_layout().dtype],
                bias_torch_dtype=DTYPE_TO_CPP[Bias.get_layout().dtype]
                if Bias is not None
                else "",
                scale_a_torch_dtype=DTYPE_TO_CPP[scale_x.get_layout().dtype]
                if scale_x is not None
                else "",
                scale_b_torch_dtype=DTYPE_TO_CPP[scale_w.get_layout().dtype]
                if scale_w is not None
                else "",
                a_layout=torch_layout_to_ck_layout(X.get_layout()),
                b_layout=torch_layout_to_ck_layout(W.get_layout()),
                c_layout=torch_layout_to_ck_layout(Y.get_layout()),
                bias_layout=torch_layout_to_ck_layout(Bias.get_layout())
                if Bias is not None
                else "",
                compile_cmd=rocm_compile_command(
                    ["<source_file_name>"], "<executable_name>", "exe"
                ),
            )
            res += runner_code

        return res

    def _is_rcr_f16(self):
        X_meta, W_meta, Y_meta = (
            T.get_layout() for T in [*self.input_nodes, self.output_node]
        )
        X_dtype, W_dtype, Y_dtype = (
            self._TORCH_DTYPE_TO_CK[m.dtype] for m in (X_meta, W_meta, Y_meta)
        )
        X_layout, W_layout, Y_layout = (
            torch_layout_to_ck_layout(m) for m in (X_meta, W_meta, Y_meta)
        )

        return (
            X_dtype == "F16"
            and W_dtype == "F16"
            and Y_dtype == "F16"
            and X_layout == "Row"
            and W_layout == "Col"
            and Y_layout == "Row"
        )

    def gen_ops(self):
        """
        Creates a list of `CKGemmOperation` instances that match the GEMM operation this template represents.
        The instances are guaranteed to have the correct layout, dtype and dimension padding for the GEMM input arguments.

        An instance may invalidate the GEMM configuration at runtime.
        Such instances will be assigned +inf runtime by the autotune process.
        """
        unfiltered_instances = (
            gen_ops_preselected()
            if config.rocm.use_preselected_instances and self._is_rcr_f16()
            else gen_ops_library()
        )
        filtered_instances = list(
            filter(lambda op: self.filter_op(op), unfiltered_instances)
        )
        # NB: when using a fixed list order, most likely we will pick the subset of instances
        # which are very similar to each other. Randomizing the choice seems to solve this.
        random.seed(-11)
        chosen_instances = (
            random.sample(
                filtered_instances,
                min(len(filtered_instances), config.rocm.n_max_profiling_configs),
            )
            if config.rocm.n_max_profiling_configs
            else filtered_instances
        )
        log.debug(
            "generated %d ck instances after filter: %s",
            len(chosen_instances),
            chosen_instances,
        )
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
        for op in ops:
            template.maybe_append_choice(
                choices,
                op=op,
            )

    def size_args(self):
        X = self.input_nodes[0]
        W = self.input_nodes[1]
        Bias = self.input_nodes[2] if len(self.input_nodes) == 3 else None
        Y = self.output_node

        M = X.get_size()[0]
        K = X.get_size()[1]
        N = W.get_size()[1]
        LDA = X.get_stride()[0 if X.get_stride()[1] == 1 else 1]
        LDB = W.get_stride()[0 if W.get_stride()[1] == 1 else 1]
        LDC = Y.get_stride()[0 if Y.get_stride()[1] == 1 else 1]
        LDD = (
            0
            if Bias is None
            else Bias.get_stride()[0 if Bias.get_stride()[1] == 1 else 1]
        )

        return M, N, K, LDA, LDB, LDC, LDD
