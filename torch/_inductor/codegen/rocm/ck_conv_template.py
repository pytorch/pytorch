# mypy: allow-untyped-defs
import copy
import logging
import random

from torch._inductor.virtualized import V


try:
    import ck4inductor  # type: ignore[import]
except ImportError:
    ck4inductor = None

if ck4inductor is not None:
    from ck4inductor.grouped_conv_fwd.gen_instances import (  # type: ignore[import]
        gen_conv_ops_library,
    )
    from ck4inductor.grouped_conv_fwd.op import (  # type: ignore[import]  # noqa: TCH002
        CKGroupedConvFwdOp,
    )
else:

    def gen_conv_ops_library():
        return []


from torch._inductor import config
from torch._inductor.codegen.rocm.ck_template import CKTemplate
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateKernel
from torch._inductor.utils import IndentedBuffer


log = logging.getLogger(__name__)


def torch_layout_to_ck_layouts(torch_layout):
    # logically, torch tensors are always NCHW,
    # and channels-last memory layout is visible in the strides
    if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
        # when input or output is NCHW
        # NB: torch.conv2d result is always NCHW
        return ["NGCHW", "GKCYX", "NGKHW"]
    elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-3], 1):
        # when input or output or weight is channels-last
        return ["NHWGC", "GKYXC", "NHWGK"]
    else:
        return None


def torch_layout_to_ck_input_layout(torch_layout):
    if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
        return "NGCHW"
    elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-3], 1):
        return "NHWGC"
    else:
        return None


def torch_layout_to_ck_weight_layout(torch_layout):
    if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
        return "GKCYX"
    elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-3], 1):
        return "GKYXC"
    else:
        return None


def torch_layout_to_ck_output_layout(torch_layout):
    if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
        return "NGKHW"
    elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-3], 1):
        return "NHWGK"
    else:
        return None


class CKGroupedConvFwdTemplate(CKTemplate):
    conv_template = r"""
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    PT_EXPORT {{kernel_definition}} {
        auto conv = {{instance_type}} {};
        auto invoker = conv.MakeInvoker();

        using ck::index_t;

        constexpr index_t NumDTensor = {{n_d_tensors}};
        constexpr index_t NDimSpatial = {{n_dim_spatial}};
        constexpr index_t GroupCount = {{group_count}};
        constexpr index_t NBatch = {{batch_size}};
        constexpr index_t NOutChannels = {{n_output_channels}};
        constexpr index_t NInChannels = {{n_input_channels}};
        const std::vector<index_t> FilterSize = { {{filter_size}} };
        const std::vector<index_t> InputSize = { {{input_size}} };
        const std::vector<index_t> ConvolutionStrides = { {{convolution_strides}} };
        const std::vector<index_t> Dilations = { {{dilations}} };
        const std::vector<index_t> LeftPads = { {{left_pads}} };
        const std::vector<index_t> RightPads = { {{right_pads}} };

        auto conv_param = ck::utils::conv::ConvParam {
            NDimSpatial,
            GroupCount,
            NBatch,
            NOutChannels,
            NInChannels,
            FilterSize,
            InputSize,
            ConvolutionStrides,
            Dilations,
            LeftPads,
            RightPads,
        };

        using InLayout  = ck::tensor_layout::convolution::{{input_layout}};
        using WeiLayout = ck::tensor_layout::convolution::{{weight_layout}};
        using OutLayout = ck::tensor_layout::convolution::{{output_layout}};

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

        const void* p_a = input;
        const void* p_b = weight;
        const std::array<const void*, NumDTensor> p_ds;
        void* p_e = output;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides;
        std::array<index_t, NDimSpatial> conv_filter_strides;
        std::array<index_t, NDimSpatial> conv_filter_dilations;
        std::array<index_t, NDimSpatial> input_left_pads;
        std::array<index_t, NDimSpatial> input_right_pads;
        const auto a_element_op = PassThrough {};
        const auto b_element_op = PassThrough {};
        const auto cde_element_op = PassThrough {};

        auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

        copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
        copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
        copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
        copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
        copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
        copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
        copy(conv_param.conv_filter_strides_, conv_filter_strides);
        copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
        copy(conv_param.input_left_pads_, input_left_pads);
        copy(conv_param.input_right_pads_, input_right_pads);

        auto argument = conv.MakeArgument(
            p_a,
            p_b,
            p_ds,
            p_e,
            a_g_n_c_wis_lengths,
            a_g_n_c_wis_strides,
            b_g_k_c_xs_lengths,
            b_g_k_c_xs_strides,
            ds_g_n_k_wos_lengths,
            ds_g_n_k_wos_strides,
            e_g_n_k_wos_lengths,
            e_g_n_k_wos_strides,
            conv_filter_strides,
            conv_filter_dilations,
            input_left_pads,
            input_right_pads,
            a_element_op,
            b_element_op,
            cde_element_op
        );
        if (!conv.IsSupportedArgument(argument)) {
            // we do our best to statically avoid this case in `filter_op`
            std::cerr << "invalid argument for conv instance " << conv.GetTypeString() << std::endl;
            argument.Print();
            return -23;
        }
        if (workspace_size) {
            *workspace_size = conv.GetWorkSpaceSize(&argument);
            return 0;
        }

        if (p_a == nullptr) {
            std::cerr << "p_a is nullptr" << std::endl;
            return -1;
        }
        if (p_b == nullptr) {
            std::cerr << "p_b is nullptr" << std::endl;
            return -1;
        }
        if (p_e == nullptr) {
            std::cerr << "p_e is nullptr" << std::endl;
            return -1;
        }

        // when debugging, do time kernel to serialize launches
        auto stream_config = StreamConfig{stream, /* time kernel */ false, /* log level */ 0};

        if (workspace != nullptr) {
            conv.SetWorkSpacePointer(&argument, workspace, stream_config);
        }

        // run the kernel
        float elapsed_time = invoker.Run(argument, stream_config);
        return 0;
    } // kernel definition
    } // extern C

    #ifdef GENERATE_CK_STANDALONE_RUNNER
    int main(int argc, char** argv) {
        (void) argc;
        (void) argv;
        return 0;
    }
    #endif // GENERATE_CK_STANDALONE_RUNNER
"""

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK conv globals

                using NWC   = ck::tensor_layout::convolution::NWC;
                using NHWC  = ck::tensor_layout::convolution::NHWC;
                using NDHWC = ck::tensor_layout::convolution::NDHWC;

                using KXC   = ck::tensor_layout::convolution::KXC;
                using KYXC  = ck::tensor_layout::convolution::KYXC;
                using KZYXC = ck::tensor_layout::convolution::KZYXC;

                using NWK   = ck::tensor_layout::convolution::NWK;
                using NHWK  = ck::tensor_layout::convolution::NHWK;
                using NDHWK = ck::tensor_layout::convolution::NDHWK;

                using GNWC   = ck::tensor_layout::convolution::GNWC;
                using GNHWC  = ck::tensor_layout::convolution::GNHWC;
                using GNDHWC = ck::tensor_layout::convolution::GNDHWC;

                using GKXC   = ck::tensor_layout::convolution::GKXC;
                using GKYXC  = ck::tensor_layout::convolution::GKYXC;
                using GKZYXC = ck::tensor_layout::convolution::GKZYXC;

                using GKCX   = ck::tensor_layout::convolution::GKCX;
                using GKCYX  = ck::tensor_layout::convolution::GKCYX;
                using GKCZYX = ck::tensor_layout::convolution::GKCZYX;

                using GNWK   = ck::tensor_layout::convolution::GNWK;
                using GNHWK  = ck::tensor_layout::convolution::GNHWK;
                using GNDHWK = ck::tensor_layout::convolution::GNDHWK;

                using NGKW   = ck::tensor_layout::convolution::NGKW;
                using NGKHW  = ck::tensor_layout::convolution::NGKHW;
                using NGKDHW = ck::tensor_layout::convolution::NGKDHW;

                using NWGC   = ck::tensor_layout::convolution::NWGC;
                using NHWGC  = ck::tensor_layout::convolution::NHWGC;
                using NDHWGC = ck::tensor_layout::convolution::NDHWGC;

                using KXGC   = ck::tensor_layout::convolution::KXGC;
                using KYXGC  = ck::tensor_layout::convolution::KYXGC;
                using KZYXGC = ck::tensor_layout::convolution::KZYXGC;

                using NWGK   = ck::tensor_layout::convolution::NWGK;
                using NHWGK  = ck::tensor_layout::convolution::NHWGK;
                using NDHWGK = ck::tensor_layout::convolution::NDHWGK;

                using NGCW   = ck::tensor_layout::convolution::NGCW;
                using NGCHW  = ck::tensor_layout::convolution::NGCHW;
                using NGCDHW = ck::tensor_layout::convolution::NGCDHW;

                using G_K    = ck::tensor_layout::convolution::G_K;

                using BlockGemmPipelineScheduler = ck::BlockGemmPipelineScheduler;
                using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;
                using BlockGemmPipelineVersion = ck::BlockGemmPipelineVersion;

                using ConvolutionForwardSpecialization = ck::tensor_operation::device::ConvolutionForwardSpecialization;

                namespace ck {
                namespace utils {
                namespace conv {

                ConvParam::ConvParam(ck::index_t n_dim,
                                    ck::index_t group_count,
                                    ck::index_t n_batch,
                                    ck::index_t n_out_channels,
                                    ck::index_t n_in_channels,
                                    const std::vector<ck::index_t>& filters_len,
                                    const std::vector<ck::index_t>& input_len,
                                    const std::vector<ck::index_t>& strides,
                                    const std::vector<ck::index_t>& dilations,
                                    const std::vector<ck::index_t>& left_pads,
                                    const std::vector<ck::index_t>& right_pads)
                    : num_dim_spatial_(static_cast<ck::long_index_t>(n_dim)),
                    G_(static_cast<ck::long_index_t>(group_count)),
                    N_(static_cast<ck::long_index_t>(n_batch)),
                    K_(static_cast<ck::long_index_t>(n_out_channels)),
                    C_(static_cast<ck::long_index_t>(n_in_channels)),
                    filter_spatial_lengths_(num_dim_spatial_),
                    input_spatial_lengths_(num_dim_spatial_),
                    output_spatial_lengths_(num_dim_spatial_),
                    conv_filter_strides_(num_dim_spatial_),
                    conv_filter_dilations_(num_dim_spatial_),
                    input_left_pads_(num_dim_spatial_),
                    input_right_pads_(num_dim_spatial_)
                {
                    if(static_cast<ck::index_t>(filter_spatial_lengths_.size()) != num_dim_spatial_ ||
                    static_cast<ck::index_t>(input_spatial_lengths_.size()) != num_dim_spatial_ ||
                    static_cast<ck::index_t>(conv_filter_strides_.size()) != num_dim_spatial_ ||
                    static_cast<ck::index_t>(conv_filter_dilations_.size()) != num_dim_spatial_ ||
                    static_cast<ck::index_t>(input_left_pads_.size()) != num_dim_spatial_ ||
                    static_cast<ck::index_t>(input_right_pads_.size()) != num_dim_spatial_)
                    {
                        throw(
                            std::runtime_error("ConvParam::ConvParam: "
                                            "parameter size is different from number of declared dimensions!"));
                    }

                    for(ck::index_t i = 0; i < num_dim_spatial_; ++i)
                    {
                        filter_spatial_lengths_[i] = static_cast<ck::long_index_t>(filters_len[i]);
                        input_spatial_lengths_[i]  = static_cast<ck::long_index_t>(input_len[i]);
                        conv_filter_strides_[i]    = static_cast<ck::long_index_t>(strides[i]);
                        conv_filter_dilations_[i]  = static_cast<ck::long_index_t>(dilations[i]);
                        input_left_pads_[i]        = static_cast<ck::long_index_t>(left_pads[i]);
                        input_right_pads_[i]       = static_cast<ck::long_index_t>(right_pads[i]);

                        // XEff = (X - 1) * conv_dilation_w + 1;
                        // Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
                        const ck::long_index_t x_eff =
                            (filter_spatial_lengths_[i] - 1) * conv_filter_dilations_[i] + 1;

                        output_spatial_lengths_[i] =
                            (input_spatial_lengths_[i] + input_left_pads_[i] + input_right_pads_[i] - x_eff) /
                                conv_filter_strides_[i] +
                            1;
                    }
                }

                } // namespace conv
                } // namespace utils
                } // namespace ck

                const std::vector<std::size_t>& HostTensorDescriptor::GetLengths() const { return mLens; }
                const std::vector<std::size_t>& HostTensorDescriptor::GetStrides() const { return mStrides; }
                std::size_t HostTensorDescriptor::GetNumOfDimension() const { return mLens.size(); }
                void HostTensorDescriptor::CalculateStrides() {
                    mStrides.clear();
                    mStrides.resize(mLens.size(), 0);
                    if(mStrides.empty())
                        return;

                    mStrides.back() = 1;
                    std::partial_sum(
                        mLens.rbegin(), mLens.rend() - 1, mStrides.rbegin() + 1, std::multiplies<std::size_t>());
                }
            """
        )
        return res

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK conv headers

                #include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp"
                #include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
                #include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

                #include "ck/library/utility/convolution_parameter.hpp"
                #include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
            """
        )
        return res

    @staticmethod
    def add_ck_conv_choices(
        choices,
        layout,
        input_nodes,
        *,
        stride,
        padding,
        dilation,
        groups,
        n_spatial_dimensions,
    ):
        template = CKGroupedConvFwdTemplate(
            input_nodes,
            layout,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            n_spatial_dimensions=n_spatial_dimensions,
        )
        ops = template.gen_ops()
        for op in ops:
            template.maybe_append_choice(
                choices,
                op=op,
            )

    def __init__(
        self,
        input_nodes,
        layout,
        *,
        stride,
        padding,
        dilation,
        groups,
        n_spatial_dimensions,
    ):
        super().__init__(
            "ck_conv_template",
            input_nodes,
            layout,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.n_spatial_dimensions = n_spatial_dimensions

    def filter_op(self, op: "CKGroupedConvFwdOp"):  # type: ignore[name-defined]
        metas = [
            T.get_layout()
            for T in [*self.input_nodes, self.output_node]
            if T is not None
        ]
        X_meta = metas[0]
        W_meta = metas[1]
        Y_meta = metas[-1]
        # disable the instance if dtypes don't match
        if op.a_element_dtype != self._TORCH_DTYPE_TO_CK[X_meta.dtype]:
            return None
        if op.b_element_dtype != self._TORCH_DTYPE_TO_CK[W_meta.dtype]:
            return None
        if op.e_element_dtype != self._TORCH_DTYPE_TO_CK[Y_meta.dtype]:
            return None
        # disable the instance if layouts don't match
        if op.a_layout != torch_layout_to_ck_input_layout(X_meta):
            return None
        if op.b_layout != torch_layout_to_ck_weight_layout(W_meta):
            return None
        if op.e_layout != torch_layout_to_ck_output_layout(Y_meta):
            return None
        # disable the instance if number of spatial dimensions doesn't match
        if op.n_dim_spatial != self.n_spatial_dimensions:
            return None
        # disable 1x1 and odd-channels conv specializations for now
        if "Default" not in op.conv_forward_specialization:
            return None
        return op

    def gen_ops(self):
        unfiltered_instances = gen_conv_ops_library()

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

    def emit_ck_instance(self, op: "CKGroupedConvFwdOp") -> tuple[str, str]:  # type: ignore[name-defined]
        # The Jinja template for generating a C++ type alias *definition* for a Universal GEMM instance
        template_definition = r"""
    // Gemm operator {{operation_name}}
    using Operation_{{operation_name}} =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
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

    def render(self, kernel: ROCmTemplateKernel, op: "CKGroupedConvFwdOp", **kwargs) -> str:  # type: ignore[override, name-defined]
        template_buffer_node = kwargs.get("template_buffer_node", None)
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = self.input_nodes[2] if 3 == len(self.input_nodes) else None

        op = copy.deepcopy(op)

        instance_definition, instance_type = self.emit_ck_instance(op)

        return self._template_from_string(self.conv_template).render(
            headers=self.header().getvalue(),
            globals=self.globals().getvalue(),
            instance_definition=instance_definition,
            instance_type=instance_type,
            kernel_definition=kernel.def_kernel(
                inputs=[X, W, Bias] if Bias is not None else [X, W],
                outputs=[Y],
                names_str="input, weight, bias, output"
                if Bias is not None
                else "input, weight, output",
                size_args=[],
            ),
            n_d_tensors=1 if Bias is not None else 0,
            n_dim_spatial=self.n_spatial_dimensions,
            group_count=self.groups,
            batch_size=X.shape[0],  # type: ignore[index]
            n_output_channels=Y.shape[1],  # type: ignore[index]
            n_input_channels=X.shape[1],  # type: ignore[index]
            filter_size=", ".join(map(str, W.shape[2:])),  # type: ignore[index]
            input_size=", ".join(map(str, X.shape[2:])),  # type: ignore[index]
            convolution_strides=", ".join(map(str, self.stride)),
            dilations=", ".join(map(str, self.dilation)),
            left_pads=", ".join(map(str, self.padding)),
            right_pads=", ".join(map(str, self.padding)),
            input_layout=op.a_layout,
            weight_layout=op.b_layout,
            output_layout=op.e_layout,
        )

    def size_args(self):
        return []
