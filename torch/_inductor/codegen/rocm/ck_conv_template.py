
from torch._inductor.codegen.rocm.ck_template import CKTemplate


class CKConvTemplate(CKTemplate):
    conv_template = r"""
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    PT_EXPORT {{kernel_definition}} {
        auto conv = {{instance_type}} {};
        auto invoker = conv.MakeInvoker();

        using ck::index_t;

        constexpr index_t NumDTensor = {{NumDTensor}};
        constexpr index_t NDimSpatial = {{NDimSpatial}};

        const void* p_a;
        const void* p_b;
        const std::array<const void*, NumDTensor> p_ds;
        void* p_e;
        const std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths;
        const std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides;
        const std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths;
        const std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides;
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths;
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides;
        const std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths;
        const std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides;
        const std::array<index_t, NDimSpatial> conv_filter_strides;
        const std::array<index_t, NDimSpatial> conv_filter_dilations;
        const std::array<index_t, NDimSpatial> input_left_pads;
        const std::array<index_t, NDimSpatial> input_right_pads;
        const AElementwiseOperation a_element_op = PassThrough;
        const BElementwiseOperation b_element_op = PassThrough;
        const CDEElementwiseOperation cde_element_op = PassThrough;

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
        // run the kernel
        float elapsed_time = invoker.Run(argument, StreamConfig{stream, /* time kernel */ false, /* log level */ kDEBUG_LOG});
        return 0;
    } // kernel definition
    } // extern C
"""
