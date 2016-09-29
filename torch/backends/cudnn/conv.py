import torch.cuda
import torch.backends.cudnn as cudnn
import ctypes

def forward(fn, input, weight, bias, output):
    with torch.cuda.device_of(input):
        handle = cudnn.get_handle()
        out_channels, in_channels = weight.size(0), weight.size(1)

        inslice = input.narrow(1, 0, in_channels // fn.groups)
        outslice = output.narrow(1, 0, out_channels // fn.groups)
        weight_slice = (
            weight.narrow(0, 0, out_channels // fn.groups)
            .narrow(1, 0, in_channels // fn.groups)
        )

        fn.input_offset = inslice[0].numel() * input.element_size()
        fn.output_offset = outslice[0].numel() * output.element_size()
        fn.weight_offset = weight_slice.numel() * weight.element_size()

        fn.idesc = cudnn.descriptor(inslice)
        fn.odesc = cudnn.descriptor(outslice)
        fn.odesc_bias = cudnn.descriptor(output)

        fn.wdesc = cudnn.FilterDescriptor()
        fn.wdesc.set(weight_slice)

        fn.conv_desc = cudnn.ConvolutionDescriptor()
        fn.conv_desc.set(weight.type(), fn.pad, fn.stride)

        fwd_alg = cudnn.convolution_forward_algorithm(
            fn.idesc, fn.wdesc, fn.conv_desc, fn.odesc)

        workspace_size = ctypes.c_size_t()
        cudnn.convolution_forward_workspace_size(
            cudnn.get_handle(), fn.idesc, fn.wdesc, fn.conv_desc,
            fn.odesc, fwd_alg, ctypes.byref(workspace_size))

        workspace = torch.cuda.ByteStorage(workspace_size.value)

        alpha = cudnn.c_type(input)(1)
        beta = cudnn.c_type(output)(0)
        for g in range(fn.groups):
            input_ptr = ctypes.c_void_p(input.data_ptr() + g * fn.input_offset)
            weight_ptr = ctypes.c_void_p(weight.data_ptr() + g * fn.weight_offset)
            output_ptr = ctypes.c_void_p(output.data_ptr() + g * fn.output_offset)
            workspace_ptr = ctypes.c_void_p(workspace.data_ptr())

            cudnn.convolution_forward(
                handle, ctypes.byref(alpha), fn.idesc, input_ptr, fn.wdesc,
                weight_ptr, fn.conv_desc, fwd_alg, workspace_ptr,
                workspace_size, ctypes.byref(beta), fn.odesc, output_ptr)

        if bias is not None:
            alpha = cudnn.c_type(input)(1)
            beta = cudnn.c_type(output)(1)

            fn.bias_desc = cudnn.descriptor(bias.view(1, bias.size(0), 1, 1))
            cudnn.add_tensor(
                handle, ctypes.byref(alpha), fn.bias_desc,
                ctypes.c_void_p(bias.data_ptr()), ctypes.byref(beta),
                fn.odesc_bias, ctypes.c_void_p(output.data_ptr()))

        return output

def backward_data(fn, grad_output, input, weight):
    with torch.cuda.device_of(input):
        handle = cudnn.get_handle()
        grad_input = input.new().resize_as_(input)

        bwd_data_alg = cudnn.convolution_backward_data_algorithm(
            fn.wdesc, fn.odesc, fn.conv_desc, fn.idesc)

        workspace_size = ctypes.c_size_t()
        cudnn.convolution_backward_data_workspace_size(
            handle, fn.wdesc, fn.odesc, fn.conv_desc, fn.idesc,
            bwd_data_alg, ctypes.byref(workspace_size))

        workspace = torch.cuda.ByteStorage(workspace_size.value)

        alpha = cudnn.c_type(input)(1)
        beta = cudnn.c_type(input)(0)
        for g in range(fn.groups):
            cudnn.convolution_backward_data(
                handle, ctypes.byref(alpha), fn.wdesc,
                ctypes.c_void_p(weight.data_ptr() + g * fn.weight_offset),
                fn.odesc,
                ctypes.c_void_p(grad_output.data_ptr() + g * fn.output_offset),
                fn.conv_desc, bwd_data_alg, ctypes.c_void_p(workspace.data_ptr()),
                workspace_size, ctypes.byref(beta), fn.idesc,
                ctypes.c_void_p(grad_input.data_ptr() + g * fn.input_offset))

        return grad_input

def backward_filter(fn, grad_output, input, weight):
    with torch.cuda.device_of(input):
        handle = cudnn.get_handle()
        grad_weight = weight.new().resize_as_(weight)

        bwd_filter_alg = cudnn.convolution_backward_filter_algorithm(
            fn.idesc, fn.odesc, fn.conv_desc, fn.wdesc)

        workspace_size = ctypes.c_size_t()
        cudnn.convolution_backward_filter_workspace_size(
            handle, fn.idesc, fn.odesc, fn.conv_desc, fn.wdesc,
            bwd_filter_alg, ctypes.byref(workspace_size))

        workspace = torch.cuda.ByteStorage(workspace_size.value)

        alpha = cudnn.c_type(input)(1)
        beta = cudnn.c_type(input)(0)
        for g in range(fn.groups):
            cudnn.convolution_backward_filter(
                handle, ctypes.byref(alpha), fn.idesc,
                ctypes.c_void_p(input.data_ptr() + g * fn.input_offset),
                fn.odesc,
                ctypes.c_void_p(grad_output.data_ptr() + g * fn.output_offset),
                fn.conv_desc, bwd_filter_alg,
                ctypes.c_void_p(workspace.data_ptr()), workspace_size,
                ctypes.byref(beta), fn.wdesc,
                ctypes.c_void_p(grad_weight.data_ptr() + g * fn.weight_offset))

        return grad_weight

def backward_bias(fn, grad_output, bias):
    with torch.cuda.device_of(grad_output):
        grad_bias = bias.new().resize_as_(bias)
        alpha = cudnn.c_type(grad_output)(1)
        beta = cudnn.c_type(grad_output)(0)

        cudnn.convolution_backward_bias(
            cudnn.get_handle(), ctypes.byref(alpha), fn.odesc_bias,
            ctypes.c_void_p(grad_output.data_ptr()), ctypes.byref(beta),
            fn.bias_desc, ctypes.c_void_p(grad_bias.data_ptr()))
        return grad_bias
