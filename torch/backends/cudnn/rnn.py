import torch.cuda
import torch.backends.cudnn as cudnn
from torch.backends.cudnn import check_error
import ctypes


def initDropoutDescriptor(fn, handle):
    dropout_desc = cudnn.DropoutDescriptor()

    dropout_states_size = ctypes.c_long()
    check_error(cudnn.lib.cudnnDropoutGetStatesSize(
        handle,
        ctypes.byref(dropout_states_size)))

    dropout_states = torch.cuda.ByteTensor(dropout_states_size.value)
    dropout_desc.set(
        handle,
        fn.dropout,
        dropout_states,
        fn.seed
    )
    return dropout_desc


def initRNNDescriptor(fn):
    rnn_desc = cudnn.RNNDescriptor()

    rnn_desc.set(
        fn.hidden_size,
        fn.num_layers,
        fn.dropout_desc,
        fn.input_mode,
        fn.bidirectional,
        fn.mode,
        fn.datatype
    )
    return rnn_desc


def initWeightDescriptor(fn, weight):
    w_desc = cudnn.FilterDescriptor()
    w_view = weight.view(-1, 1, 1)  # seems that filters require >=3 dimensions
    w_desc.set(w_view)
    return w_desc


def _inputSize(fn):
    return (fn.seq_length, fn.mini_batch, fn.input_size)


def _hiddenSize(fn):
    return (fn.num_layers * fn.num_directions, fn.mini_batch, fn.hidden_size)


def _outputSize(fn):
    return (fn.seq_length, fn.mini_batch, fn.hidden_size * fn.num_directions)


def getNumWeights(handle, rnn_desc, x_desc, datatype):
    weight_size = ctypes.c_long()
    check_error(cudnn.lib.cudnnGetRNNParamsSize(
        handle,
        rnn_desc,
        x_desc,
        ctypes.byref(weight_size),
        datatype
    ))
    elem_size = cudnn._sizeofmap[datatype]
    assert(weight_size.value % elem_size == 0)
    return weight_size.value // elem_size


def getParameters(fn, handle, weight_buf):

    cudnn_methods = [
        cudnn.lib.cudnnGetRNNLinLayerMatrixParams,
        cudnn.lib.cudnnGetRNNLinLayerBiasParams
    ]

    # if fn.mode == cudnn.CUDNN_RNN_RELU or fn.mode == cudnn.CUDNN_RNN_TANH:
    #     linear_name = ["ih", "hh"]
    # elif fn.mode == cudnn.CUDNN_LSTM:
    #     linear_name = ["ii", "if", "ic", "io", "hi", "hf", "hc", "ho"]
    # elif fn.mode == cudnn.CUDNN_GRU:
    #     linear_name = ["ir", "iu", "ic", "hr", "hu", "hc"]
    # else:
    #     raise Exception("Unknown mode: {}".format(fn.mode))

    params = []
    num_linear_layers = _numLinearLayers(fn)
    num_layers = fn.num_directions * fn.num_layers
    for layer in range(num_layers):
        layer_params = []
        for cudnn_method in cudnn_methods:
            for linear_id in range(num_linear_layers):
                lin_layer_mat_desc = cudnn.FilterDescriptor()
                matrix_pointer = ctypes.c_void_p()
                check_error(cudnn_method(
                    handle,
                    fn.rnn_desc,
                    layer,
                    fn.x_descs[0],
                    fn.w_desc,
                    ctypes.c_void_p(weight_buf.data_ptr()),
                    linear_id,
                    lin_layer_mat_desc,
                    ctypes.byref(matrix_pointer)))

                data_type = ctypes.c_int()
                format = ctypes.c_int()
                nb_dims = ctypes.c_int()
                min_dim = 3
                filter_dim_a = torch.IntTensor(min_dim)
                check_error(cudnn.lib.cudnnGetFilterNdDescriptor(
                    lin_layer_mat_desc,
                    min_dim,
                    ctypes.byref(data_type),
                    ctypes.byref(format),
                    ctypes.byref(nb_dims),
                    ctypes.c_void_p(filter_dim_a.data_ptr())))

                filter_dim_a.resize_(nb_dims.value)
                elem_size = cudnn._sizeofmap[fn.datatype]
                offset_bytes = (matrix_pointer.value - weight_buf.data_ptr())
                assert(offset_bytes % elem_size == 0)
                offset = offset_bytes // elem_size

                # for all the RNN types provided by CUDNN, all the ih weights
                # are the same size and are allocated in a contiguous chunk
                # (same for the hh weights, and the ih and hh biases).
                # Since we're storing all the weights in a single tensor anyway,
                # might as well merge the CUDNN ones into a single tensor as well
                if linear_id == 0 or linear_id == num_linear_layers / 2:
                    assert(filter_dim_a.prod() == filter_dim_a[0])
                    param = fn.weight_buf.new().set_(
                        weight_buf.storage(), offset,
                        filter_dim_a[0] * num_linear_layers / 2, filter_dim_a[2])
                    layer_params.append(param)
                else:
                    assert(cur_offset == offset)

                cur_offset = offset + filter_dim_a[0]


        params.append(layer_params)

    return params


def _copyParams(params_from, params_to):
    for layer_params_from, layer_params_to in zip(params_from, params_to):
        for param_from, param_to in zip(layer_params_from, layer_params_to):
            assert(param_from.type() == param_to.type())
            param_to.copy_(param_from)


def forward(fn, input, hx, weight, output, hy):
    with torch.cuda.device_of(input):
        lib = cudnn.lib
        handle = cudnn.get_handle()
        fn.datatype = cudnn._typemap[input.type()]

        if fn.mode == cudnn.CUDNN_LSTM:
            hx, cx = hx
            hy, cy = hy
        else:
            cx, cy = None, None

        if fn.batch_first:
            input = input.transpose(0, 1)

        if input.dim() != 3:
            raise Exception(
                'input must have 3 dimensions: seq_length, mini_batch, input_size')
        if fn.input_size != input.size(2):
            raise Exception('input.size(2) must be equal to input_size provided')
        if fn.dropout != 0 and cudnn.lib.version < 5103:
            raise Exception('dropout supported only in cudnn v5.1 and above')

        fn.seq_length, fn.mini_batch, fn.input_size = input.size()
        hidden_size = _hiddenSize(fn)
        output_size = _outputSize(fn)
        x = input.contiguous()
        output.resize_(*output_size)
        hy.resize_(*hidden_size).zero_()
        if cy:
            cy.resize_(*hidden_size).zero_()
        y = output

        # init descriptors
        fn.dropout_desc = initDropoutDescriptor(fn, handle)
        fn.rnn_desc     = initRNNDescriptor(fn)
        fn.x_descs      = cudnn.descriptor(x[0], fn.seq_length)
        fn.y_descs      = cudnn.descriptor(y[0], fn.seq_length)
        fn.hx_desc      = cudnn.descriptor(hx)
        fn.hy_desc      = cudnn.descriptor(hx)
        fn.cx_desc      = cudnn.descriptor(cx) if cx else None
        fn.cy_desc      = cudnn.descriptor(cx) if cx else None

        # create the weight buffer and copy the weights into it
        num_weights = getNumWeights(
            handle, fn.rnn_desc, fn.x_descs[0], fn.datatype)
        fn.weight_buf = input.new(num_weights)
        fn.w_desc       = initWeightDescriptor(fn, fn.weight_buf)
        w = fn.weight_buf
        # this zero might not seem necessary, but it is in the case
        # where biases are disabled; then they won't be copied and must be zero'd.
        # Alternatively, _copyParams could be written more carefully.
        w.zero_()
        params = getParameters(fn, handle, w)
        _copyParams(weight, params)

        if tuple(hx.size()) != hidden_size:
            raise Exception('Expected hx size {}, got {}'.format(
               hidden_size, tuple(hx.size())))
        if cx and tuple(cx.size()) != hidden_size:
            raise Exception('Expected cx size {}, got {}'.format(
                hidden_size, tuple(cx.size())))

        workspace_size = ctypes.c_long()
        check_error(lib.cudnnGetRNNWorkspaceSize(
            handle,
            fn.rnn_desc,
            fn.seq_length,
            fn.x_descs,
            ctypes.byref(workspace_size)
        ))
        fn.workspace = torch.cuda.ByteTensor(workspace_size.value)
        if fn.train:
            reserve_size = ctypes.c_long()
            check_error(lib.cudnnGetRNNTrainingReserveSize(
                handle,
                fn.rnn_desc,
                fn.seq_length,
                fn.x_descs,
                ctypes.byref(reserve_size)
            ))
            fn.reserve = torch.cuda.ByteTensor(reserve_size.value)

            check_error(lib.cudnnRNNForwardTraining(
                handle,
                fn.rnn_desc,
                fn.seq_length,
                fn.x_descs, ctypes.c_void_p(x.data_ptr()),
                fn.hx_desc, ctypes.c_void_p(hx.data_ptr()),
                fn.cx_desc, ctypes.c_void_p(cx.data_ptr()) if cx else None,
                fn.w_desc, ctypes.c_void_p(w.data_ptr()),
                fn.y_descs, ctypes.c_void_p(y.data_ptr()),
                fn.hy_desc, ctypes.c_void_p(hy.data_ptr()),
                fn.cy_desc, ctypes.c_void_p(cy.data_ptr()) if cx else None,
                ctypes.c_void_p(fn.workspace.data_ptr()), fn.workspace.size(0),
                ctypes.c_void_p(fn.reserve.data_ptr()), fn.reserve.size(0)
            ))
        else:  # inference
            check_error(lib.cudnnRNNForwardInference(
                handle,
                fn.rnn_desc,
                fn.seq_length,
                fn.x_descs, ctypes.c_void_p(x.data_ptr()),
                fn.hx_desc, ctypes.c_void_p(hx.data_ptr()),
                fn.cx_desc, ctypes.c_void_p(cx.data_ptr()) if cx else None,
                fn.w_desc, ctypes.c_void_p(w.data_ptr()),
                fn.y_descs, ctypes.c_void_p(y.data_ptr()),
                fn.hy_desc, ctypes.c_void_p(hy.data_ptr()),
                fn.cy_desc, ctypes.c_void_p(cy.data_ptr()) if cx else None,
                ctypes.c_void_p(fn.reserve.data_ptr()), fn.reserve.size(0)
            ))

        if fn.batch_first:
            output = output.transpose(0, 1)


def backward_grad(fn, input, hx, weight, output, grad_output, grad_hy, grad_input, grad_hx):
    with torch.cuda.device_of(input):
        handle = cudnn.get_handle()

        if fn.mode == cudnn.CUDNN_LSTM:
            hx, cx = hx
            grad_hx, grad_cx = grad_hx
            grad_hy, grad_cy = grad_hy
        else:
            cx, grad_cx, grad_cy = None, None, None

        if fn.batch_first:
            input = input.transpose(0, 1)
            grad_output = grad_output.transpose(0, 1)
            output = output.transpose(0, 1)

        input_size = _inputSize(fn)
        hidden_size = _hiddenSize(fn)
        output_size = _outputSize(fn)

        x = input.contiguous()
        dy = grad_output.contiguous()
        y = output
        w = fn.weight_buf
        dx = grad_input.resize_as_(input)
        dhy = grad_hy.resize_(*hidden_size)
        dcy = grad_cy.resize_(*hidden_size) if grad_cy else None
        dhx = grad_hx.resize_(*hidden_size)
        dcx = grad_cx.resize_(*hidden_size) if grad_cx else None

        if fn.dropout != 0 and lib.version < 5103:
            raise Exception('dropout supported only in cudnn v 5.1 and above')
        if not fn.train:
            raise Exception('backward_grad can only be called when training!')
        if tuple(input.size()) != input_size:
            raise Exception('Expected input size {}, got {}'.format(
                input_size, tuple(input.size())))
        if tuple(output.size()) != _outputSize(fn):
            raise Exception('Expected output size {}, got {}'.format(
                output_size, output.size()))
        if hx and tuple(hx.size()) != hidden_size:
            raise Exception('Expected hx size {}, got {}'.format(
                hidden_size, hx.size()))
        if cx and tuple(cx.size()) != hidden_size:
            raise Exception('Expected cx size {}, got {}'.format(
                hidden_size, cx.size()))
        if dhy and tuple(dhy.size()) != hidden_size:
            raise Exception('Expected dhy size {}, got {}'.format(
                hidden_size, dhy.size()))
        if dcy and tuple(dcy.size()) != hidden_size:
            raise Exception('Expected dcy size {}, got {}'.format(
                hidden_size, dcy.size()))

        check_error(cudnn.lib.cudnnRNNBackwardData(
            handle,
            fn.rnn_desc,
            fn.seq_length,
            fn.y_descs, ctypes.c_void_p(y.data_ptr()),
            fn.y_descs, ctypes.c_void_p(dy.data_ptr()),
            fn.hy_desc, ctypes.c_void_p(dhy.data_ptr()),
            fn.cy_desc, ctypes.c_void_p(dcy.data_ptr()) if cx else None,
            fn.w_desc, ctypes.c_void_p(w.data_ptr()),
            fn.hx_desc, ctypes.c_void_p(hx.data_ptr()),
            fn.cx_desc, ctypes.c_void_p(cx.data_ptr()) if cx else None,
            fn.x_descs, ctypes.c_void_p(dx.data_ptr()),
            fn.hx_desc, ctypes.c_void_p(dhx.data_ptr()),
            fn.cx_desc, ctypes.c_void_p(dcx.data_ptr()) if cx else None,
            ctypes.c_void_p(fn.workspace.data_ptr()), fn.workspace.size(0),
            ctypes.c_void_p(fn.reserve.data_ptr()), fn.reserve.size(0)
        ))

        if fn.batch_first:
            grad_input = grad_input.transpose(0, 1)


def _numLinearLayers(fn):
    if fn.mode == cudnn.CUDNN_LSTM:
        return 8
    elif fn.mode == cudnn.CUDNN_GRU:
        return 6
    elif fn.mode == cudnn.CUDNN_RNN_RELU:
        return 2
    elif fn.mode == cudnn.CUDNN_RNN_TANH:
        return 2
    else:
        raise Exception('Unknown mode: {}'.format(fn.mode))


def backward_weight(fn, input, hx, output, weight, grad_weight):
    with torch.cuda.device_of(input):
        handle = cudnn.get_handle()

        if fn.mode == cudnn.CUDNN_LSTM:
            hx, cx = hx
        else:
            cx = None

        if fn.batch_first:
            input = input.transpose(1, 2)
            output = output.transpose(1, 2)

        input_size = _inputSize(fn)
        hidden_size = _hiddenSize(fn)
        if not fn.train:
            raise Exception('backward_weight can only be called when training!')
        if fn.dropout != 0 and lib.version < 5103:
            raise Exception('dropout supported only in cudnn v 5.1 and above')
        if tuple(input.size()) != input_size:
            raise Exception('Expected input size {}, got {}'.format(
                input_size, tuple(input.size())))
        if not fn.train:
            raise Exception('backward_weight can only be called when training!')
        if tuple(hx.size()) != hidden_size:
            raise Exception('Expected input size {}, got {}'.format(
                hidden_size, hx.size()))

        x = input.contiguous()
        y = output
        dw = fn.weight_buf.new().resize_as_(fn.weight_buf).zero_()

        check_error(cudnn.lib.cudnnRNNBackwardWeights(
            handle,
            fn.rnn_desc,
            fn.seq_length,
            fn.x_descs, ctypes.c_void_p(x.data_ptr()),
            fn.hx_desc, ctypes.c_void_p(hx.data_ptr()),
            fn.y_descs, ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(fn.workspace.data_ptr()), fn.workspace.size(0),
            fn.w_desc, ctypes.c_void_p(dw.data_ptr()),
            ctypes.c_void_p(fn.reserve.data_ptr()), fn.reserve.size(0)
        ))

        # copy the weights from the weight_buf into grad_weight
        grad_params = getParameters(fn, handle, dw)
        _copyParams(grad_params, grad_weight)
        return grad_weight
