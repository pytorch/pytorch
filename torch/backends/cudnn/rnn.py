import torch.cuda
import torch.backends.cudnn as cudnn
from torch.backends.cudnn import check_error
import ctypes


def initDropoutDescriptor(fn, handle):
    fn.dropout_desc = cudnn.DropoutDescriptor()

    dropout_states_size = ctypes.c_long()
    check_error(cudnn.lib.cudnnDropoutGetStatesSize(
        handle,
        ctypes.byref(dropout_states_size)))

    fn.dropout_states = torch.cuda.ByteTensor(dropout_states_size.value)

    fn.dropout_desc.set(
        handle,
        fn.dropout,
        fn.dropout_states,
        fn.seed
    )


def initRNNDescriptor(fn):
    fn.rnn_desc = cudnn.RNNDescriptor()

    fn.rnn_desc.set(
        fn.hidden_size,
        fn.num_layers,
        fn.dropout_desc,
        fn.input_mode,
        fn.bidirectional,
        fn.mode,
        fn.datatype
    )


def initWeightDescriptor(fn, weight):
    fn.w_desc = cudnn.FilterDescriptor()
    w_view = weight.view(-1, 1, 1)  # seems that filters require >=3 dimensions
    fn.w_desc.set(w_view)


def initIODescriptor(fn, input, output):
    fn.x_descs = cudnn.descriptor(input[0], fn.seq_length)
    fn.y_descs = cudnn.descriptor(output[0], fn.seq_length)


def initHiddenDescriptors(fn, hx):
    fn.hx_desc = cudnn.descriptor(hx)
    fn.hy_desc = cudnn.descriptor(hx)


def initCellDescriptors(fn, cx):
    if cx:
        fn.cx_desc = cudnn.descriptor(cx)
        fn.cy_desc = cudnn.descriptor(cx)
    else:
        fn.cx_desc = fn.cy_desc = None

def _inputSize(fn):
    return (fn.seq_length, fn.mini_batch, fn.input_size)


def _hiddenSize(fn):
    return (fn.num_layers * fn.num_directions, fn.mini_batch, fn.hidden_size)


def _outputSize(fn):
    return (fn.seq_length, fn.mini_batch, fn.hidden_size * fn.num_directions)


def getNumWeights(fn, handle):
    weight_size = ctypes.c_long()
    check_error(cudnn.lib.cudnnGetRNNParamsSize(
        handle,
        fn.rnn_desc,
        fn.x_descs[0],
        ctypes.byref(weight_size),
        fn.datatype
    ))
    elem_size = cudnn._sizeofmap[fn.datatype]
    assert(weight_size.value % elem_size == 0)
    return weight_size.value // elem_size


def _parametersHelper(fn, handle, weight, cudnn_method):
    linear_params = []
    num_linear_layers = _numLinearLayers(fn)
    num_layers = fn.num_directions * fn.num_layers
    for layer in range(num_layers):
        layer_info = []
        for layer_id in range(num_linear_layers):
            lin_layer_mat_desc = cudnn.FilterDescriptor()
            matrix_pointer = ctypes.c_void_p()
            check_error(cudnn_method(
                handle,
                fn.rnn_desc,
                layer,
                fn.x_descs[0],
                fn.w_desc,
                ctypes.c_void_p(weight.data_ptr()),
                layer_id,
                lin_layer_mat_desc,
                ctypes.byref(matrix_pointer)))

            data_type = ctypes.c_int()
            format = ctypes.c_int()
            nb_dims = ctypes.c_int()
            min_dim = 3
            filter_dim_a = torch.IntStorage(min_dim)
            check_error(cudnn.lib.cudnnGetFilterNdDescriptor(
                lin_layer_mat_desc,
                min_dim,
                ctypes.byref(data_type),
                ctypes.byref(format),
                ctypes.byref(nb_dims),
                ctypes.c_void_p(filter_dim_a.data_ptr())))

            filter_dim_a.resize_(nb_dims.value)
            elem_size = cudnn._sizeofmap[fn.datatype]
            offset_bytes = (matrix_pointer.value - weight.data_ptr())
            assert(offset_bytes % elem_size == 0)
            offset = offset_bytes // elem_size
            params = weight.new().set_(weight.storage(), offset, filter_dim_a.long())
            layer_info.append(params)

        linear_params.append(layer_info)

    return linear_params

def parameters(fn, handle, weight):
    parameters = {}
    parameters['weight'] = _parametersHelper(
        fn, handle, weight, cudnn.lib.cudnnGetRNNLinLayerMatrixParams)
    parameters['bias'] = _parametersHelper(
        fn, handle, weight, cudnn.lib.cudnnGetRNNLinLayerBiasParams)
    return parameters


def forward(fn, input, hx, cx, weight, output, hy, cy):
    with torch.cuda.device_of(input):
        lib = cudnn.lib

        handle = cudnn.get_handle()

        fn.datatype = cudnn._typemap[input.type()]

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
        w = weight

        initDropoutDescriptor(fn, handle)
        initRNNDescriptor(fn)
        initIODescriptor(fn, x, y)
        initHiddenDescriptors(fn, hx)
        initCellDescriptors(fn, cx)
        initWeightDescriptor(fn, weight)

        params = parameters(fn, handle, weight)
        for ptype in params:
            for l, layer in enumerate(params[ptype]):
                for ll, param in enumerate(layer):
                    print(ptype, l, ll, param.storage_offset(), tuple(param.size()))

        if tuple(hx.size()) != hidden_size:
            raise Exception('Expected hx size {}, got {}'.format(
                tuple(hidden_size, hx.size())))
        if cx and tuple(cx.size()) != hidden_size:
            raise Exception('Expected cx size {}, got {}'.format(
                tuple(hidden_size, cx.size())))
        if weight.nelement() != getNumWeights(fn, handle):
            raise Exception("Expected #weights {}, got {}".format(
                getNumWeights(fn, handle), weight.nelement()))
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


def backward_grad(fn, input, hx, cx, weight, output, grad_output, grad_hy, grad_cy, grad_input, grad_hx, grad_cx):
    with torch.cuda.device_of(input):
        handle = cudnn.get_handle()
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
        w = weight
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

        # return grad_input, grad_hy, grad_cy


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
        dw = grad_weight.resize_as_(weight)

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

        return dw
