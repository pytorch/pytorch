#!/usr/bin/env python2

from caffe2.proto import caffe2_pb2, caffe2_legacy_pb2
from caffe.proto import caffe_pb2
from caffe2.python import core, utils


def _StateMeetsRule(state, rule):
    """A function that reproduces Caffe's StateMeetsRule functionality."""
    if rule.HasField('phase') and rule.phase != state.phase:
        return False
    if rule.HasField('min_level') and state.level < rule.min_level:
        return False
    if rule.HasField('max_level') and state.level > rule.max_lavel:
        return False
    curr_stages = set(list(state.stage))
    # all stages in rule.stages should be in, otherwise it's not a match.
    if len(rule.stage) and any([s not in curr_stages for s in rule.stage]):
        return False
    # none of the stage in rule.stages should be in, otherwise it's not a match.
    if len(rule.not_stage) and any([s in curr_stages for s in rule.not_stage]):
        return False
    # If none of the nonmatch happens, return True.
    return True


def _ShouldInclude(net_state, layer):
    """A function that reproduces Caffe's inclusion and exclusion rule."""
    ret = (len(layer.include) == 0)
    # check exclude rules: if any exclusion is met, we shouldn't include.
    ret &= not any([_StateMeetsRule(net_state, rule) for rule in layer.exclude])
    if len(layer.include):
        # check include rules: if any inclusion is met, we should include.
        ret |= any([_StateMeetsRule(net_state, rule) for rule in layer.include])
    return ret


class TranslatorRegistry(object):
    registry_ = {}

    @classmethod
    def Register(cls, op_name):
        """A decorator for registering gradient mappings."""

        def Wrapper(func):
            cls.registry_[op_name] = func
            return func

        return Wrapper

    @classmethod
    def TranslateLayer(cls, layer, pretrained_blobs, is_test):
        try:
            caffe_ops, params = cls.registry_[layer.type](
                layer, pretrained_blobs, is_test)
        except KeyError:
            raise KeyError('No translator registered for layer: %s yet.' %
                           str(layer))
        if caffe_ops is None:
            caffe_ops = []
        if type(caffe_ops) is not list:
            caffe_ops = [caffe_ops]
        return caffe_ops, params

    @classmethod
    def TranslateModel(
        cls,
        caffe_net,
        pretrained_net,
        is_test=False,
        net_state=None,
    ):
        net_state = caffe_pb2.NetState() if net_state is None else net_state
        net = caffe2_pb2.NetDef()
        net.name = caffe_net.name
        net_params = caffe2_pb2.TensorProtos()
        if len(caffe_net.layer) == 0:
            raise ValueError(
                'I think something is wrong. This translation script '
                'only accepts new style layers that are stored in the '
                'layer field.'
            )
        for layer in caffe_net.layer:
            if not _ShouldInclude(net_state, layer):
                print('Current net state does not need layer {}'
                      .format(layer.name))
                continue
            print('Translate layer {}'.format(layer.name))
            # Get pretrained one
            pretrained_layers = (
                [l for l in pretrained_net.layer
                 if l.name == layer.name] + [l
                                             for l in pretrained_net.layers
                                             if l.name == layer.name]
            )
            if len(pretrained_layers) > 1:
                raise ValueError(
                    'huh? more than one pretrained layer of one name?')
            elif len(pretrained_layers) == 1:
                pretrained_blobs = [
                    utils.CaffeBlobToNumpyArray(blob)
                    for blob in pretrained_layers[0].blobs
                ]
            else:
                # No pretrained layer for the given layer name. We'll just pass
                # no parameter blobs.
                # print 'No pretrained layer for layer', layer.name
                pretrained_blobs = []
            operators, params = cls.TranslateLayer(
                layer, pretrained_blobs, is_test)
            net.op.extend(operators)
            net_params.protos.extend(params)
        return net, net_params


def TranslateModel(*args, **kwargs):
    return TranslatorRegistry.TranslateModel(*args, **kwargs)


def ConvertTensorProtosToInitNet(net_params):
    """Takes the net_params returned from TranslateModel, and wrap it as an
    init net that contain GivenTensorFill.

    This is a very simple feature that only works with float tensors, and is
    only intended to be used in an environment where you want a single
    initialization file - for more complex cases, use a db to store the
    parameters.
    """
    init_net = caffe2_pb2.NetDef()
    for tensor in net_params.protos:
        if len(tensor.float_data) == 0:
            raise RuntimeError(
                "Only float tensors are supported in this util.")
        op = core.CreateOperator(
            "GivenTensorFill", [], [tensor.name],
            arg=[
                utils.MakeArgument("shape", list(tensor.dims)),
                utils.MakeArgument("values", tensor.float_data)])
        init_net.op.extend([op])
    return init_net


def BaseTranslate(layer, caffe2_type):
    """A simple translate interface that maps the layer input and output."""
    caffe2_op = caffe2_pb2.OperatorDef()
    caffe2_op.type = caffe2_type
    caffe2_op.input.extend(layer.bottom)
    caffe2_op.output.extend(layer.top)
    return caffe2_op


def AddArgument(op, key, value):
    """Makes an argument based on the value type."""
    op.arg.extend([utils.MakeArgument(key, value)])

################################################################################
# Common translators for layers.
################################################################################


@TranslatorRegistry.Register("Input")
def TranslateInput(layer, pretrained_blobs, is_test):
    return [], []


@TranslatorRegistry.Register("Data")
def TranslateData(layer, pretrained_blobs, is_test):
    return [], []


# A function used in convolution, pooling and deconvolution to deal with
# conv pool specific parameters.
def _TranslateStridePadKernelHelper(param, caffe_op):
    try:
        if (len(param.stride) > 1 or len(param.kernel_size) > 1 or
                len(param.pad) > 1):
            raise NotImplementedError(
                "Translator currently does not support non-conventional "
                "pad/kernel/stride settings."
            )
        stride = param.stride[0] if len(param.stride) else 1
        pad = param.pad[0] if len(param.pad) else 0
        kernel = param.kernel_size[0] if len(param.kernel_size) else 0
    except TypeError:
        # This catches the case of a PoolingParameter, in which case we are
        # having non-repeating pad, stride and kernel.
        stride = param.stride
        pad = param.pad
        kernel = param.kernel_size
    # Get stride
    if param.HasField("stride_h") or param.HasField("stride_w"):
        AddArgument(caffe_op, "stride_h", param.stride_h)
        AddArgument(caffe_op, "stride_w", param.stride_w)
    else:
        AddArgument(caffe_op, "stride", stride)
    # Get pad
    if param.HasField("pad_h") or param.HasField("pad_w"):
        if param.pad_h == param.pad_w:
            AddArgument(caffe_op, "pad", param.pad_h)
        else:
            AddArgument(caffe_op, "pad_t", param.pad_h)
            AddArgument(caffe_op, "pad_b", param.pad_h)
            AddArgument(caffe_op, "pad_l", param.pad_w)
            AddArgument(caffe_op, "pad_r", param.pad_w)
    else:
        AddArgument(caffe_op, "pad", pad)
    # Get kernel
    if param.HasField("kernel_h") or param.HasField("kernel_w"):
        AddArgument(caffe_op, "kernel_h", param.kernel_h)
        AddArgument(caffe_op, "kernel_w", param.kernel_w)
    else:
        AddArgument(caffe_op, "kernel", kernel)


@TranslatorRegistry.Register("Convolution")
def TranslateConv(layer, pretrained_blobs, is_test):
    param = layer.convolution_param
    if param.group > 1:
        return TranslateConvWithGroups(layer, pretrained_blobs, is_test)
    # If there is no odd things, we will basically translate it to a standard
    # caffe2 op.
    caffe_op = BaseTranslate(layer, "Conv")
    output = caffe_op.output[0]
    caffe_op.input.extend([output + '_w', output + '_b'])
    _TranslateStridePadKernelHelper(param, caffe_op)
    weight = utils.NumpyArrayToCaffe2Tensor(pretrained_blobs[0], output + '_w')
    bias = utils.NumpyArrayToCaffe2Tensor(
        pretrained_blobs[1].flatten(), output + '_b'
    )
    return caffe_op, [weight, bias]


def TranslateConvWithGroups(layer, pretrained_blobs, is_test):
    print(
        "Legacy warning: convolution with groups seem to be less and less " +
        "popular, so we no longer have it as a first-class citizen op. " +
        "Instead, we will simulate it with depth split followed by conv " +
        "followed by depth concat."
    )
    caffe_ops = []
    caffe_params = []
    param = layer.convolution_param
    weight, bias = pretrained_blobs
    bias = bias.flatten()
    n, c, h, w = weight.shape
    g = param.group  # group
    od = int(n / g)  # output dimension
    if (od * g != n):
        # This should not happen: n should always be divisible by g.
        raise ValueError("This should not happen.")
    output = layer.top[0]
    # first, depth_split
    depth_split_op = core.CreateOperator(
        "DepthSplit",
        str(layer.bottom[0]),
        ['_' + output + '_gconv_split_' + str(i) for i in range(g)],
        split=[c for i in range(g)],
        order="NCHW"
    )
    caffe_ops.append(depth_split_op)
    # second, convolutions
    for i in range(g):
        # convolution layer i
        this_weight = utils.NumpyArrayToCaffe2Tensor(
            weight[i * od:(i + 1) * od], output + '_gconv_' + str(i) + '_w'
        )
        this_bias = utils.NumpyArrayToCaffe2Tensor(
            bias[i * od:(i + 1) * od], output + '_gconv_' + str(i) + '_b'
        )
        conv_op = core.CreateOperator(
            "Conv",
            [depth_split_op.output[i], this_weight.name, this_bias.name],
            ['_' + output + '_gconv_conv_' + str(i)],
            order="NCHW"
        )
        _TranslateStridePadKernelHelper(param, conv_op)
        caffe_ops.append(conv_op)
        caffe_params.extend([this_weight, this_bias])
    # third, depth concat
    depth_concat_op = core.CreateOperator(
        "Concat",
        ['_' + output + '_gconv_conv_' + str(i) for i in range(g)],
        [output, '_' + output + '_gconv_concat_dims'],
        order="NCHW"
    )
    caffe_ops.append(depth_concat_op)
    return caffe_ops, caffe_params


@TranslatorRegistry.Register("Deconvolution")
def TranslateDeconv(layer, pretrained_blobs, is_test):
    param = layer.convolution_param
    if param.group > 1:
        raise NotImplementedError(
            "Translator currently does not support group deconvolution."
        )
    caffe_op = BaseTranslate(layer, "ConvTranspose")
    output = caffe_op.output[0]
    _TranslateStridePadKernelHelper(param, caffe_op)
    caffe_op.input.extend([output + '_w', output + '_b'])
    AddArgument(caffe_op, "order", "NCHW")
    weight = utils.NumpyArrayToCaffe2Tensor(pretrained_blobs[0], output + '_w')
    bias = utils.NumpyArrayToCaffe2Tensor(
        pretrained_blobs[1].flatten(), output + '_b'
    )
    return caffe_op, [weight, bias]


@TranslatorRegistry.Register("ReLU")
def TranslateRelu(layer, pretrained_blobs, is_test):
    return BaseTranslate(layer, "Relu"), []


@TranslatorRegistry.Register("Pooling")
def TranslatePool(layer, pretrained_blobs, is_test):
    param = layer.pooling_param
    if param.pool == caffe_pb2.PoolingParameter.MAX:
        caffe_op = BaseTranslate(layer, "MaxPool")
    elif param.pool == caffe_pb2.PoolingParameter.AVE:
        caffe_op = BaseTranslate(layer, "AveragePool")
    _TranslateStridePadKernelHelper(param, caffe_op)
    AddArgument(caffe_op, "order", "NCHW")
    AddArgument(caffe_op, "legacy_pad",
                caffe2_legacy_pb2.CAFFE_LEGACY_POOLING)
    return caffe_op, []


@TranslatorRegistry.Register("LRN")
def TranslateLRN(layer, pretrained_blobs, is_test):
    caffe_op = BaseTranslate(layer, "LRN")
    caffe_op.output.extend(['_' + caffe_op.output[0] + '_scale'])
    param = layer.lrn_param
    if param.norm_region != caffe_pb2.LRNParameter.ACROSS_CHANNELS:
        raise ValueError(
            "Does not support norm region other than across channels.")
    AddArgument(caffe_op, "size", int(param.local_size))
    AddArgument(caffe_op, "alpha", float(param.alpha))
    AddArgument(caffe_op, "beta", float(param.beta))
    AddArgument(caffe_op, "bias", float(param.k))
    AddArgument(caffe_op, "order", "NCHW")
    return caffe_op, []


@TranslatorRegistry.Register("InnerProduct")
def TranslateInnerProduct(layer, pretrained_blobs, is_test):
    caffe_op = BaseTranslate(layer, "FC")
    output = caffe_op.output[0]
    caffe_op.input.extend([output + '_w', output + '_b'])
    weight = utils.NumpyArrayToCaffe2Tensor(
        pretrained_blobs[0][0, 0], output + '_w'
    )
    bias = utils.NumpyArrayToCaffe2Tensor(
        pretrained_blobs[1].flatten(), output + '_b'
    )
    return caffe_op, [weight, bias]


@TranslatorRegistry.Register("Dropout")
def TranslateDropout(layer, pretrained_blobs, is_test):
    caffe_op = BaseTranslate(layer, "Dropout")
    caffe_op.output.extend(['_' + caffe_op.output[0] + '_mask'])
    param = layer.dropout_param
    AddArgument(caffe_op, "ratio", param.dropout_ratio)
    if (is_test):
        AddArgument(caffe_op, "is_test", 1)
    return caffe_op, []


@TranslatorRegistry.Register("Softmax")
def TranslateSoftmax(layer, pretrained_blobs, is_test):
    caffe_op = BaseTranslate(layer, "Softmax")
    return caffe_op, []


@TranslatorRegistry.Register("SoftmaxWithLoss")
def TranslateSoftmaxWithLoss(layer, pretrained_blobs, is_test):
    softmax_op = core.CreateOperator(
        "Softmax", [layer.bottom[0]],
        layer.bottom[0] + "_translator_autogen_softmax")
    xent_op = core.CreateOperator(
        "LabelCrossEntropy",
        [softmax_op.output[0], layer.bottom[1]],
        layer.bottom[0] + "_translator_autogen_xent")
    loss_op = core.CreateOperator(
        "AveragedLoss",
        xent_op.output[0],
        layer.top[0])
    return [softmax_op, xent_op, loss_op], []


@TranslatorRegistry.Register("Accuracy")
def TranslateAccuracy(layer, pretrained_blobs, is_test):
    caffe_op = BaseTranslate(layer, "Accuracy")
    if layer.accuracy_param.top_k != 1:
        print("Warning: Translation does not support Accuracy layers top_k >1.")
    return caffe_op, []


@TranslatorRegistry.Register("Concat")
def TranslateConcat(layer, pretrained_blobs, is_test):
    caffe_op = BaseTranslate(layer, "Concat")
    caffe_op.output.extend(['_' + caffe_op.output[0] + '_dims'])
    AddArgument(caffe_op, "order", "NCHW")
    return caffe_op, []


@TranslatorRegistry.Register("TanH")
def TranslateTanH(layer, pretrained_blobs, is_test):
    caffe_op = BaseTranslate(layer, "Tanh")
    return caffe_op, []


@TranslatorRegistry.Register("InstanceNorm")
def TranslateInstanceNorm(layer, pretrained_blobs, is_test):
    caffe_op = BaseTranslate(layer, "InstanceNorm")
    output = caffe_op.output[0]
    weight = utils.NumpyArrayToCaffe2Tensor(
        pretrained_blobs[0].flatten(), output + '_w')
    bias = utils.NumpyArrayToCaffe2Tensor(
        pretrained_blobs[1].flatten(), output + '_b')
    caffe_op.input.extend([output + '_w', output + '_b'])
    AddArgument(caffe_op, "order", "NCHW")
    return caffe_op, [weight, bias]
