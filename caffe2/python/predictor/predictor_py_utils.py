## @package predictor_py_utils
# Module caffe2.python.predictor.predictor_py_utils
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core


def create_predict_net(predictor_export_meta):
    """
    Return the input prediction net.
    """
    # Construct a new net to clear the existing settings.
    net = core.Net(predictor_export_meta.predict_net.name or "predict")
    net.Proto().op.extend(predictor_export_meta.predict_net.op)
    net.Proto().external_input.extend(
        predictor_export_meta.inputs + predictor_export_meta.parameters)
    net.Proto().external_output.extend(predictor_export_meta.outputs)
    if predictor_export_meta.net_type is not None:
        net.Proto().type = predictor_export_meta.net_type
    if predictor_export_meta.num_workers is not None:
        net.Proto().num_workers = predictor_export_meta.num_workers
    return net.Proto()


def create_predict_init_net(ws, predictor_export_meta):
    """
    Return an initialization net that zero-fill all the input and
    output blobs, using the shapes from the provided workspace. This is
    necessary as there is no shape inference functionality in Caffe2.
    """
    net = core.Net("predict-init")

    def zero_fill(blob):
        shape = predictor_export_meta.shapes.get(blob)
        if shape is None:
            if blob not in ws.blobs:
                raise Exception(
                    "{} not in workspace but needed for shape: {}".format(
                        blob, ws.blobs))

            shape = ws.blobs[blob].fetch().shape
        net.ConstantFill([], blob, shape=shape, value=0.0)

    external_blobs = predictor_export_meta.inputs + \
        predictor_export_meta.outputs
    for blob in external_blobs:
        zero_fill(blob)

    net.Proto().external_input.extend(external_blobs)
    if predictor_export_meta.extra_init_net:
        net.AppendNet(predictor_export_meta.extra_init_net)
    return net.Proto()


def get_comp_name(string, name):
    if name:
        return string + '_' + name
    return string


def _ProtoMapGet(field, key):
    '''
    Given the key, get the value of the repeated field.
    Helper function used by protobuf since it doesn't have map construct
    '''
    for v in field:
        if (v.key == key):
            return v.value
    return None


def GetPlan(meta_net_def, key):
    return _ProtoMapGet(meta_net_def.plans, key)


def GetPlanOriginal(meta_net_def, key):
    return _ProtoMapGet(meta_net_def.plans, key)


def GetBlobs(meta_net_def, key):
    blobs = _ProtoMapGet(meta_net_def.blobs, key)
    if blobs is None:
        return []
    return blobs


def GetNet(meta_net_def, key):
    return _ProtoMapGet(meta_net_def.nets, key)


def GetNetOriginal(meta_net_def, key):
    return _ProtoMapGet(meta_net_def.nets, key)


def GetApplicationSpecificInfo(meta_net_def, key):
    return _ProtoMapGet(meta_net_def.applicationSpecificInfo, key)


def AddBlobs(meta_net_def, blob_name, blob_def):
    blobs = _ProtoMapGet(meta_net_def.blobs, blob_name)
    if blobs is None:
        blobs = meta_net_def.blobs.add()
        blobs.key = blob_name
        blobs = blobs.value
    for blob in blob_def:
        blobs.append(blob)


def AddPlan(meta_net_def, plan_name, plan_def):
    meta_net_def.plans.add(key=plan_name, value=plan_def)


def AddNet(meta_net_def, net_name, net_def):
    meta_net_def.nets.add(key=net_name, value=net_def)
