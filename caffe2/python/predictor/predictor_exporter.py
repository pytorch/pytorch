## @package predictor_exporter
# Module caffe2.python.predictor.predictor_exporter
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.proto import metanet_pb2
from caffe2.python import workspace, core, scope
from caffe2.python.predictor_constants import predictor_constants
import caffe2.python.predictor.serde as serde
import caffe2.python.predictor.predictor_py_utils as utils
from builtins import bytes
import collections


def get_predictor_exporter_helper(submodelNetName):
    """ constracting stub for the PredictorExportMeta
        Only used to construct names to subfields,
        such as calling to predict_net_name
        Args:
            submodelNetName - name of the model
    """
    stub_net = core.Net(submodelNetName)
    pred_meta = PredictorExportMeta(predict_net=stub_net,
                                    parameters=[],
                                    inputs=[],
                                    outputs=[],
                                    shapes=None,
                                    name=submodelNetName,
                                    extra_init_net=None)
    return pred_meta


class PredictorExportMeta(collections.namedtuple(
    'PredictorExportMeta',
        'predict_net, parameters, inputs, outputs, shapes, name, \
        extra_init_net, net_type, num_workers, trainer_prefix')):
    """
    Metadata to be used for serializaing a net.

    parameters, inputs, outputs could be either BlobReference or blob's names

    predict_net can be either core.Net, NetDef, PlanDef or object

    Override the named tuple to provide optional name parameter.
    name will be used to identify multiple prediction nets.

    net_type is the type field in caffe2 NetDef - can be 'simple', 'dag', etc.

    num_workers specifies for net type 'dag' how many threads should run ops

    trainer_prefix specifies the type of trainer.
    """
    def __new__(
        cls,
        predict_net,
        parameters,
        inputs,
        outputs,
        shapes=None,
        name="",
        extra_init_net=None,
        net_type=None,
        num_workers=None,
        trainer_prefix=None,
    ):
        inputs = [str(i) for i in inputs]
        outputs = [str(o) for o in outputs]
        assert len(set(inputs)) == len(inputs), (
            "All inputs to the predictor should be unique")
        parameters = [str(p) for p in parameters]
        assert set(parameters).isdisjoint(inputs), (
            "Parameters and inputs are required to be disjoint. "
            "Intersection: {}".format(set(parameters).intersection(inputs)))
        assert set(parameters).isdisjoint(outputs), (
            "Parameters and outputs are required to be disjoint. "
            "Intersection: {}".format(set(parameters).intersection(outputs)))
        shapes = shapes or {}

        if isinstance(predict_net, (core.Net, core.Plan)):
            predict_net = predict_net.Proto()

        assert isinstance(predict_net, (caffe2_pb2.NetDef, caffe2_pb2.PlanDef))
        return super(PredictorExportMeta, cls).__new__(
            cls, predict_net, parameters, inputs, outputs, shapes, name,
            extra_init_net, net_type, num_workers, trainer_prefix)

    def inputs_name(self):
        return utils.get_comp_name(predictor_constants.INPUTS_BLOB_TYPE,
                                   self.name)

    def outputs_name(self):
        return utils.get_comp_name(predictor_constants.OUTPUTS_BLOB_TYPE,
                                   self.name)

    def parameters_name(self):
        return utils.get_comp_name(predictor_constants.PARAMETERS_BLOB_TYPE,
                                   self.name)

    def global_init_name(self):
        return utils.get_comp_name(predictor_constants.GLOBAL_INIT_NET_TYPE,
                                   self.name)

    def predict_init_name(self):
        return utils.get_comp_name(predictor_constants.PREDICT_INIT_NET_TYPE,
                                   self.name)

    def predict_net_name(self):
        return utils.get_comp_name(predictor_constants.PREDICT_NET_TYPE,
                                   self.name)

    def train_init_plan_name(self):
        plan_name = utils.get_comp_name(predictor_constants.TRAIN_INIT_PLAN_TYPE,
                                   self.name)
        return self.trainer_prefix + '_' + plan_name \
            if self.trainer_prefix else plan_name

    def train_plan_name(self):
        plan_name = utils.get_comp_name(predictor_constants.TRAIN_PLAN_TYPE,
                                   self.name)
        return self.trainer_prefix + '_' + plan_name \
            if self.trainer_prefix else plan_name


def prepare_prediction_net(filename, db_type, device_option=None):
    '''
    Helper function which loads all required blobs from the db
    and returns prediction net ready to be used
    '''
    metanet_def = load_from_db(filename, db_type, device_option)

    global_init_net = utils.GetNet(
        metanet_def, predictor_constants.GLOBAL_INIT_NET_TYPE)
    workspace.RunNetOnce(global_init_net)

    predict_init_net = utils.GetNet(
        metanet_def, predictor_constants.PREDICT_INIT_NET_TYPE)
    workspace.RunNetOnce(predict_init_net)

    predict_net = core.Net(
        utils.GetNet(metanet_def, predictor_constants.PREDICT_NET_TYPE))
    workspace.CreateNet(predict_net)

    return predict_net


def _global_init_net(predictor_export_meta):
    net = core.Net("global-init")
    net.Load(
        [predictor_constants.PREDICTOR_DBREADER],
        predictor_export_meta.parameters)
    net.Proto().external_input.extend([predictor_constants.PREDICTOR_DBREADER])
    net.Proto().external_output.extend(predictor_export_meta.parameters)

    # Add the model_id in the predict_net to the global_init_net
    utils.AddModelIdArg(predictor_export_meta, net.Proto())
    return net.Proto()


def get_meta_net_def(predictor_export_meta, ws=None):
    """
    """

    ws = ws or workspace.C.Workspace.current
    meta_net_def = metanet_pb2.MetaNetDef()

    # Predict net is the core network that we use.
    utils.AddNet(meta_net_def, predictor_export_meta.predict_init_name(),
                 utils.create_predict_init_net(ws, predictor_export_meta))
    utils.AddNet(meta_net_def, predictor_export_meta.global_init_name(),
                 _global_init_net(predictor_export_meta))
    utils.AddNet(meta_net_def, predictor_export_meta.predict_net_name(),
                 utils.create_predict_net(predictor_export_meta))
    utils.AddBlobs(meta_net_def, predictor_export_meta.parameters_name(),
                   predictor_export_meta.parameters)
    utils.AddBlobs(meta_net_def, predictor_export_meta.inputs_name(),
                   predictor_export_meta.inputs)
    utils.AddBlobs(meta_net_def, predictor_export_meta.outputs_name(),
                   predictor_export_meta.outputs)
    return meta_net_def


def set_model_info(meta_net_def, project_str, model_class_str, version):
    assert isinstance(meta_net_def, metanet_pb2.MetaNetDef)
    meta_net_def.modelInfo.project = project_str
    meta_net_def.modelInfo.modelClass = model_class_str
    meta_net_def.modelInfo.version = version


def save_to_db(db_type, db_destination, predictor_export_meta):
    meta_net_def = get_meta_net_def(predictor_export_meta)
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
        workspace.FeedBlob(
            predictor_constants.META_NET_DEF,
            serde.serialize_protobuf_struct(meta_net_def)
        )

    blobs_to_save = [predictor_constants.META_NET_DEF] + \
        predictor_export_meta.parameters
    op = core.CreateOperator(
        "Save",
        blobs_to_save, [],
        absolute_path=True,
        db=db_destination, db_type=db_type)

    workspace.RunOperatorOnce(op)


def load_from_db(filename, db_type, device_option=None):
    # global_init_net in meta_net_def will load parameters from
    # predictor_constants.PREDICTOR_DBREADER
    create_db = core.CreateOperator(
        'CreateDB', [],
        [core.BlobReference(predictor_constants.PREDICTOR_DBREADER)],
        db=filename, db_type=db_type)
    assert workspace.RunOperatorOnce(create_db), (
        'Failed to create db {}'.format(filename))

    # predictor_constants.META_NET_DEF is always stored before the parameters
    load_meta_net_def = core.CreateOperator(
        'Load',
        [core.BlobReference(predictor_constants.PREDICTOR_DBREADER)],
        [core.BlobReference(predictor_constants.META_NET_DEF)])
    assert workspace.RunOperatorOnce(load_meta_net_def)

    blob = workspace.FetchBlob(predictor_constants.META_NET_DEF)
    meta_net_def = serde.deserialize_protobuf_struct(
        blob if isinstance(blob, bytes)
        else str(blob).encode('utf-8'),
        metanet_pb2.MetaNetDef)

    if device_option is None:
        device_option = scope.CurrentDeviceScope()

    if device_option is not None:
        # Set the device options of all loaded blobs
        for kv in meta_net_def.nets:
            net = kv.value
            for op in net.op:
                op.device_option.CopyFrom(device_option)

    return meta_net_def
