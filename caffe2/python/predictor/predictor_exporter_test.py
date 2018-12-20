from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import unittest
import numpy as np
from caffe2.python import cnn, workspace, core
from future.utils import viewitems

from caffe2.python.predictor_constants import predictor_constants as pc
import caffe2.python.predictor.predictor_exporter as pe
import caffe2.python.predictor.predictor_py_utils as pred_utils
from caffe2.proto import caffe2_pb2, metanet_pb2


class MetaNetDefTest(unittest.TestCase):
    def test_minimal(self):
        '''
        Tests that a NetsMap message can be created with a NetDef message
        '''
        # This calls the constructor for a metanet_pb2.NetsMap
        metanet_pb2.NetsMap(key="test_key", value=caffe2_pb2.NetDef())

    def test_adding_net(self):
        '''
        Tests that NetDefs can be added to MetaNetDefs
        '''
        meta_net_def = metanet_pb2.MetaNetDef()
        net_def = caffe2_pb2.NetDef()
        meta_net_def.nets.add(key="test_key", value=net_def)

class PredictorExporterTest(unittest.TestCase):
    def _create_model(self):
        m = cnn.CNNModelHelper()
        m.FC("data", "y",
             dim_in=5, dim_out=10,
             weight_init=m.XavierInit,
             bias_init=m.XavierInit)
        return m

    def setUp(self):
        np.random.seed(1)
        m = self._create_model()

        self.predictor_export_meta = pe.PredictorExportMeta(
            predict_net=m.net.Proto(),
            parameters=[str(b) for b in m.params],
            inputs=["data"],
            outputs=["y"],
            shapes={"y": (1, 10), "data": (1, 5)},
        )
        workspace.RunNetOnce(m.param_init_net)

        self.params = {
            param: workspace.FetchBlob(param)
            for param in self.predictor_export_meta.parameters}
        # Reset the workspace, to ensure net creation proceeds as expected.
        workspace.ResetWorkspace()

    def test_meta_constructor(self):
        '''
        Test that passing net itself instead of proto works
        '''
        m = self._create_model()
        pe.PredictorExportMeta(
            predict_net=m.net,
            parameters=m.params,
            inputs=["data"],
            outputs=["y"],
            shapes={"y": (1, 10), "data": (1, 5)},
        )

    def test_param_intersection(self):
        '''
        Test that passes intersecting parameters and input/output blobs
        '''
        m = self._create_model()
        with self.assertRaises(Exception):
            pe.PredictorExportMeta(
                predict_net=m.net,
                parameters=m.params,
                inputs=["data"] + m.params,
                outputs=["y"],
                shapes={"y": (1, 10), "data": (1, 5)},
            )
        with self.assertRaises(Exception):
            pe.PredictorExportMeta(
                predict_net=m.net,
                parameters=m.params,
                inputs=["data"],
                outputs=["y"] + m.params,
                shapes={"y": (1, 10), "data": (1, 5)},
            )

    def test_meta_net_def_net_runs(self):
        for param, value in viewitems(self.params):
            workspace.FeedBlob(param, value)

        extra_init_net = core.Net('extra_init')
        extra_init_net.ConstantFill('data', 'data', value=1.0)
        pem = pe.PredictorExportMeta(
            predict_net=self.predictor_export_meta.predict_net,
            parameters=self.predictor_export_meta.parameters,
            inputs=self.predictor_export_meta.inputs,
            outputs=self.predictor_export_meta.outputs,
            shapes=self.predictor_export_meta.shapes,
            extra_init_net=extra_init_net,
            net_type='dag',
        )

        db_type = 'minidb'
        db_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".{}".format(db_type))
        pe.save_to_db(
            db_type=db_type,
            db_destination=db_file.name,
            predictor_export_meta=pem)

        workspace.ResetWorkspace()

        meta_net_def = pe.load_from_db(
            db_type=db_type,
            filename=db_file.name,
        )

        self.assertTrue("data" not in workspace.Blobs())
        self.assertTrue("y" not in workspace.Blobs())

        init_net = pred_utils.GetNet(meta_net_def, pc.PREDICT_INIT_NET_TYPE)

        # 0-fills externalblobs blobs and runs extra_init_net
        workspace.RunNetOnce(init_net)

        self.assertTrue("data" in workspace.Blobs())
        self.assertTrue("y" in workspace.Blobs())

        print(workspace.FetchBlob("data"))
        np.testing.assert_array_equal(
            workspace.FetchBlob("data"), np.ones(shape=(1, 5)))
        np.testing.assert_array_equal(
            workspace.FetchBlob("y"), np.zeros(shape=(1, 10)))

        # Load parameters from DB
        global_init_net = pred_utils.GetNet(meta_net_def,
                                            pc.GLOBAL_INIT_NET_TYPE)
        workspace.RunNetOnce(global_init_net)

        # Run the net with a reshaped input and verify we are
        # producing good numbers (with our custom implementation)
        workspace.FeedBlob("data", np.random.randn(2, 5).astype(np.float32))
        predict_net = pred_utils.GetNet(meta_net_def, pc.PREDICT_NET_TYPE)
        self.assertEqual(predict_net.type, 'dag')
        workspace.RunNetOnce(predict_net)
        np.testing.assert_array_almost_equal(
            workspace.FetchBlob("y"),
            workspace.FetchBlob("data").dot(self.params["y_w"].T) +
            self.params["y_b"])

    def test_load_device_scope(self):
        for param, value in self.params.items():
            workspace.FeedBlob(param, value)

        pem = pe.PredictorExportMeta(
            predict_net=self.predictor_export_meta.predict_net,
            parameters=self.predictor_export_meta.parameters,
            inputs=self.predictor_export_meta.inputs,
            outputs=self.predictor_export_meta.outputs,
            shapes=self.predictor_export_meta.shapes,
            net_type='dag',
        )

        db_type = 'minidb'
        db_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".{}".format(db_type))
        pe.save_to_db(
            db_type=db_type,
            db_destination=db_file.name,
            predictor_export_meta=pem)

        workspace.ResetWorkspace()
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 1)):
            meta_net_def = pe.load_from_db(
                db_type=db_type,
                filename=db_file.name,
            )

        init_net = core.Net(pred_utils.GetNet(meta_net_def,
                            pc.GLOBAL_INIT_NET_TYPE))
        predict_init_net = core.Net(pred_utils.GetNet(
            meta_net_def, pc.PREDICT_INIT_NET_TYPE))

        # check device options
        for op in list(init_net.Proto().op) + list(predict_init_net.Proto().op):
            self.assertEqual(1, op.device_option.device_id)
            self.assertEqual(caffe2_pb2.CPU, op.device_option.device_type)

    def test_db_fails_without_params(self):
        with self.assertRaises(Exception):
            for db_type in ["minidb"]:
                db_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".{}".format(db_type))
                pe.save_to_db(
                    db_type=db_type,
                    db_destination=db_file.name,
                    predictor_export_meta=self.predictor_export_meta)
