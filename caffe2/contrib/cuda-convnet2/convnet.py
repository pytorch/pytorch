# Copyright 2014 Google Inc. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as n
import numpy.random as nr
import random as r
from python_util.util import *
from python_util.data import *
from python_util.options import *
from python_util.gpumodel import *
import sys
import math as m
import layer as lay
from convdata import ImageDataProvider, CIFARDataProvider, DummyConvNetLogRegDataProvider
from os import linesep as NL
import copy as cp
import os

class Driver(object):
    def __init__(self, convnet):
        self.convnet = convnet
        
    def on_start_batch(self, batch_data, train):
        pass
    
    def on_finish_batch(self):
        pass

class GradCheckDriver(Driver):
    def on_start_batch(self, batch_data, train):
        data = batch_data[2]
        self.convnet.libmodel.checkGradients(data)

class TrainingDriver(Driver):
    def on_start_batch(self, batch_data, train):
        data = batch_data[2]
        self.convnet.libmodel.startBatch(data, self.convnet.get_progress(), not train)

class MultiviewTestDriver(TrainingDriver):
    def on_start_batch(self, batch_data, train):
        self.write_output = False
        if train:
            TrainingDriver.on_start_batch(self, batch_data, train)
        else:
            data = batch_data[2]
            num_views = self.convnet.test_data_provider.num_views
            if self.convnet.test_out != "" and self.convnet.logreg_name != "":
                self.write_output = True
                self.test_file_name = os.path.join(self.convnet.test_out, 'test_preds_%d' % batch_data[1])
                self.probs = n.zeros((data[0].shape[1]/num_views, self.convnet.test_data_provider.get_num_classes()), dtype=n.single)
                self.convnet.libmodel.startMultiviewTest(data, num_views, self.probs, self.convnet.logreg_name)
            else:
                self.convnet.libmodel.startMultiviewTest(data, num_views)
            
    def on_finish_batch(self):
        if self.write_output:
            if not os.path.exists(self.convnet.test_out):
                os.makedirs(self.convnet.test_out)
            pickle(self.test_file_name,  {'data': self.probs,
                                          'note': 'generated from %s' % self.convnet.save_file})

class FeatureWriterDriver(Driver):
    def __init__(self, convnet):
        Driver.__init__(self, convnet)
        self.last_batch = convnet.test_batch_range[-1]
        
    def on_start_batch(self, batch_data, train):
        if train:
            raise ModelStateException("FeatureWriter must be used in conjunction with --test-only=1. It writes test data features.")
        
        self.batchnum, self.data = batch_data[1], batch_data[2]
        
        if not os.path.exists(self.convnet.feature_path):
            os.makedirs(self.convnet.feature_path)
        
        self.num_ftrs = self.convnet.layers[self.convnet.write_features]['outputs']
        self.ftrs = n.zeros((self.data[0].shape[1], self.num_ftrs), dtype=n.single)
        self.convnet.libmodel.startFeatureWriter(self.data, [self.ftrs], [self.convnet.write_features])
    
    def on_finish_batch(self):
        path_out = os.path.join(self.convnet.feature_path, 'data_batch_%d' % self.batchnum)
        pickle(path_out, {'data': self.ftrs, 'labels': self.data[1]})
        print "Wrote feature file %s" % path_out
        if self.batchnum == self.last_batch:
            pickle(os.path.join(self.convnet.feature_path, 'batches.meta'), {'source_model':self.convnet.load_file,
                                                                             'num_vis':self.num_ftrs,
                                                                             'batch_size': self.convnet.test_data_provider.batch_meta['batch_size']})

class ConvNet(IGPUModel):
    def __init__(self, op, load_dic, dp_params={}):
        filename_options = []
        for v in ('color_noise', 'multiview_test', 'inner_size', 'scalar_mean', 'minibatch_size'):
            dp_params[v] = op.get_value(v)

        IGPUModel.__init__(self, "ConvNet", op, load_dic, filename_options, dp_params=dp_params)
        
    def import_model(self):
        lib_name = "cudaconvnet._ConvNet"
        print "========================="
        print "Importing %s C++ module" % lib_name
        self.libmodel = __import__(lib_name,fromlist=['_ConvNet'])
        
    def init_model_lib(self):
        self.libmodel.initModel(self.layers,
                                self.device_ids,
                                self.minibatch_size,
                                self.conserve_mem)
        
    def init_model_state(self):
        ms = self.model_state
        layers = ms['layers'] if self.loaded_from_checkpoint else {}
        ms['layers'] = lay.LayerParser.parse_layers(os.path.join(self.layer_path, self.layer_def),
                                                    os.path.join(self.layer_path, self.layer_params), self, layers=layers)
        
        self.do_decouple_conv()
        self.do_unshare_weights()

        self.op.set_value('conv_to_local', [], parse=False)
        self.op.set_value('unshare_weights', [], parse=False)
        
        self.set_driver()
    
    def do_decouple_conv(self):
        # Convert convolutional layers to local
        if len(self.op.get_value('conv_to_local')) > 0:
            for lname in self.op.get_value('conv_to_local'):
                if self.model_state['layers'][lname]['type'] == 'conv':
                    lay.LocalLayerParser.conv_to_local(self.model_state['layers'], lname)
    
    def do_unshare_weights(self):
        # Decouple weight matrices
        if len(self.op.get_value('unshare_weights')) > 0:
            for name_str in self.op.get_value('unshare_weights'):
                if name_str:
                    name = lay.WeightLayerParser.get_layer_name(name_str)
                    if name is not None:
                        name, idx = name[0], name[1]
                        if name not in self.model_state['layers']:
                            raise ModelStateException("Layer '%s' does not exist; unable to unshare" % name)
                        layer = self.model_state['layers'][name]
                        lay.WeightLayerParser.unshare_weights(layer, self.model_state['layers'], matrix_idx=idx)
                    else:
                        raise ModelStateException("Invalid layer name '%s'; unable to unshare." % name_str)
    
    def set_driver(self):
        if self.op.get_value('check_grads'):
            self.driver = GradCheckDriver(self)
        elif self.op.get_value('multiview_test'):
            self.driver = MultiviewTestDriver(self)
        elif self.op.get_value('write_features'):
            self.driver = FeatureWriterDriver(self)
        else:
            self.driver = TrainingDriver(self)

    def fill_excused_options(self):
        if self.op.get_value('check_grads'):
            self.op.set_value('save_path', '')
            self.op.set_value('train_batch_range', '0')
            self.op.set_value('test_batch_range', '0')
            self.op.set_value('data_path', '')
            
    # Make sure the data provider returned data in proper format
    def parse_batch_data(self, batch_data, train=True):
        if max(d.dtype != n.single for d in batch_data[2]):
            raise DataProviderException("All matrices returned by data provider must consist of single-precision floats.")
        return batch_data

    def start_batch(self, batch_data, train=True):
        self.driver.on_start_batch(batch_data, train)
            
    def finish_batch(self):
        ret = IGPUModel.finish_batch(self)
        self.driver.on_finish_batch()
        return ret
    
    def print_iteration(self):
        print "%d.%d (%.2f%%)..." % (self.epoch, self.batchnum, 100 * self.get_progress()),
        
    def print_train_time(self, compute_time_py):
        print "(%.3f sec)" % (compute_time_py)
        
    def print_costs(self, cost_outputs):
        costs, num_cases = cost_outputs[0], cost_outputs[1]
        children = set()
        for errname in costs:
            if sum(errname in self.layers[z]['children'] for z in costs) == 0:
#                print self.layers[errname]['children']
                for child in set(self.layers[errname]['children']) & set(costs.keys()):
                    costs[errname] = [v + u for v, u in zip(costs[errname], costs[child])]
                    children.add(child)
            
                filtered_costs = eval(self.layers[errname]['outputFilter'])(costs[errname], num_cases)
                print "%s: " % errname,
                if 'outputFilterFormatter' not in self.layers[errname]:
                    print ", ".join("%.6f" % v for v in filtered_costs),
                else:
                    print eval(self.layers[errname]['outputFilterFormatter'])(self,filtered_costs),
                if m.isnan(filtered_costs[0]) or m.isinf(filtered_costs[0]):
                    print "<- error nan or inf!"
                    sys.exit(1)
        for c in children:
            del costs[c]
        
    def print_train_results(self):
        self.print_costs(self.train_outputs[-1])
        
    def print_test_status(self):
        pass
        
    def print_test_results(self):
        print NL + "======================Test output======================"
        self.print_costs(self.test_outputs[-1])
        if not self.test_only:
            print NL + "----------------------Averages-------------------------"
            self.print_costs(self.aggregate_test_outputs(self.test_outputs[-len(self.test_batch_range):]))
        print NL + "-------------------------------------------------------",
        for name,val in sorted(self.layers.items(), key=lambda x: x[1]['id']): # This is kind of hacky but will do for now.
            l = self.layers[name]
            if 'weights' in l:
                wscales = [(l['name'], i, n.mean(n.abs(w)), n.mean(n.abs(wi))) for i,(w,wi) in enumerate(zip(l['weights'],l['weightsInc']))]
                print ""
                print NL.join("Layer '%s' weights[%d]: %e [%e] [%e]" % (s[0], s[1], s[2], s[3], s[3]/s[2] if s[2] > 0 else 0) for s in wscales),
                print "%sLayer '%s' biases: %e [%e]" % (NL, l['name'], n.mean(n.abs(l['biases'])), n.mean(n.abs(l['biasesInc']))),
        print ""
        
    def conditional_save(self):
        self.save_state()
        
    def aggregate_test_outputs(self, test_outputs):
        test_outputs = cp.deepcopy(test_outputs)
        num_cases = sum(t[1] for t in test_outputs)
        for i in xrange(1 ,len(test_outputs)):
            for k,v in test_outputs[i][0].items():
                for j in xrange(len(v)):
                    test_outputs[0][0][k][j] += test_outputs[i][0][k][j]
        
        return (test_outputs[0][0], num_cases)
    
    @classmethod
    def get_options_parser(cls):
        op = IGPUModel.get_options_parser()
        op.add_option("mini", "minibatch_size", IntegerOptionParser, "Minibatch size", default=128)
        op.add_option("layer-def", "layer_def", StringOptionParser, "Layer definition file", set_once=False)
        op.add_option("layer-params", "layer_params", StringOptionParser, "Layer parameter file")
        op.add_option("layer-path", "layer_path", StringOptionParser, "Layer file path prefix", default="")
        op.add_option("check-grads", "check_grads", BooleanOptionParser, "Check gradients and quit?", default=0, excuses=['data_path','save_path', 'save_file_override', 'train_batch_range','test_batch_range'])
        op.add_option("multiview-test", "multiview_test", BooleanOptionParser, "Cropped DP: test on multiple patches?", default=0)
        op.add_option("inner-size", "inner_size", IntegerOptionParser, "Cropped DP: crop size (0 = don't crop)", default=0, set_once=True)
        op.add_option("conv-to-local", "conv_to_local", ListOptionParser(StringOptionParser), "Convert given conv layers to unshared local", default=[])
        op.add_option("unshare-weights", "unshare_weights", ListOptionParser(StringOptionParser), "Unshare weight matrices in given layers", default=[])
        op.add_option("conserve-mem", "conserve_mem", BooleanOptionParser, "Conserve GPU memory (slower)?", default=0)
        op.add_option("color-noise", "color_noise", FloatOptionParser, "Add PCA noise to color channels with given scale", default=0.0)
        op.add_option("test-out", "test_out", StringOptionParser, "Output test case predictions to given path", default="", requires=['logreg_name', 'multiview_test'])
        op.add_option("logreg-name", "logreg_name", StringOptionParser, "Logreg cost layer name (for --test-out)", default="")
        op.add_option("scalar-mean", "scalar_mean", FloatOptionParser, "Subtract this scalar from image (-1 = don't)", default=-1)
        
        op.add_option("write-features", "write_features", StringOptionParser, "Write test data features from given layer", default="", requires=['feature-path'])
        op.add_option("feature-path", "feature_path", StringOptionParser, "Write test data features to this path (to be used with --write-features)", default="")

        op.delete_option('max_test_err')
        op.options["testing_freq"].default = 57
        op.options["num_epochs"].default = 50000
        op.options['dp_type'].default = None

        DataProvider.register_data_provider('dummy-lr-n', 'Dummy ConvNet logistic regression', DummyConvNetLogRegDataProvider)
        DataProvider.register_data_provider('image', 'JPEG-encoded image data provider', ImageDataProvider)
        DataProvider.register_data_provider('cifar', 'CIFAR-10 data provider', CIFARDataProvider)
  
        return op

if __name__ == "__main__":
#    nr.seed(6)

    op = ConvNet.get_options_parser()

    op, load_dic = IGPUModel.parse_options(op)
    model = ConvNet(op, load_dic)
    model.start()
