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
import os
from time import time, asctime, localtime, strftime
from util import *
from data import *
from options import *
from math import ceil, floor, sqrt
from data import DataProvider, dp_types
import sys
import shutil
import platform
from os import linesep as NL
from threading import Thread
import tempfile as tf

class ModelStateException(Exception):
    pass

class CheckpointWriter(Thread):
    def __init__(self, path, dic):
        Thread.__init__(self)
        self.path = path
        self.dic = dic
        
    def run(self):
        save_dir = os.path.dirname(self.path)
        save_file = os.path.basename(self.path)
        # Write checkpoint to temporary filename
        tmpfile = tf.NamedTemporaryFile(dir=os.path.dirname(save_dir), delete=False)
        pickle(tmpfile, self.dic) # Also closes tf
        # Move it to final filename
        os.rename(tmpfile.name, self.path)
        # Delete old checkpoints
        for f in os.listdir(save_dir):
            if f != save_file:
                os.remove(os.path.join(save_dir, f))

# GPU Model interface
class IGPUModel:
    def __init__(self, model_name, op, load_dic, filename_options=[], dp_params={}):
        # these are input parameters
        self.model_name = model_name
        self.op = op
        self.options = op.options
        self.load_dic = load_dic
        self.filename_options = filename_options
        self.dp_params = dp_params
        self.device_ids = self.op.get_value('gpu')
        self.fill_excused_options()
        self.checkpoint_writer = None
        #assert self.op.all_values_given()
        
        for o in op.get_options_list():
            setattr(self, o.name, o.value)
        self.loaded_from_checkpoint = load_dic is not None
        # these are things that the model must remember but they're not input parameters
        if self.loaded_from_checkpoint:
            self.model_state = load_dic["model_state"]
            self.save_file = self.options["save_file_override"].value if self.options["save_file_override"].value_given else self.options['load_file'].value
            if not os.path.isdir(self.save_file) and os.path.exists(self.save_file):
                self.save_file = os.path.dirname(self.save_file)
#            print self.options["save_file_override"].value, self.save_file
        else:
            self.model_state = {}
            self.save_file = self.options["save_file_override"].value if self.options["save_file_override"].value_given else os.path.join(self.options['save_path'].value, model_name + "_" + '_'.join(['%s_%s' % (char, self.options[opt].get_str_value()) for opt, char in filename_options]) + '_' + strftime('%Y-%m-%d_%H.%M.%S'))
            self.model_state["train_outputs"] = []
            self.model_state["test_outputs"] = []
            self.model_state["epoch"] = 1
            self.model_state["batchnum"] = self.train_batch_range[0]
#            print self.save_file

        self.init_data_providers()
        if load_dic: 
            self.train_data_provider.advance_batch()
            
        # model state often requries knowledge of data provider, so it's initialized after
        try:
            self.init_model_state()
        except ModelStateException, e:
            print e
            sys.exit(1)
        for var, val in self.model_state.iteritems():
            setattr(self, var, val)
            
        self.import_model()
        self.init_model_lib()

    def import_model(self):
        print "========================="
        print "Importing %s C++ module" % ('_' + self.model_name)
        self.libmodel = __import__('_' + self.model_name) 
                   
    def fill_excused_options(self):
        pass
    
    def init_data_providers(self):
        self.dp_params['convnet'] = self
        try:
            self.test_data_provider = DataProvider.get_instance(self.data_path, self.test_batch_range,
                                                                type=self.dp_type, dp_params=self.dp_params, test=True)
            self.train_data_provider = DataProvider.get_instance(self.data_path, self.train_batch_range,
                                                                     self.model_state["epoch"], self.model_state["batchnum"],
                                                                     type=self.dp_type, dp_params=self.dp_params, test=False)
        except DataProviderException, e:
            print "Unable to create data provider: %s" % e
            self.print_data_providers()
            sys.exit()
        
    def init_model_state(self):
        pass
       
    def init_model_lib(self):
        pass
    
    def start(self):
        if self.test_only:
            self.test_outputs += [self.get_test_error()]
            self.print_test_results()
        else:
            self.train()
        self.cleanup()
        if self.force_save:
            self.save_state().join()
        sys.exit(0)
    
    def train(self):
        print "========================="
        print "Training %s" % self.model_name
        self.op.print_values()
        print "========================="
        self.print_model_state()
        print "Running on CUDA device(s) %s" % ", ".join("%d" % d for d in self.device_ids)
        print "Current time: %s" % asctime(localtime())
        print "Saving checkpoints to %s" % self.save_file
        print "========================="
        next_data = self.get_next_batch()
        while self.epoch <= self.num_epochs:
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            self.print_iteration()
            sys.stdout.flush()
            
            compute_time_py = time()
            self.start_batch(data)
            
            # load the next batch while the current one is computing
            next_data = self.get_next_batch()
            
            batch_output = self.finish_batch()
            self.train_outputs += [batch_output]
            self.print_train_results()

            if self.get_num_batches_done() % self.testing_freq == 0:
                self.sync_with_host()
                self.test_outputs += [self.get_test_error()]
                self.print_test_results()
                self.print_test_status()
                self.conditional_save()
            
            self.print_elapsed_time(time() - compute_time_py)
    
    def cleanup(self):
        if self.checkpoint_writer is not None:
            self.checkpoint_writer.join()
            self.checkpoint_writer = None
        
    def print_model_state(self):
        pass
    
    def get_num_batches_done(self):
        return len(self.train_batch_range) * (self.epoch - 1) + self.batchnum - self.train_batch_range[0] + 1
    
    def get_next_batch(self, train=True):
        dp = self.train_data_provider
        if not train:
            dp = self.test_data_provider
        return self.parse_batch_data(dp.get_next_batch(), train=train)
    
    def parse_batch_data(self, batch_data, train=True):
        return batch_data[0], batch_data[1], batch_data[2]['data']
    
    def start_batch(self, batch_data, train=True):
        self.libmodel.startBatch(batch_data[2], not train)
    
    def finish_batch(self):
        return self.libmodel.finishBatch()
    
    def print_iteration(self):
        print "\t%d.%d..." % (self.epoch, self.batchnum),
    
    def print_elapsed_time(self, compute_time_py):
        print "(%.3f sec)" % (compute_time_py)
    
    def print_train_results(self):
        batch_error = self.train_outputs[-1][0]
        if not (batch_error > 0 and batch_error < 2e20):
            print "Crazy train error: %.6f" % batch_error
            self.cleanup()

        print "Train error: %.6f " % (batch_error),

    def print_test_results(self):
        batch_error = self.test_outputs[-1][0]
        print "%s\t\tTest error: %.6f" % (NL, batch_error),

    def print_test_status(self):
        status = (len(self.test_outputs) == 1 or self.test_outputs[-1][0] < self.test_outputs[-2][0]) and "ok" or "WORSE"
        print status,
        
    def sync_with_host(self):
        if self.checkpoint_writer is not None:
            self.checkpoint_writer.join()
            self.checkpoint_writer = None
        self.libmodel.syncWithHost()
        
    def conditional_save(self):
        batch_error = self.test_outputs[-1][0]
        if batch_error > 0 and batch_error < self.max_test_err:
            self.save_state()
        else:
            print "\tTest error > %g, not saving." % self.max_test_err,
    
    def aggregate_test_outputs(self, test_outputs):
        test_error = tuple([sum(t[r] for t in test_outputs) / (1 if self.test_one else len(self.test_batch_range)) for r in range(len(test_outputs[-1]))])
        return test_error
    
    def get_test_error(self):
        next_data = self.get_next_batch(train=False)
        test_outputs = []
        while True:
            data = next_data
            start_time_test = time()
            self.start_batch(data, train=False)
            load_next = (not self.test_one or self.test_only) and data[1] < self.test_batch_range[-1]
            if load_next: # load next batch
                next_data = self.get_next_batch(train=False)
            test_outputs += [self.finish_batch()]
            if self.test_only: # Print the individual batch results for safety
                print "batch %d: %s" % (data[1], str(test_outputs[-1])),
                self.print_elapsed_time(time() - start_time_test)
            if not load_next:
                break
            sys.stdout.flush()
            
        return self.aggregate_test_outputs(test_outputs)
    
    def set_var(self, var_name, var_val):
        setattr(self, var_name, var_val)
        self.model_state[var_name] = var_val
        return var_val
        
    def get_var(self, var_name):
        return self.model_state[var_name]
        
    def has_var(self, var_name):
        return var_name in self.model_state
        
    def save_state(self):
        for att in self.model_state:
            if hasattr(self, att):
                self.model_state[att] = getattr(self, att)
        
        dic = {"model_state": self.model_state,
               "op": self.op}
            
        checkpoint_file = "%d.%d" % (self.epoch, self.batchnum)
        checkpoint_file_full_path = os.path.join(self.save_file, checkpoint_file)
        if not os.path.exists(self.save_file):
            os.makedirs(self.save_file)
    
        assert self.checkpoint_writer is None
        self.checkpoint_writer = CheckpointWriter(checkpoint_file_full_path, dic)
        self.checkpoint_writer.start()
        print "-------------------------------------------------------"
        print "Saved checkpoint to %s" % self.save_file
        print "=======================================================",
        return self.checkpoint_writer
        
    def get_progress(self):
        num_batches_total = self.num_epochs * len(self.train_batch_range)
        return min(1.0, max(0.0, float(self.get_num_batches_done()-1) / num_batches_total))
    
    @staticmethod
    def load_checkpoint(load_dir):
        if os.path.isdir(load_dir):
            return unpickle(os.path.join(load_dir, sorted(os.listdir(load_dir), key=alphanum_key)[-1]))
        return unpickle(load_dir)

    @staticmethod
    def get_options_parser():
        op = OptionsParser()
        op.add_option("load-file", "load_file", StringOptionParser, "Load file", default="", excuses=OptionsParser.EXCUSE_ALL)
        op.add_option("save-path", "save_path", StringOptionParser, "Save path", excuses=['save_file_override'])
        op.add_option("save-file", "save_file_override", StringOptionParser, "Save file override", excuses=['save_path'])
        op.add_option("train-range", "train_batch_range", RangeOptionParser, "Data batch range: training")
        op.add_option("test-range", "test_batch_range", RangeOptionParser, "Data batch range: testing")
        op.add_option("data-provider", "dp_type", StringOptionParser, "Data provider", default="default")
        op.add_option("test-freq", "testing_freq", IntegerOptionParser, "Testing frequency", default=25)
        op.add_option("epochs", "num_epochs", IntegerOptionParser, "Number of epochs", default=500)
        op.add_option("data-path", "data_path", StringOptionParser, "Data path")
        
        op.add_option("max-test-err", "max_test_err", FloatOptionParser, "Maximum test error for saving")
        op.add_option("test-only", "test_only", BooleanOptionParser, "Test and quit?", default=0)
        op.add_option("test-one", "test_one", BooleanOptionParser, "Test on one batch at a time?", default=1)
        op.add_option("force-save", "force_save", BooleanOptionParser, "Force save before quitting", default=0)
        op.add_option("gpu", "gpu", ListOptionParser(IntegerOptionParser), "GPU override")
        return op

    @staticmethod
    def print_data_providers():
        print "Available data providers:"
        for dp, desc in dp_types.iteritems():
            print "    %s: %s" % (dp, desc)
            

    @staticmethod
    def parse_options(op):
        try:
            load_dic = None
            options = op.parse()
            load_location = None
#            print options['load_file'].value_given, options['save_file_override'].value_given
#            print options['save_file_override'].value
            if options['load_file'].value_given:
                load_location = options['load_file'].value
            elif options['save_file_override'].value_given and os.path.exists(options['save_file_override'].value):
                load_location = options['save_file_override'].value
            
            if load_location is not None:
                load_dic = IGPUModel.load_checkpoint(load_location)
                old_op = load_dic["op"]
                old_op.merge_from(op)
                op = old_op
            op.eval_expr_defaults()
            return op, load_dic
        except OptionMissingException, e:
            print e
            op.print_usage()
        except OptionException, e:
            print e
        except UnpickleError, e:
            print "Error loading checkpoint:"
            print e
        sys.exit()
