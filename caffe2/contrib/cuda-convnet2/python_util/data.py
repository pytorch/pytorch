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
from numpy.random import randn, rand, random_integers
import os
from threading import Thread
from util import *

BATCH_META_FILE = "batches.meta"

class DataLoaderThread(Thread):
    def __init__(self, path, tgt):
        Thread.__init__(self)
        self.path = path
        self.tgt = tgt
    def run(self):
        self.tgt += [unpickle(self.path)]
        
class DataProvider:
    BATCH_REGEX = re.compile('^data_batch_(\d+)(\.\d+)?$')
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        if batch_range == None:
            batch_range = DataProvider.get_batch_nums(data_dir)
        if init_batchnum is None or init_batchnum not in batch_range:
            init_batchnum = batch_range[0]

        self.data_dir = data_dir
        self.batch_range = batch_range
        self.curr_epoch = init_epoch
        self.curr_batchnum = init_batchnum
        self.dp_params = dp_params
        self.batch_meta = self.get_batch_meta(data_dir)
        self.data_dic = None
        self.test = test
        self.batch_idx = batch_range.index(init_batchnum)

    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()

        return epoch, batchnum, self.data_dic
            
    def get_batch(self, batch_num):
        fname = self.get_data_file_name(batch_num)
        if os.path.isdir(fname): # batch in sub-batches
            sub_batches = sorted(os.listdir(fname), key=alphanum_key)
            #print sub_batches
            num_sub_batches = len(sub_batches)
            tgts = [[] for i in xrange(num_sub_batches)]
            threads = [DataLoaderThread(os.path.join(fname, s), tgt) for (s, tgt) in zip(sub_batches, tgts)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            return [t[0] for t in tgts]
        return unpickle(self.get_data_file_name(batch_num))
    
    def get_data_dims(self,idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1
    
    def advance_batch(self):
        self.batch_idx = self.get_next_batch_idx()
        self.curr_batchnum = self.batch_range[self.batch_idx]
        if self.batch_idx == 0: # we wrapped
            self.curr_epoch += 1
            
    def get_next_batch_idx(self):
        return (self.batch_idx + 1) % len(self.batch_range)
    
    def get_next_batch_num(self):
        return self.batch_range[self.get_next_batch_idx()]
    
    # get filename of current batch
    def get_data_file_name(self, batchnum=None):
        if batchnum is None:
            batchnum = self.curr_batchnum
        return os.path.join(self.data_dir, 'data_batch_%d' % batchnum)
    
    @classmethod
    def get_instance(cls, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, type="default", dp_params={}, test=False):
        # why the fuck can't i reference DataProvider in the original definition?
        #cls.dp_classes['default'] = DataProvider
        type = type or DataProvider.get_batch_meta(data_dir)['dp_type'] # allow data to decide data provider
        if type.startswith("dummy-"):
            name = "-".join(type.split('-')[:-1]) + "-n"
            if name not in dp_types:
                raise DataProviderException("No such data provider: %s" % type)
            _class = dp_classes[name]
            dims = int(type.split('-')[-1])
            return _class(dims)
        elif type in dp_types:
            _class = dp_classes[type]
            return _class(data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        
        raise DataProviderException("No such data provider: %s" % type)
    
    @classmethod
    def register_data_provider(cls, name, desc, _class):
        if name in dp_types:
            raise DataProviderException("Data provider %s already registered" % name)
        dp_types[name] = desc
        dp_classes[name] = _class
        
    @staticmethod
    def get_batch_meta(data_dir):
        return unpickle(os.path.join(data_dir, BATCH_META_FILE))
    
    @staticmethod
    def get_batch_filenames(srcdir):
        return sorted([f for f in os.listdir(srcdir) if DataProvider.BATCH_REGEX.match(f)], key=alphanum_key)
    
    @staticmethod
    def get_batch_nums(srcdir):
        names = DataProvider.get_batch_filenames(srcdir)
        return sorted(list(set(int(DataProvider.BATCH_REGEX.match(n).group(1)) for n in names)))
        
    @staticmethod
    def get_num_batches(srcdir):
        return len(DataProvider.get_batch_nums(srcdir))
    
class DummyDataProvider(DataProvider):
    def __init__(self, data_dim):
        #self.data_dim = data_dim
        self.batch_range = [1]
        self.batch_meta = {'num_vis': data_dim, 'data_in_rows':True}
        self.curr_epoch = 1
        self.curr_batchnum = 1
        self.batch_idx = 0
        
    def get_next_batch(self):
        epoch,  batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        data = rand(512, self.get_data_dims()).astype(n.single)
        return self.curr_epoch, self.curr_batchnum, {'data':data}

class LabeledDataProvider(DataProvider):   
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        
    def get_num_classes(self):
        return len(self.batch_meta['label_names'])
        
class LabeledDummyDataProvider(DummyDataProvider):
    def __init__(self, data_dim, num_classes=10, num_cases=7):
        #self.data_dim = data_dim
        self.batch_range = [1]
        self.batch_meta = {'num_vis': data_dim,
                           'label_names': [str(x) for x in range(num_classes)],
                           'data_in_rows':True}
        self.num_cases = num_cases
        self.num_classes = num_classes
        self.curr_epoch = 1
        self.curr_batchnum = 1
        self.batch_idx=0
        self.data = None
        
    def get_num_classes(self):
        return self.num_classes
    
    def get_next_batch(self):
        epoch,  batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        if self.data is None:
            data = rand(self.num_cases, self.get_data_dims()).astype(n.single) # <--changed to rand
            labels = n.require(n.c_[random_integers(0,self.num_classes-1,self.num_cases)], requirements='C', dtype=n.single)
            self.data, self.labels = data, labels
        else:
            data, labels = self.data, self.labels
#        print data.shape, labels.shape
        return self.curr_epoch, self.curr_batchnum, [data.T, labels.T ]

    
dp_types = {"dummy-n": "Dummy data provider for n-dimensional data",
            "dummy-labeled-n": "Labeled dummy data provider for n-dimensional data"}
dp_classes = {"dummy-n": DummyDataProvider,
              "dummy-labeled-n": LabeledDummyDataProvider}
    
class DataProviderException(Exception):
    pass
