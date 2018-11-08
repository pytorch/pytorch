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

from python_util.data import *
import numpy.random as nr
import numpy as n
import random as r
from time import time
from threading import Thread
from math import sqrt
import sys
#from matplotlib import pylab as pl
from PIL import Image
from StringIO import StringIO
from time import time
import itertools as it
    
class JPEGBatchLoaderThread(Thread):
    def __init__(self, dp, batch_num, label_offset, list_out):
        Thread.__init__(self)
        self.list_out = list_out
        self.label_offset = label_offset
        self.dp = dp
        self.batch_num = batch_num
        
    @staticmethod
    def load_jpeg_batch(rawdics, dp, label_offset):
        if type(rawdics) != list:
            rawdics = [rawdics]
        nc_total = sum(len(r['data']) for r in rawdics)

        jpeg_strs = list(it.chain.from_iterable(rd['data'] for rd in rawdics))
        labels = list(it.chain.from_iterable(rd['labels'] for rd in rawdics))
        
        img_mat = n.empty((nc_total * dp.data_mult, dp.inner_pixels * dp.num_colors), dtype=n.float32)
        lab_mat = n.zeros((nc_total, dp.get_num_classes()), dtype=n.float32)
        dp.convnet.libmodel.decodeJpeg(jpeg_strs, img_mat, dp.img_size, dp.inner_size, dp.test, dp.multiview)
        lab_vec = n.tile(n.asarray([(l[nr.randint(len(l))] if len(l) > 0 else -1) + label_offset for l in labels], dtype=n.single).reshape((nc_total, 1)), (dp.data_mult,1))
        for c in xrange(nc_total):
            lab_mat[c, [z + label_offset for z in labels[c]]] = 1
        lab_mat = n.tile(lab_mat, (dp.data_mult, 1))
        

        return {'data': img_mat[:nc_total * dp.data_mult,:],
                'labvec': lab_vec[:nc_total * dp.data_mult,:],
                'labmat': lab_mat[:nc_total * dp.data_mult,:]}
    
    def run(self):
        rawdics = self.dp.get_batch(self.batch_num)
        p = JPEGBatchLoaderThread.load_jpeg_batch(rawdics,
                                                  self.dp,
                                                  self.label_offset)
        self.list_out.append(p)
        
class ColorNoiseMakerThread(Thread):
    def __init__(self, pca_stdevs, pca_vecs, num_noise, list_out):
        Thread.__init__(self)
        self.pca_stdevs, self.pca_vecs = pca_stdevs, pca_vecs
        self.num_noise = num_noise
        self.list_out = list_out
        
    def run(self):
        noise = n.dot(nr.randn(self.num_noise, 3).astype(n.single) * self.pca_stdevs.T, self.pca_vecs.T)
        self.list_out.append(noise)

class ImageDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean'].astype(n.single)
        self.color_eig = self.batch_meta['color_pca'][1].astype(n.single)
        self.color_stdevs = n.c_[self.batch_meta['color_pca'][0].astype(n.single)]
        self.color_noise_coeff = dp_params['color_noise']
        self.num_colors = 3
        self.img_size = int(sqrt(self.batch_meta['num_vis'] / self.num_colors))
        self.mini = dp_params['minibatch_size']
        self.inner_size = dp_params['inner_size'] if dp_params['inner_size'] > 0 else self.img_size
        self.inner_pixels = self.inner_size **2
        self.border_size = (self.img_size - self.inner_size) / 2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.batch_size = self.batch_meta['batch_size']
        self.label_offset = 0 if 'label_offset' not in self.batch_meta else self.batch_meta['label_offset']
        self.scalar_mean = dp_params['scalar_mean'] 
        # Maintain pointers to previously-returned data matrices so they don't get garbage collected.
        self.data = [None, None] # These are pointers to previously-returned data matrices

        self.loader_thread, self.color_noise_thread = None, None
        self.convnet = dp_params['convnet']
            
        self.num_noise = self.batch_size
        self.batches_generated, self.loaders_started = 0, 0
        self.data_mean_crop = self.data_mean.reshape((self.num_colors,self.img_size,self.img_size))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((1,3*self.inner_size**2))

        if self.scalar_mean >= 0:
            self.data_mean_crop = self.scalar_mean
            
    def showimg(self, img):
        from matplotlib import pylab as pl
        pixels = img.shape[0] / 3
        size = int(sqrt(pixels))
        img = img.reshape((3,size,size)).swapaxes(0,2).swapaxes(0,1)
        pl.imshow(img, interpolation='nearest')
        pl.show()
            
    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.inner_size**2 * 3
        if idx == 2:
            return self.get_num_classes()
        return 1

    def start_loader(self, batch_idx):
        self.load_data = []
        self.loader_thread = JPEGBatchLoaderThread(self,
                                                   self.batch_range[batch_idx],
                                                   self.label_offset,
                                                   self.load_data)
        self.loader_thread.start()
        
    def start_color_noise_maker(self):
        color_noise_list = []
        self.color_noise_thread = ColorNoiseMakerThread(self.color_stdevs, self.color_eig, self.num_noise, color_noise_list)
        self.color_noise_thread.start()
        return color_noise_list

    def set_labels(self, datadic):
        pass
    
    def get_data_from_loader(self):
        if self.loader_thread is None:
            self.start_loader(self.batch_idx)
            self.loader_thread.join()
            self.data[self.d_idx] = self.load_data[0]

            self.start_loader(self.get_next_batch_idx())
        else:
            # Set the argument to join to 0 to re-enable batch reuse
            self.loader_thread.join()
            if not self.loader_thread.is_alive():
                self.data[self.d_idx] = self.load_data[0]
                self.start_loader(self.get_next_batch_idx())
            #else:
            #    print "Re-using batch"
        self.advance_batch()
    
    def add_color_noise(self):
        # At this point the data already has 0 mean.
        # So I'm going to add noise to it, but I'm also going to scale down
        # the original data. This is so that the overall scale of the training
        # data doesn't become too different from the test data.

        s = self.data[self.d_idx]['data'].shape
        cropped_size = self.get_data_dims(0) / 3
        ncases = s[0]

        if self.color_noise_thread is None:
            self.color_noise_list = self.start_color_noise_maker()
            self.color_noise_thread.join()
            self.color_noise = self.color_noise_list[0]
            self.color_noise_list = self.start_color_noise_maker()
        else:
            self.color_noise_thread.join(0)
            if not self.color_noise_thread.is_alive():
                self.color_noise = self.color_noise_list[0]
                self.color_noise_list = self.start_color_noise_maker()

        self.data[self.d_idx]['data'] = self.data[self.d_idx]['data'].reshape((ncases*3, cropped_size))
        self.color_noise = self.color_noise[:ncases,:].reshape((3*ncases, 1))
        self.data[self.d_idx]['data'] += self.color_noise * self.color_noise_coeff
        self.data[self.d_idx]['data'] = self.data[self.d_idx]['data'].reshape((ncases, 3* cropped_size))
        self.data[self.d_idx]['data'] *= 1.0 / (1.0 + self.color_noise_coeff) # <--- NOTE: This is the slow line, 0.25sec. Down from 0.75sec when I used division.
    
    def get_next_batch(self):
        self.d_idx = self.batches_generated % 2
        epoch, batchnum = self.curr_epoch, self.curr_batchnum

        self.get_data_from_loader()

        # Subtract mean
        self.data[self.d_idx]['data'] -= self.data_mean_crop
        
        if self.color_noise_coeff > 0 and not self.test:
            self.add_color_noise()
        self.batches_generated += 1
        
        return epoch, batchnum, [self.data[self.d_idx]['data'].T, self.data[self.d_idx]['labvec'].T, self.data[self.d_idx]['labmat'].T]
        
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data, add_mean=True):
        mean = self.data_mean_crop.reshape((data.shape[0],1)) if data.flags.f_contiguous or self.scalar_mean else self.data_mean_crop.reshape((data.shape[0],1))
        return n.require((data + (mean if add_mean else 0)).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
       
class CIFARDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.img_size = 32 
        self.num_colors = 3
        self.inner_size =  dp_params['inner_size'] if dp_params['inner_size'] > 0 else self.batch_meta['img_size']
        self.border_size = (self.img_size - self.inner_size) / 2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 9
        self.scalar_mean = dp_params['scalar_mean'] 
        self.data_mult = self.num_views if self.multiview else 1
        self.data_dic = []
        for i in batch_range:
            self.data_dic += [unpickle(self.get_data_file_name(i))]
            self.data_dic[-1]["labels"] = n.require(self.data_dic[-1]['labels'], dtype=n.single)
            self.data_dic[-1]["labels"] = n.require(n.tile(self.data_dic[-1]["labels"].reshape((1, n.prod(self.data_dic[-1]["labels"].shape))), (1, self.data_mult)), requirements='C')
            self.data_dic[-1]['data'] = n.require(self.data_dic[-1]['data'] - self.scalar_mean, dtype=n.single, requirements='C')
        
        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((self.num_colors,self.img_size,self.img_size))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        bidx = batchnum - self.batch_range[0]

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(self.data_dic[bidx]['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, self.data_dic[bidx]['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * self.num_colors if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(self.num_colors, self.img_size, self.img_size, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0), (0, self.border_size), (0, self.border_size*2),
                                  (self.border_size, 0), (self.border_size, self.border_size), (self.border_size, self.border_size*2),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views):
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))

class DummyConvNetLogRegDataProvider(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim)

        self.img_size = int(sqrt(data_dim/3))
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        dic = {'data': dic[0], 'labels': dic[1]}
        print dic['data'].shape, dic['labels'].shape
        return epoch, batchnum, [dic['data'], dic['labels']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1
