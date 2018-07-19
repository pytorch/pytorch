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

import os
import sys
from tarfile import TarFile, TarInfo
from matplotlib import pylab as pl
import numpy as n
import getopt as opt
from python_util.util import *
from math import sqrt, ceil, floor
from python_util.gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from python_util.options import *
from PIL import Image
from time import sleep

class ShowNetError(Exception):
    pass

class ShowConvNet(ConvNet):
    def __init__(self, op, load_dic):
        ConvNet.__init__(self, op, load_dic)

    def init_data_providers(self):
        self.need_gpu = self.op.get_value('show_preds') 
        class Dummy:
            def advance_batch(self):
                pass
        if self.need_gpu:
            ConvNet.init_data_providers(self)
        else:
            self.train_data_provider = self.test_data_provider = Dummy()
    
    def import_model(self):
        if self.need_gpu:
            ConvNet.import_model(self)
            
    def init_model_state(self):
        if self.op.get_value('show_preds'):
            self.softmax_name = self.op.get_value('show_preds')
            
    def init_model_lib(self):
        if self.need_gpu:
            ConvNet.init_model_lib(self)

    def plot_cost(self):
        if self.show_cost not in self.train_outputs[0][0]:
            raise ShowNetError("Cost function with name '%s' not defined by given convnet." % self.show_cost)
#        print self.test_outputs
        train_errors = [eval(self.layers[self.show_cost]['outputFilter'])(o[0][self.show_cost], o[1])[self.cost_idx] for o in self.train_outputs]
        test_errors = [eval(self.layers[self.show_cost]['outputFilter'])(o[0][self.show_cost], o[1])[self.cost_idx] for o in self.test_outputs]
        if self.smooth_test_errors:
            test_errors = [sum(test_errors[max(0,i-len(self.test_batch_range)):i])/(i-max(0,i-len(self.test_batch_range))) for i in xrange(1,len(test_errors)+1)]
        numbatches = len(self.train_batch_range)
        test_errors = n.row_stack(test_errors)
        test_errors = n.tile(test_errors, (1, self.testing_freq))
        test_errors = list(test_errors.flatten())
        test_errors += [test_errors[-1]] * max(0,len(train_errors) - len(test_errors))
        test_errors = test_errors[:len(train_errors)]

        numepochs = len(train_errors) / float(numbatches)
        pl.figure(1)
        x = range(0, len(train_errors))
        pl.plot(x, train_errors, 'k-', label='Training set')
        pl.plot(x, test_errors, 'r-', label='Test set')
        pl.legend()
        ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
        epoch_label_gran = int(ceil(numepochs / 20.)) 
        epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) if numepochs >= 10 else epoch_label_gran 
        ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))

        pl.xticks(ticklocs, ticklabels)
        pl.xlabel('Epoch')
#        pl.ylabel(self.show_cost)
        pl.title('%s[%d]' % (self.show_cost, self.cost_idx))
#        print "plotted cost"
        
    def make_filter_fig(self, filters, filter_start, fignum, _title, num_filters, combine_chans, FILTERS_PER_ROW=16):
        MAX_ROWS = 24
        MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
        num_colors = filters.shape[0]
        f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
        filter_end = min(filter_start+MAX_FILTERS, num_filters)
        filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))
    
        filter_pixels = filters.shape[1]
        filter_size = int(sqrt(filters.shape[1]))
        fig = pl.figure(fignum)
        fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
        num_filters = filter_end - filter_start
        if not combine_chans:
            bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size*num_colors * f_per_row + f_per_row + 1), dtype=n.single)
        else:
            bigpic = n.zeros((3, filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)
    
        for m in xrange(filter_start,filter_end ):
            filter = filters[:,:,m]
            y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
            if not combine_chans:
                for c in xrange(num_colors):
                    filter_pic = filter[c,:].reshape((filter_size,filter_size))
                    bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                           1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
            else:
                filter_pic = filter.reshape((3, filter_size,filter_size))
                bigpic[:,
                       1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                       1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
                
        pl.xticks([])
        pl.yticks([])
        if not combine_chans:
            pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
        else:
            bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
            pl.imshow(bigpic, interpolation='nearest')        
        
    def plot_filters(self):
        FILTERS_PER_ROW = 16
        filter_start = 0 # First filter to show
        if self.show_filters not in self.layers:
            raise ShowNetError("Layer with name '%s' not defined by given convnet." % self.show_filters)
        layer = self.layers[self.show_filters]
        filters = layer['weights'][self.input_idx]
#        filters = filters - filters.min()
#        filters = filters / filters.max()
        if layer['type'] == 'fc': # Fully-connected layer
            num_filters = layer['outputs']
            channels = self.channels
            filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
        elif layer['type'] in ('conv', 'local'): # Conv layer
            num_filters = layer['filters']
            channels = layer['filterChannels'][self.input_idx]
            if layer['type'] == 'local':
                filters = filters.reshape((layer['modules'], channels, layer['filterPixels'][self.input_idx], num_filters))
                filters = filters[:, :, :, self.local_plane] # first map for now (modules, channels, pixels)
                filters = filters.swapaxes(0,2).swapaxes(0,1)
                num_filters = layer['modules']
#                filters = filters.swapaxes(0,1).reshape(channels * layer['filterPixels'][self.input_idx], num_filters * layer['modules'])
#                num_filters *= layer['modules']
                FILTERS_PER_ROW = layer['modulesX']
            else:
                filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
        
        
        # Convert YUV filters to RGB
        if self.yuv_to_rgb and channels == 3:
            R = filters[0,:,:] + 1.28033 * filters[2,:,:]
            G = filters[0,:,:] + -0.21482 * filters[1,:,:] + -0.38059 * filters[2,:,:]
            B = filters[0,:,:] + 2.12798 * filters[1,:,:]
            filters[0,:,:], filters[1,:,:], filters[2,:,:] = R, G, B
        combine_chans = not self.no_rgb and channels == 3
        
        # Make sure you don't modify the backing array itself here -- so no -= or /=
        if self.norm_filters:
            #print filters.shape
            filters = filters - n.tile(filters.reshape((filters.shape[0] * filters.shape[1], filters.shape[2])).mean(axis=0).reshape(1, 1, filters.shape[2]), (filters.shape[0], filters.shape[1], 1))
            filters = filters / n.sqrt(n.tile(filters.reshape((filters.shape[0] * filters.shape[1], filters.shape[2])).var(axis=0).reshape(1, 1, filters.shape[2]), (filters.shape[0], filters.shape[1], 1)))
            #filters = filters - n.tile(filters.min(axis=0).min(axis=0), (3, filters.shape[1], 1))
            #filters = filters / n.tile(filters.max(axis=0).max(axis=0), (3, filters.shape[1], 1))
        #else:
        filters = filters - filters.min()
        filters = filters / filters.max()

        self.make_filter_fig(filters, filter_start, 2, 'Layer %s' % self.show_filters, num_filters, combine_chans, FILTERS_PER_ROW=FILTERS_PER_ROW)
    
    def plot_predictions(self):
        epoch, batch, data = self.get_next_batch(train=False) # get a test batch
        num_classes = self.test_data_provider.get_num_classes()
        NUM_ROWS = 2
        NUM_COLS = 4
        NUM_IMGS = NUM_ROWS * NUM_COLS if not self.save_preds else data[0].shape[1]
        NUM_TOP_CLASSES = min(num_classes, 5) # show this many top labels
        NUM_OUTPUTS = self.model_state['layers'][self.softmax_name]['outputs']
        PRED_IDX = 1
        
        label_names = [lab.split(',')[0] for lab in self.test_data_provider.batch_meta['label_names']]
        if self.only_errors:
            preds = n.zeros((data[0].shape[1], NUM_OUTPUTS), dtype=n.single)
        else:
            preds = n.zeros((NUM_IMGS, NUM_OUTPUTS), dtype=n.single)
            #rand_idx = nr.permutation(n.r_[n.arange(1), n.where(data[1] == 552)[1], n.where(data[1] == 795)[1], n.where(data[1] == 449)[1], n.where(data[1] == 274)[1]])[:NUM_IMGS]
            rand_idx = nr.randint(0, data[0].shape[1], NUM_IMGS)
            if NUM_IMGS < data[0].shape[1]:
                data = [n.require(d[:,rand_idx], requirements='C') for d in data]
#        data += [preds]
        # Run the model
        print  [d.shape for d in data], preds.shape
        self.libmodel.startFeatureWriter(data, [preds], [self.softmax_name])
        IGPUModel.finish_batch(self)
        print preds
        data[0] = self.test_data_provider.get_plottable_data(data[0])

        if self.save_preds:
            if not gfile.Exists(self.save_preds):
                gfile.MakeDirs(self.save_preds)
            preds_thresh = preds > 0.5 # Binarize predictions
            data[0] = data[0] * 255.0
            data[0][data[0]<0] = 0
            data[0][data[0]>255] = 255
            data[0] = n.require(data[0], dtype=n.uint8)
            dir_name = '%s_predictions_batch_%d' % (os.path.basename(self.save_file), batch)
            tar_name = os.path.join(self.save_preds, '%s.tar' % dir_name)
            tfo = gfile.GFile(tar_name, "w")
            tf = TarFile(fileobj=tfo, mode='w')
            for img_idx in xrange(NUM_IMGS):
                img = data[0][img_idx,:,:,:]
                imsave = Image.fromarray(img)
                prefix = "CORRECT" if data[1][0,img_idx] == preds_thresh[img_idx,PRED_IDX] else "FALSE_POS" if preds_thresh[img_idx,PRED_IDX] == 1 else "FALSE_NEG"
                file_name = "%s_%.2f_%d_%05d_%d.png" % (prefix, preds[img_idx,PRED_IDX], batch, img_idx, data[1][0,img_idx])
#                gf = gfile.GFile(file_name, "w")
                file_string = StringIO()
                imsave.save(file_string, "PNG")
                tarinf = TarInfo(os.path.join(dir_name, file_name))
                tarinf.size = file_string.tell()
                file_string.seek(0)
                tf.addfile(tarinf, file_string)
            tf.close()
            tfo.close()
#                gf.close()
            print "Wrote %d prediction PNGs to %s" % (preds.shape[0], tar_name)
        else:
            fig = pl.figure(3, figsize=(12,9))
            fig.text(.4, .95, '%s test samples' % ('Mistaken' if self.only_errors else 'Random'))
            if self.only_errors:
                # what the net got wrong
                if NUM_OUTPUTS > 1:
                    err_idx = [i for i,p in enumerate(preds.argmax(axis=1)) if p not in n.where(data[2][:,i] > 0)[0]]
                else:
                    err_idx = n.where(data[1][0,:] != preds[:,0].T)[0]
                    print err_idx
                err_idx = r.sample(err_idx, min(len(err_idx), NUM_IMGS))
                data[0], data[1], preds = data[0][:,err_idx], data[1][:,err_idx], preds[err_idx,:]
                
            
            import matplotlib.gridspec as gridspec
            import matplotlib.colors as colors
            cconv = colors.ColorConverter()
            gs = gridspec.GridSpec(NUM_ROWS*2, NUM_COLS,
                                   width_ratios=[1]*NUM_COLS, height_ratios=[2,1]*NUM_ROWS )
            #print data[1]
            for row in xrange(NUM_ROWS):
                for col in xrange(NUM_COLS):
                    img_idx = row * NUM_COLS + col
                    if data[0].shape[0] <= img_idx:
                        break
                    pl.subplot(gs[(row * 2) * NUM_COLS + col])
                    #pl.subplot(NUM_ROWS*2, NUM_COLS, row * 2 * NUM_COLS + col + 1)
                    pl.xticks([])
                    pl.yticks([])
                    img = data[0][img_idx,:,:,:]
                    pl.imshow(img, interpolation='lanczos')
                    show_title = data[1].shape[0] == 1
                    true_label = [int(data[1][0,img_idx])] if show_title else n.where(data[1][:,img_idx]==1)[0]
                    #print true_label
                    #print preds[img_idx,:].shape
                    #print preds[img_idx,:].max()
                    true_label_names = [label_names[i] for i in true_label]
                    img_labels = sorted(zip(preds[img_idx,:], label_names), key=lambda x: x[0])[-NUM_TOP_CLASSES:]
                    #print img_labels
                    axes = pl.subplot(gs[(row * 2 + 1) * NUM_COLS + col])
                    height = 0.5
                    ylocs = n.array(range(NUM_TOP_CLASSES))*height
                    pl.barh(ylocs, [l[0] for l in img_labels], height=height, \
                            color=['#ffaaaa' if l[1] in true_label_names else '#aaaaff' for l in img_labels])
                    #pl.title(", ".join(true_labels))
                    if show_title:
                        pl.title(", ".join(true_label_names), fontsize=15, fontweight='bold')
                    else:
                        print true_label_names
                    pl.yticks(ylocs + height/2, [l[1] for l in img_labels], x=1, backgroundcolor=cconv.to_rgba('0.65', alpha=0.5), weight='bold')
                    for line in enumerate(axes.get_yticklines()): 
                        line[1].set_visible(False) 
                    #pl.xticks([width], [''])
                    #pl.yticks([])
                    pl.xticks([])
                    pl.ylim(0, ylocs[-1] + height)
                    pl.xlim(0, 1)

    def start(self):
        self.op.print_values()
#        print self.show_cost
        if self.show_cost:
            self.plot_cost()
        if self.show_filters:
            self.plot_filters()
        if self.show_preds:
            self.plot_predictions()

        if pl:
            pl.show()
        sys.exit(0)
            
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'inner_size', 'train_batch_range', 'test_batch_range', 'multiview_test', 'data_path', 'pca_noise', 'scalar_mean'):
                op.delete_option(option)
        op.add_option("show-cost", "show_cost", StringOptionParser, "Show specified objective function", default="")
        op.add_option("show-filters", "show_filters", StringOptionParser, "Show learned filters in specified layer", default="")
        op.add_option("norm-filters", "norm_filters", BooleanOptionParser, "Individually normalize filters shown with --show-filters", default=0)
        op.add_option("input-idx", "input_idx", IntegerOptionParser, "Input index for layer given to --show-filters", default=0)
        op.add_option("cost-idx", "cost_idx", IntegerOptionParser, "Cost function return value index for --show-cost", default=0)
        op.add_option("no-rgb", "no_rgb", BooleanOptionParser, "Don't combine filter channels into RGB in layer given to --show-filters", default=False)
        op.add_option("yuv-to-rgb", "yuv_to_rgb", BooleanOptionParser, "Convert RGB filters to YUV in layer given to --show-filters", default=False)
        op.add_option("channels", "channels", IntegerOptionParser, "Number of channels in layer given to --show-filters (fully-connected layers only)", default=0)
        op.add_option("show-preds", "show_preds", StringOptionParser, "Show predictions made by given softmax on test set", default="")
        op.add_option("save-preds", "save_preds", StringOptionParser, "Save predictions to given path instead of showing them", default="")
        op.add_option("only-errors", "only_errors", BooleanOptionParser, "Show only mistaken predictions (to be used with --show-preds)", default=False, requires=['show_preds'])
        op.add_option("local-plane", "local_plane", IntegerOptionParser, "Local plane to show", default=0)
        op.add_option("smooth-test-errors", "smooth_test_errors", BooleanOptionParser, "Use running average for test error plot?", default=1)

        op.options['load_file'].default = None
        return op
    
if __name__ == "__main__":
    #nr.seed(6)
    try:
        op = ShowConvNet.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
        model = ShowConvNet(op, load_dic)
        model.start()
    except (UnpickleError, ShowNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e 
