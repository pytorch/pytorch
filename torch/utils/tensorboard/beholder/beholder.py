# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ..proto.summary_pb2 import Summary
from ..proto.summary_pb2 import SummaryMetadata
from ..proto.tensor_pb2 import TensorProto
from ..proto.tensor_shape_pb2 import TensorShapeProto

import os
import time

import numpy as np
# import tensorflow as tf

# from tensorboard.plugins.beholder import im_util
# from . import im_util
from .file_system_tools import read_pickle,\
    write_pickle, write_file
from .shared_config import PLUGIN_NAME, TAG_NAME,\
    SUMMARY_FILENAME, DEFAULT_CONFIG, CONFIG_FILENAME, SUMMARY_COLLECTION_KEY_NAME, SECTION_INFO_FILENAME
from . import video_writing
# from .visualizer import Visualizer


class Beholder(object):

    def __init__(self, logdir):
        self.PLUGIN_LOGDIR = logdir + '/plugins/' + PLUGIN_NAME

        self.is_recording = False
        self.video_writer = video_writing.VideoWriter(
            self.PLUGIN_LOGDIR,
            outputs=[video_writing.FFmpegVideoOutput, video_writing.PNGVideoOutput])

        self.last_image_shape = []
        self.last_update_time = time.time()
        self.config_last_modified_time = -1
        self.previous_config = dict(DEFAULT_CONFIG)

        if not os.path.exists(self.PLUGIN_LOGDIR + '/config.pkl'):
            os.makedirs(self.PLUGIN_LOGDIR)
            write_pickle(DEFAULT_CONFIG,
                         '{}/{}'.format(self.PLUGIN_LOGDIR, CONFIG_FILENAME))

        # self.visualizer = Visualizer(self.PLUGIN_LOGDIR)
    def _get_config(self):
        '''Reads the config file from disk or creates a new one.'''
        filename = '{}/{}'.format(self.PLUGIN_LOGDIR, CONFIG_FILENAME)
        modified_time = os.path.getmtime(filename)

        if modified_time != self.config_last_modified_time:
            config = read_pickle(filename, default=self.previous_config)
            self.previous_config = config
        else:
            config = self.previous_config

        self.config_last_modified_time = modified_time
        return config

    def _write_summary(self, frame):
        '''Writes the frame to disk as a tensor summary.'''
        path = '{}/{}'.format(self.PLUGIN_LOGDIR, SUMMARY_FILENAME)
        smd = SummaryMetadata()
        tensor = TensorProto(
            dtype='DT_FLOAT',
            float_val=frame.reshape(-1).tolist(),
            tensor_shape=TensorShapeProto(
                dim=[TensorShapeProto.Dim(size=frame.shape[0]),
                     TensorShapeProto.Dim(size=frame.shape[1]),
                     TensorShapeProto.Dim(size=frame.shape[2])]
            )
        )
        summary = Summary(value=[Summary.Value(
            tag=TAG_NAME, metadata=smd, tensor=tensor)]).SerializeToString()
        write_file(summary, path)

    @staticmethod
    def stats(tensor_and_name):
        imgstats = []
        for (img, name) in tensor_and_name:
            immax = img.max()
            immin = img.min()
            imgstats.append(
                {
                    'height': img.shape[0],
                    'max': str(immax),
                    'mean': str(img.mean()),
                    'min': str(immin),
                    'name': name,
                    'range': str(immax - immin),
                    'shape': str((img.shape[1], img.shape[2]))
                })
        return imgstats

    def _get_final_image(self, config, trainable=None, arrays=None, frame=None):
        if config['values'] == 'frames':
            # print('===frames===')
            final_image = frame
        elif config['values'] == 'arrays':
            # print('===arrays===')
            final_image = np.concatenate([arr for arr, _ in arrays])
            stat = self.stats(arrays)
            write_pickle(
                stat, '{}/{}'.format(self.PLUGIN_LOGDIR, SECTION_INFO_FILENAME))
        elif config['values'] == 'trainable_variables':
            # print('===trainable===')
            final_image = np.concatenate([arr for arr, _ in trainable])
            stat = self.stats(trainable)
            write_pickle(
                stat, '{}/{}'.format(self.PLUGIN_LOGDIR, SECTION_INFO_FILENAME))
        if len(final_image.shape) == 2:  # Map grayscale images to 3D tensors.
            final_image = np.expand_dims(final_image, -1)

        return final_image

    def _enough_time_has_passed(self, FPS):
        '''For limiting how often frames are computed.'''
        if FPS == 0:
            return False
        else:
            earliest_time = self.last_update_time + (1.0 / FPS)
            return time.time() >= earliest_time

    def _update_frame(self, trainable, arrays, frame, config):
        final_image = self._get_final_image(config, trainable, arrays, frame)
        self._write_summary(final_image)
        self.last_image_shape = final_image.shape

        return final_image

    def _update_recording(self, frame, config):
        '''Adds a frame to the current video output.'''
        # pylint: disable=redefined-variable-type
        should_record = config['is_recording']

        if should_record:
            if not self.is_recording:
                self.is_recording = True
                print('Starting recording using %s',
                      self.video_writer.current_output().name())
            self.video_writer.write_frame(frame)
        elif self.is_recording:
            self.is_recording = False
            self.video_writer.finish()
            print('Finished recording')

    # TODO: blanket try and except for production? I don't someone's script to die
    #       after weeks of running because of a visualization.
    def update(self, trainable=None, arrays=None, frame=None):
        '''Creates a frame and writes it to disk.

        Args:
            trainable: a list of namedtuple (tensors, name).
            arrays: a list of namedtuple (tensors, name).
            frame: lalala
        '''

        new_config = self._get_config()
        if True or self._enough_time_has_passed(self.previous_config['FPS']):
            # self.visualizer.update(new_config)
            self.last_update_time = time.time()
            final_image = self._update_frame(
                trainable, arrays, frame, new_config)
            self._update_recording(final_image, new_config)

    ##############################################################################
    # @staticmethod
    # def gradient_helper(optimizer, loss, var_list=None):
    #   '''A helper to get the gradients out at each step.

    #   Args:
    #     optimizer: the optimizer op.
    #     loss: the op that computes your loss value.

    #   Returns: the gradient tensors and the train_step op.
    #   '''
    #   if var_list is None:
    #     var_list = tf.trainable_variables()

    #   grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    #   grads = [pair[0] for pair in grads_and_vars]

    #   return grads, optimizer.apply_gradients(grads_and_vars)


# implements pytorch backward later
class BeholderHook():
    pass
    # """SessionRunHook implementation that runs Beholder every step.

    # Convenient when using tf.train.MonitoredSession:
    # ```python
    # beholder_hook = BeholderHook(LOG_DIRECTORY)
    # with MonitoredSession(..., hooks=[beholder_hook]) as sess:
    #   sess.run(train_op)
    # ```
    # """
    # def __init__(self, logdir):
    #   """Creates new Hook instance

    #   Args:
    #     logdir: Directory where Beholder should write data.
    #   """
    #   self._logdir = logdir
    #   self.beholder = None

    # def begin(self):
    #   self.beholder = Beholder(self._logdir)

    # def after_run(self, run_context, unused_run_values):
    #   self.beholder.update(run_context.session)
