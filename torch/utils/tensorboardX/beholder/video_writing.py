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

import abc
import os
import subprocess
import time

import numpy as np


class VideoWriter(object):
    """Video file writer that can use different output types.

    Each VideoWriter instance writes video files to a specified directory, using
    the first available VideoOutput from the provided list.
    """

    def __init__(self, directory, outputs):
        self.directory = directory
        # Filter to the available outputs
        self.outputs = [out for out in outputs if out.available()]
        if not self.outputs:
            raise IOError('No available video outputs')
        self.output_index = 0
        self.output = None
        self.frame_shape = None

    def current_output(self):
        return self.outputs[self.output_index]

    def write_frame(self, np_array):
        # Reset whenever we encounter a new frame shape.
        if self.frame_shape != np_array.shape:
            if self.output:
                self.output.close()
            self.output = None
            self.frame_shape = np_array.shape
            print('Starting video with frame shape: %s', self.frame_shape)
        # Write the frame, advancing across output types as necessary.
        original_output_index = self.output_index
        for self.output_index in range(original_output_index, len(self.outputs)):
            try:
                if not self.output:
                    new_output = self.outputs[self.output_index]
                    if self.output_index > original_output_index:
                        print('Falling back to video output %s',
                              new_output.name())
                    self.output = new_output(self.directory, self.frame_shape)
                self.output.emit_frame(np_array)
                return
            except (IOError, OSError) as e:
                print('Video output type %s not available: %s',
                      self.current_output().name(), str(e))
                if self.output:
                    self.output.close()
                self.output = None
        raise IOError('Exhausted available video outputs')

    def finish(self):
        if self.output:
            self.output.close()
        self.output = None
        self.frame_shape = None
        # Reconsider failed outputs when video is manually restarted.
        self.output_index = 0


class VideoOutput(object):
    """Base class for video outputs supported by VideoWriter."""

    __metaclass__ = abc.ABCMeta

    # Would add @abc.abstractmethod in python 3.3+
    @classmethod
    def available(cls):
        raise NotImplementedError()

    @classmethod
    def name(cls):
        return cls.__name__

    @abc.abstractmethod
    def emit_frame(self, np_array):
        raise NotImplementedError()

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError()


class PNGVideoOutput(VideoOutput):
    """Video output implemented by writing individual PNGs to disk."""

    @classmethod
    def available(cls):
        return True

    def __init__(self, directory, frame_shape):
        del frame_shape  # unused
        self.directory = directory + '/video-frames-{}'.format(time.time())
        self.frame_num = 0
        os.makedirs(self.directory)

    def emit_frame(self, np_array):
        filename = self.directory + '/{:05}.png'.format(self.frame_num)
        self._write_image(np_array.astype(np.uint8), filename)
        self.frame_num += 1

    def _write_image(self, im, filename):
        from PIL import Image
        Image.fromarray(im).save(filename)

    def close(self):
        pass


class FFmpegVideoOutput(VideoOutput):
    """Video output implemented by streaming to FFmpeg with .mp4 output."""

    @classmethod
    def available(cls):
        # Silently check if ffmpeg is available.
        try:
            with open(os.devnull, 'wb') as devnull:
                subprocess.check_call(
                    ['ffmpeg', '-version'], stdout=devnull, stderr=devnull)
            return True
        except (OSError, subprocess.CalledProcessError):
            return False

    def __init__(self, directory, frame_shape):
        self.filename = directory + '/video-{}.webm'.format(time.time())
        if len(frame_shape) != 3:
            raise ValueError(
                'Expected rank-3 array for frame, got %s' % str(frame_shape))
        # Set input pixel format based on channel count.
        if frame_shape[2] == 1:
            pix_fmt = 'gray'
        elif frame_shape[2] == 3:
            pix_fmt = 'rgb24'
        else:
            raise ValueError('Unsupported channel count %d' % frame_shape[2])

        command = [
            'ffmpeg',
            '-y',  # Overwite output
            # Input options - raw video file format and codec.
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            # Width x height.
            '-s', '%dx%d' % (frame_shape[1], frame_shape[0]),
            '-pix_fmt', pix_fmt,
            '-r', '15',  # Frame rate: arbitrarily use 15 frames per second.
            '-i', '-',  # Use stdin.
            '-an',  # No audio.
            # Output options - use lossless VP9 codec inside .webm.
            '-vcodec', 'libvpx-vp9',
            '-lossless', '1',
            # Using YUV is most compatible, though conversion from RGB skews colors.
            '-pix_fmt', 'yuv420p',
            self.filename
        ]
        PIPE = subprocess.PIPE
        self.ffmpeg = subprocess.Popen(
            command, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    def _handle_error(self):
        _, stderr = self.ffmpeg.communicate()
        bar = '=' * 40
        print('Error writing to FFmpeg:\n{}\n{}\n{}',
              bar, stderr, bar)

    def emit_frame(self, np_array):
        try:
            self.ffmpeg.stdin.write(np_array.tobytes())
            self.ffmpeg.stdin.flush()
        except IOError:
            self._handle_error()
            raise IOError('Failure invoking FFmpeg')

    def close(self):
        if self.ffmpeg.poll() is None:
            # Close stdin and consume and discard stderr/stdout.
            self.ffmpeg.communicate()
        self.ffmpeg = None
