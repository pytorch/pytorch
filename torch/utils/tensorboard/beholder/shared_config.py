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

PLUGIN_NAME = 'beholder'
TAG_NAME = 'beholder-frame'
SUMMARY_FILENAME = 'frame.summary'
CONFIG_FILENAME = 'config.pkl'
SECTION_INFO_FILENAME = 'section-info.pkl'
SUMMARY_COLLECTION_KEY_NAME = 'summaries_beholder'

DEFAULT_CONFIG = {
    'values': 'trainable_variables',
    'mode': 'variance',
    'scaling': 'layer',
    'window_size': 15,
    'FPS': 10,
    'is_recording': False,
    'show_all': False,
    'colormap': 'magma'
}

SECTION_HEIGHT = 128
IMAGE_WIDTH = 512 + 256

TB_WHITE = 245
