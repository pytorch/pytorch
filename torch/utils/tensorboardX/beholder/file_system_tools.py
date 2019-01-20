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

import pickle

# import tensorflow as tf
# from google.protobuf import message


def write_file(contents, path, mode='wb'):
    with open(path, mode) as new_file:
        new_file.write(contents)


def write_pickle(obj, path):
    with open(path, 'wb') as new_file:
        pickle.dump(obj, new_file)


def read_pickle(path, default=None):
    with open(path, 'rb') as pickle_file:
        result = pickle.load(pickle_file)
    return result
