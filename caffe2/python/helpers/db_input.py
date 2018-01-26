# Copyright (c) 2016-present, Facebook, Inc.
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
##############################################################################

## @package db_input
# Module caffe2.python.helpers.db_input
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def db_input(model, blobs_out, batch_size, db, db_type):
    dbreader_name = "dbreader_" + db
    dbreader = model.param_init_net.CreateDB(
        [],
        dbreader_name,
        db=db,
        db_type=db_type,
    )
    return model.net.TensorProtosDBInput(
        dbreader, blobs_out, batch_size=batch_size)
