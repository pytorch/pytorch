from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import caffe2.python._import_c_extension as C

CAFFE2_NO_OPERATOR_SCHEMA = C.define_caffe2_no_operator_schema
build_options = C.get_build_options()
