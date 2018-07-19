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
