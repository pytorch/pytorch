## @package db_input
# Module caffe2.python.helpers.db_input





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
