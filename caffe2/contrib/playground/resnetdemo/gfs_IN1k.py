from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# # example1 using gfs as input source.

def gen_input_builder_fun(self, model, dataset, is_train):
    if is_train:
        input_path = self.opts['input']['train_input_path']
    else:
        input_path = self.opts['input']['test_input_path']

    reader = model.CreateDB("reader",
                            db=input_path,
                            db_type='lmdb',
                            shard_id=self.shard_id,
                            num_shards=self.opts['distributed']['num_shards'],)

    def AddImageInput(model, reader, batch_size, img_size):
        '''
        Image input operator that loads data from reader and
        applies certain transformations to the images.
        '''
        data, label = model.ImageInput(
            reader,
            ["data", "label"],
            batch_size=batch_size,
            use_caffe_datum=True,
            mean=128.,
            std=128.,
            scale=256,
            crop=img_size,
            mirror=1,
            is_test=True
        )
        data = model.StopGradient(data, data)

    def add_image_input(model):
        AddImageInput(
            model,
            reader,
            batch_size=self.opts['epoch_iter']['batch_per_device'],
            img_size=self.opts['input']['imsize'],
        )
    return add_image_input


def get_input_dataset(opts):
    return []


def get_model_input_fun(self):
    pass
