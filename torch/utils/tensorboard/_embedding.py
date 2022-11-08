import math
import os
import numpy as np
from ._convert_np import make_np
from ._utils import make_grid
from tensorboard.compat import tf
from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo


def make_tsv(metadata, save_path, metadata_header=None):
    if not metadata_header:
        metadata = [str(x) for x in metadata]
    else:
        assert len(metadata_header) == len(
            metadata[0]
        ), "len of header must be equal to the number of columns in metadata"
        metadata = ["\t".join(str(e) for e in l) for l in [metadata_header] + metadata]

    metadata_bytes = tf.compat.as_bytes("\n".join(metadata) + "\n")
    
    with open(os.path.join(save_path, "metadata.tsv"), "wb") as f:
        f.write(tf.compat.as_bytes(metadata_bytes))
        

# https://github.com/tensorflow/tensorboard/issues/44 image label will be squared
def make_sprite(label_img, save_path):
    from PIL import Image
    from io import BytesIO

    # this ensures the sprite image has correct dimension as described in
    # https://www.tensorflow.org/get_started/embedding_viz
    nrow = int(math.ceil((label_img.size(0)) ** 0.5))
    arranged_img_CHW = make_grid(make_np(label_img), ncols=nrow)

    # augment images so that #images equals nrow*nrow
    arranged_augment_square_HWC = np.zeros(
        (arranged_img_CHW.shape[2], arranged_img_CHW.shape[2], 3)
    )
    arranged_img_HWC = arranged_img_CHW.transpose(1, 2, 0)  # chw -> hwc
    arranged_augment_square_HWC[: arranged_img_HWC.shape[0], :, :] = arranged_img_HWC
    im = Image.fromarray(np.uint8((arranged_augment_square_HWC * 255).clip(0, 255)))

    with BytesIO() as buf:
        im.save(buf, format="PNG")
        im_bytes = buf.getvalue()

    with open(os.path.join(save_path, "sprite.png"), "wb") as f:
        f.write(tf.compat.as_bytes(im_bytes))


def get_embedding_info(metadata, label_img, filesys, subdir, global_step, tag):
    info = EmbeddingInfo()
    info.tensor_name = "{}:{}".format(tag, str(global_step).zfill(5))
    info.tensor_path = os.path.join(subdir, "tensors.tsv")
    if metadata is not None:
        info.metadata_path = os.path.join(subdir, "metadata.tsv")
    if label_img is not None:
        info.sprite.image_path = os.path.join(subdir, "sprite.png")
        info.sprite.single_image_dim.extend([label_img.size(3), label_img.size(2)])
    return info


def write_pbtxt(save_path, contents):
    with open(os.path.join(save_path, "projector_config.pbtxt"), "wb") as f:
        f.write(tf.compat.as_bytes(contents))


def make_mat(matlist, save_path):
    with open(os.path.join(save_path, "tensors.tsv"), "wb") as f:
        for x in matlist:
            x = [str(i.item()) for i in x]
            f.write(tf.compat.as_bytes("\t".join(x) + "\n"))
