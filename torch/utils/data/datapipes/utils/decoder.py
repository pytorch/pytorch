# This file takes partial of the implementation from NVIDIA's webdataset at here:
# https://github.com/tmbdev/webdataset/blob/master/webdataset/autodecode.py

import io
import json
import os.path
import pickle
import tempfile

import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper


__all__ = [
    "Decoder",
    "ImageHandler",
    "MatHandler",
    "audiohandler",
    "basichandlers",
    "extension_extract_fn",
    "handle_extension",
    "imagehandler",
    "mathandler",
    "videohandler",
]


################################################################
# handle basic datatypes
################################################################
def basichandlers(extension, data):

    if extension in "txt text transcript":
        return data.decode("utf-8")

    if extension in "cls cls2 class count index inx id".split():
        try:
            return int(data)
        except ValueError:
            return None

    if extension in "json jsn":
        return json.loads(data)

    if extension in "pyd pickle".split():
        return pickle.loads(data)

    if extension in "pt".split():
        stream = io.BytesIO(data)
        return torch.load(stream)

    # if extension in "ten tb".split():
    #     from . import tenbin
    #     return tenbin.decode_buffer(data)

    # if extension in "mp msgpack msg".split():
    #     import msgpack
    #     return msgpack.unpackb(data)

    return None


################################################################
# handle images
################################################################
imagespecs = {
    "l8": ("numpy", "uint8", "l"),
    "rgb8": ("numpy", "uint8", "rgb"),
    "rgba8": ("numpy", "uint8", "rgba"),
    "l": ("numpy", "float", "l"),
    "rgb": ("numpy", "float", "rgb"),
    "rgba": ("numpy", "float", "rgba"),
    "torchl8": ("torch", "uint8", "l"),
    "torchrgb8": ("torch", "uint8", "rgb"),
    "torchrgba8": ("torch", "uint8", "rgba"),
    "torchl": ("torch", "float", "l"),
    "torchrgb": ("torch", "float", "rgb"),
    "torch": ("torch", "float", "rgb"),
    "torchrgba": ("torch", "float", "rgba"),
    "pill": ("pil", None, "l"),
    "pil": ("pil", None, "rgb"),
    "pilrgb": ("pil", None, "rgb"),
    "pilrgba": ("pil", None, "rgba"),
}

def handle_extension(extensions, f):
    """
    Returns a decoder handler function for the list of extensions.
    Extensions can be a space separated list of extensions.
    Extensions can contain dots, in which case the corresponding number
    of extension components must be present in the key given to f.
    Comparisons are case insensitive.
    Examples:
    handle_extension("jpg jpeg", my_decode_jpg)  # invoked for any file.jpg
    handle_extension("seg.jpg", special_case_jpg)  # invoked only for file.seg.jpg
    """

    extensions = extensions.lower().split()

    def g(key, data):
        extension = key.lower().split(".")

        for target in extensions:
            target = target.split(".")
            if len(target) > len(extension):
                continue

            if extension[-len(target):] == target:
                return f(data)
            return None
    return g


class ImageHandler:
    """
    Decode image data using the given `imagespec`.
    The `imagespec` specifies whether the image is decoded
    to numpy/torch/pi, decoded to uint8/float, and decoded
    to l/rgb/rgba:

    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchl: torch float l
    - torchrgb: torch float rgb
    - torch: torch float rgb
    - torchrgba: torch float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba
    """
    def __init__(self, imagespec):
        assert imagespec in list(imagespecs.keys()), f"unknown image specification: {imagespec}"
        self.imagespec = imagespec.lower()

    def __call__(self, extension, data):
        if extension.lower() not in "jpg jpeg png ppm pgm pbm pnm".split():
            return None

        try:
            import numpy as np
        except ImportError as e:
            raise ModuleNotFoundError("Package `numpy` is required to be installed for default image decoder."
                                      "Please use `pip install numpy` to install the package") from e

        try:
            import PIL.Image
        except ImportError as e:
            raise ModuleNotFoundError("Package `PIL` is required to be installed for default image decoder."
                                      "Please use `pip install Pillow` to install the package") from e

        imagespec = self.imagespec
        atype, etype, mode = imagespecs[imagespec]

        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = img.convert(mode.upper())
            if atype == "pil":
                return img
            elif atype == "numpy":
                result = np.asarray(img)
                assert result.dtype == np.uint8, f"numpy image array should be type uint8, but got {result.dtype}"
                if etype == "uint8":
                    return result
                else:
                    return result.astype("f") / 255.0
            elif atype == "torch":
                result = np.asarray(img)
                assert result.dtype == np.uint8, f"numpy image array should be type uint8, but got {result.dtype}"

                if etype == "uint8":
                    result = np.array(result.transpose(2, 0, 1))
                    return torch.tensor(result)
                else:
                    result = np.array(result.transpose(2, 0, 1))
                    return torch.tensor(result) / 255.0
            return None

def imagehandler(imagespec):
    return ImageHandler(imagespec)


################################################################
# torch video
################################################################
def videohandler(extension, data):
    if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
        return None

    try:
        import torchvision.io
    except ImportError as e:
        raise ModuleNotFoundError("Package `torchvision` is required to be installed for default video file loader."
                                  "Please use `pip install torchvision` or `conda install torchvision -c pytorch`"
                                  "to install the package") from e

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
            return torchvision.io.read_video(fname)


################################################################
# torchaudio
################################################################
def audiohandler(extension, data):
    if extension not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]:
        return None

    try:
        import torchaudio  # type: ignore[import]
    except ImportError as e:
        raise ModuleNotFoundError("Package `torchaudio` is required to be installed for default audio file loader."
                                  "Please use `pip install torchaudio` or `conda install torchaudio -c pytorch`"
                                  "to install the package") from e

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
            return torchaudio.load(fname)


################################################################
# mat
################################################################
class MatHandler:
    def __init__(self, **loadmat_kwargs) -> None:
        try:
            import scipy.io as sio
        except ImportError as e:
            raise ModuleNotFoundError("Package `scipy` is required to be installed for mat file."
                                      "Please use `pip install scipy` or `conda install scipy`"
                                      "to install the package") from e
        self.sio = sio
        self.loadmat_kwargs = loadmat_kwargs

    def __call__(self, extension, data):
        if extension != 'mat':
            return None
        with io.BytesIO(data) as stream:
            return self.sio.loadmat(stream, **self.loadmat_kwargs)

def mathandler(**loadmat_kwargs):
    return MatHandler(**loadmat_kwargs)


################################################################
# a sample decoder
################################################################
# Extract extension from pathname
def extension_extract_fn(pathname):
    ext = os.path.splitext(pathname)[1]
    # Remove dot
    if ext:
        ext = ext[1:]
    return ext


class Decoder:
    """
    Decode key/data sets using a list of handlers.
    For each key/data item, this iterates through the list of
    handlers until some handler returns something other than None.
    """

    def __init__(self, *handler, key_fn=extension_extract_fn):
        self.handlers = list(handler) if handler else []
        self.key_fn = key_fn

    # Insert new handler from the beginning of handlers list to make sure the new
    # handler having the highest priority
    def add_handler(self, *handler):
        if not handler:
            return
        self.handlers = list(handler) + self.handlers

    @staticmethod
    def _is_stream_handle(data):
        obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
        return isinstance(obj_to_check, (io.BufferedIOBase, io.RawIOBase))

    def decode1(self, key, data):
        if not data:
            return data

        # if data is a stream handle, we need to read all the content before decoding
        if Decoder._is_stream_handle(data):
            ds = data
            # The behavior of .read can differ between streams (e.g. HTTPResponse), hence this is used instead
            data = b"".join(data)
            ds.close()

        for f in self.handlers:
            result = f(key, data)
            if result is not None:
                return result
        return data

    def decode(self, data):
        result = {}
        # single data tuple(pathname, data stream)
        if isinstance(data, tuple):
            data = [data]

        if data is not None:
            for k, v in data:
                # TODO: xinyu, figure out why Nvidia do this?
                if k[0] == "_":
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                        result[k] = v
                        continue
                result[k] = self.decode1(self.key_fn(k), v)
        return result

    def __call__(self, data):
        return self.decode(data)
