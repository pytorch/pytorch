# This file takes partial of the implementation from NVIDIA's webdataset at here:
# https://github.com/tmbdev/webdataset/blob/master/webdataset/autodecode.py

import io
import json
import os.path
import pickle
import tempfile

from typing import Callable, List, Optional, Union

import torch


_all__ = ["Handler", "Decoder", "basichandlers", "imagehandler"]


class Handler:
    r"""
    Handler to use `decode_fn` to decode 'data' based on 'key', and to determine
    if the key-data pair should be handled. When `key_fn` is specified, it will
    be applied to 'key' for decoding data and determining if it's correct Handler.
    args:
        keys: The keys to determine the right handler for key-data pair
        decode_fn: Optional function to decode data
        key_fn: Optional function to extract detailed key from key-data pair

    Note:
        For subclass of `Handler`, importing third-party library in `__init__` to
        error out `ModuleNotFoundError` at construction time
    """
    def __init__(self,
                 keys: Union[str, List[str]],
                 decode_fn: Optional[Callable] = None,
                 key_fn: Optional[Callable] = None) -> None:
        self.keys = keys if isinstance(keys, list) else list(keys)
        if decode_fn is None:
            assert hasattr(self, 'decode_fn'), \
                "Expected {} Class has implementation of `decode_fn`".format(type(self).__name__)
        else:
            self.decode_fn = decode_fn
        self.key_fn = key_fn

    def __call__(self, key, data):
        if self.key_fn is not None:
            key = self.key_fn(key)
        return self.decode_fn(key, data)

    def __contains__(self, key):
        if self.key_fn is not None:
            key = self.key_fn(key)
        return key in self.keys


################################################################
# handle basic datatypes
################################################################
def _text_decode_fn(key, data):
    return data.decode("utf-8")

_text_handler = Handler(["txt", "text", "transcript"], _text_decode_fn)


def _int_decode_fn(key, data):
    return int(data)

_int_handler = Handler(["cls", "cls2", "class", "count", "index", "inx", "id"], _int_decode_fn)


def _json_decode_fn(key, data):
    return json.loads(data)

_json_handler = Handler(["json", "jsn"], _json_decode_fn)


def _pickle_decode_fn(key, data):
    return pickle.loads(data)

_pickle_handler = Handler(["pyd", "pickle"], _pickle_decode_fn)


def _torch_decode_fn(key, data):
    stream = io.BytesIO(data)
    return torch.load(stream)

_torch_handler = Handler("pt", _torch_decode_fn)


basichandlers = (_text_handler, _int_handler, _json_handler, _pickle_handler, _torch_handler)


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


class ImageHandler(Handler):
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
    def __init__(self,
                 imagespec: str,
                 keys="jpg jpeg png ppm pgm pbm pnm".split(),
                 key_fn=None) -> None:
        assert imagespec in list(imagespecs.keys()), \
            "unknown image specification: {}".format(imagespec)

        try:
            import numpy
            self.np = numpy
        except ImportError as e:
            raise ModuleNotFoundError("Package `numpy` is required to be installed for default image decoder."
                                      "Please use `pip install numpy` to install the package")

        try:
            import PIL.Image
            self.pil = PIL.Image
        except ImportError as e:
            raise ModuleNotFoundError("Package `PIL` is required to be installed for default image decoder."
                                      "Please use `pip install Pillow` to install the package")

        super().__init__(keys, None, key_fn)
        self.imagespec = imagespec.lower()

    def decode_fn(self, key, data):

        imagespec = self.imagespec
        atype, etype, mode = imagespecs[imagespec]

        with io.BytesIO(data) as stream:
            img = self.pil.open(stream)
            img.load()
            img = img.convert(mode.upper())
            if atype == "pil":
                return img
            elif atype == "numpy":
                result = self.np.asarray(img)  # type: ignore[misc]
                if result.dtype != self.np.uint8:  # type: ignore[misc]
                    raise TypeError("numpy image array should be type uint8, but got {}".format(result.dtype))
                if etype == "uint8":
                    return result
                else:
                    return result.astype("f") / 255.0
            elif atype == "torch":
                result = self.np.asarray(img)  # type: ignore[misc]
                if result.dtype != self.np.uint8:  # type: ignore[misc]
                    raise TypeError("numpy image array should be type uint8, but got {}".format(result.dtype))

                if etype == "uint8":
                    result = self.np.array(result.transpose(2, 0, 1))  # type: ignore[misc]
                    return torch.tensor(result)
                else:
                    result = self.np.array(result.transpose(2, 0, 1))  # type: ignore[misc]
                    return torch.tensor(result) / 255.0
            return None


def imagehandler(imagespec):
    return ImageHandler(imagespec)


################################################################
# torch video
################################################################
class VideoHandler(Handler):
    def __init__(self,
                 keys="mp4 ogv mjpeg avi mov h264 mpg webm wmv".split(),
                 key_fn=None):
        try:
            import torchvision.io
            self.tvio = torchvision.io
        except ImportError as e:
            raise ModuleNotFoundError("Package `torchvision` is required to be installed for default video file loader."
                                      "Please use `pip install torchvision` or `conda install torchvision -c pytorch`"
                                      "to install the package")

    def decode_fn(self, key, data):
        with tempfile.TemporaryDirectory() as dirname:
            fname = os.path.join(dirname, f"file.{key}")
            with open(fname, "wb") as stream:
                stream.write(data)
                return self.tvio.read_video(fname)


################################################################
# torchaudio
################################################################
class AudioHandler(Handler):
    def __init__(self,
                 keys="flac mp3 sox wav m4a ogg wma".split(),
                 key_fn=None):
        try:
            import torchaudio  # type: ignore[import]
            self.torchaudio = torchaudio
        except ImportError as e:
            raise ModuleNotFoundError("Package `torchaudio` is required to be installed for default audio file loader."
                                      "Please use `pip install torchaudio` or `conda install torchaudio -c pytorch`"
                                      "to install the package")
        super().__init__(keys, None, key_fn)

    def decode_fn(self, key, data):
        with tempfile.TemporaryDirectory() as dirname:
            fname = os.path.join(dirname, f"file.{key}")
            with open(fname, "wb") as stream:
                stream.write(data)
                return self.torchaudio.load(fname)


################################################################
# mat
################################################################
class MatHandler(Handler):
    def __init__(self,
                 keys="mat",
                 key_fn=None,
                 **loadmat_kwargs) -> None:
        try:
            import scipy.io as sio
        except ImportError as e:
            raise ModuleNotFoundError("Package `scipy` is required to be installed for mat file."
                                      "Please use `pip install scipy` or `conda install scipy`"
                                      "to install the package")
        super().__init__(keys, None, key_fn)
        self.sio = sio
        self.loadmat_kwargs = loadmat_kwargs

    def decode_fn(self, key, data):
        with io.BytesIO(data) as stream:
            return self.sio.loadmat(stream, **self.loadmat_kwargs)


################################################################
# a sample decoder
################################################################
# Extract extension from pathname
def _default_key_fn(pathname):
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
    args:
        handlers: `Handler`(s) to decode key-data pair
        key_fn: Function to extract key from data. Extracting extension
            from file path is set as default
    """
    handlers: List[Handler]

    def __init__(self,
                 *handler: Handler,
                 key_fn: Callable = _default_key_fn) -> None:
        self.handlers = list(handler) if handler else []
        self.key_fn = key_fn

    # Insert new handler from the beginning of handlers list to make sure the new
    # handler having the highest priority
    def add_handler(self, *handler: Handler) -> None:
        if not handler:
            return
        self.handlers[0: 0] = list(handler)

    def decode1(self, key, data):
        if not data:
            return data

        # if data is a stream handle, we need to read all the content before decoding
        if isinstance(data, io.BufferedIOBase) or isinstance(data, io.RawIOBase):
            data = data.read()

        for f in self.handlers:
            if key in f:
                return f(key, data)
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
