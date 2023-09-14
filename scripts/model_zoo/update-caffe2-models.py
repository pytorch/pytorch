#! /usr/bin/env python3

import os
import subprocess
import sys
import tarfile
import tempfile

from urllib.request import urlretrieve

from caffe2.python.models.download import (
    deleteDirectory,
    downloadFromURLToFile,
    getURLFromName,
)


class SomeClass:
    # largely copied from
    # https://github.com/onnx/onnx-caffe2/blob/master/tests/caffe2_ref_test.py
    def _download(self, model):
        model_dir = self._caffe2_model_dir(model)
        assert not os.path.exists(model_dir)
        os.makedirs(model_dir)
        for f in ["predict_net.pb", "init_net.pb", "value_info.json"]:
            url = getURLFromName(model, f)
            dest = os.path.join(model_dir, f)
            try:
                try:
                    downloadFromURLToFile(url, dest, show_progress=False)
                except TypeError:
                    # show_progress not supported prior to
                    # Caffe2 78c014e752a374d905ecfb465d44fa16e02a28f1
                    # (Sep 17, 2017)
                    downloadFromURLToFile(url, dest)
            except Exception as e:
                print(f"Abort: {e}")
                print("Cleaning up...")
                deleteDirectory(model_dir)
                exit(1)

    def _caffe2_model_dir(self, model):
        caffe2_home = os.path.expanduser("~/.caffe2")
        models_dir = os.path.join(caffe2_home, "models")
        return os.path.join(models_dir, model)

    def _onnx_model_dir(self, model):
        onnx_home = os.path.expanduser("~/.onnx")
        models_dir = os.path.join(onnx_home, "models")
        model_dir = os.path.join(models_dir, model)
        return model_dir, os.path.dirname(model_dir)

    # largely copied from
    # https://github.com/onnx/onnx/blob/master/onnx/backend/test/runner/__init__.py
    def _prepare_model_data(self, model):
        model_dir, models_dir = self._onnx_model_dir(model)
        if os.path.exists(model_dir):
            return
        os.makedirs(model_dir)
        url = f"https://s3.amazonaws.com/download.onnx/models/{model}.tar.gz"

        # On Windows, NamedTemporaryFile cannot be opened for a
        # second time
        download_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            download_file.close()
            print(f"Start downloading model {model} from {url}")
            urlretrieve(url, download_file.name)
            print("Done")
            with tarfile.open(download_file.name) as t:
                t.extractall(models_dir)
        except Exception as e:
            print(f"Failed to prepare data for model {model}: {e}")
            raise
        finally:
            os.remove(download_file.name)


models = [
    "bvlc_alexnet",
    "densenet121",
    "inception_v1",
    "inception_v2",
    "resnet50",
    # TODO currently onnx can't translate squeezenet :(
    # 'squeezenet',
    "vgg16",
    # TODO currently vgg19 doesn't work in the CI environment,
    # possibly due to OOM
    # 'vgg19'
]


def download_models():
    sc = SomeClass()
    for model in models:
        print("update-caffe2-models.py:  downloading", model)
        caffe2_model_dir = sc._caffe2_model_dir(model)
        onnx_model_dir, onnx_models_dir = sc._onnx_model_dir(model)
        if not os.path.exists(caffe2_model_dir):
            sc._download(model)
        if not os.path.exists(onnx_model_dir):
            sc._prepare_model_data(model)


def generate_models():
    sc = SomeClass()
    for model in models:
        print("update-caffe2-models.py:  generating", model)
        caffe2_model_dir = sc._caffe2_model_dir(model)
        onnx_model_dir, onnx_models_dir = sc._onnx_model_dir(model)
        subprocess.check_call(["echo", model])
        with open(os.path.join(caffe2_model_dir, "value_info.json"), "r") as f:
            value_info = f.read()
        subprocess.check_call(
            [
                "convert-caffe2-to-onnx",
                "--caffe2-net-name",
                model,
                "--caffe2-init-net",
                os.path.join(caffe2_model_dir, "init_net.pb"),
                "--value-info",
                value_info,
                "-o",
                os.path.join(onnx_model_dir, "model.pb"),
                os.path.join(caffe2_model_dir, "predict_net.pb"),
            ]
        )
        subprocess.check_call(
            ["tar", "-czf", model + ".tar.gz", model], cwd=onnx_models_dir
        )


def upload_models():
    sc = SomeClass()
    for model in models:
        print("update-caffe2-models.py:  uploading", model)
        onnx_model_dir, onnx_models_dir = sc._onnx_model_dir(model)
        subprocess.check_call(
            [
                "aws",
                "s3",
                "cp",
                model + ".tar.gz",
                f"s3://download.onnx/models/{model}.tar.gz",
                "--acl",
                "public-read",
            ],
            cwd=onnx_models_dir,
        )


def cleanup():
    sc = SomeClass()
    for model in models:
        onnx_model_dir, onnx_models_dir = sc._onnx_model_dir(model)
        os.remove(os.path.join(os.path.dirname(onnx_model_dir), model + ".tar.gz"))


if __name__ == "__main__":
    try:
        subprocess.check_call(["aws", "sts", "get-caller-identity"])
    except:
        print(
            "update-caffe2-models.py:  please run `aws configure` manually to set up credentials"
        )
        sys.exit(1)
    if sys.argv[1] == "download":
        download_models()
    if sys.argv[1] == "generate":
        generate_models()
    elif sys.argv[1] == "upload":
        upload_models()
    elif sys.argv[1] == "cleanup":
        cleanup()
