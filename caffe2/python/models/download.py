## @package download
# Module caffe2.python.models.download
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os
import sys
import signal
import re

# Import urllib
try:
    import urllib.error as urlliberror
    import urllib.request as urllib
    HTTPError = urlliberror.HTTPError
    URLError = urlliberror.URLError
except ImportError:
    import urllib2 as urllib
    HTTPError = urllib.HTTPError
    URLError = urllib.URLError

# urllib requires more work to deal with a redirect, so not using vanity url
DOWNLOAD_BASE_URL = "https://s3.amazonaws.com/download.caffe2.ai/models/"
DOWNLOAD_COLUMNS = 70


# Don't let urllib hang up on big downloads
def signalHandler(signal, frame):
    print("Killing download...")
    exit(0)


signal.signal(signal.SIGINT, signalHandler)


def deleteDirectory(top_dir):
    for root, dirs, files in os.walk(top_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top_dir)


def progressBar(percentage):
    full = int(DOWNLOAD_COLUMNS * percentage / 100)
    bar = full * "#" + (DOWNLOAD_COLUMNS - full) * " "
    sys.stdout.write(u"\u001b[1000D[" + bar + "] " + str(percentage) + "%")
    sys.stdout.flush()


def downloadFromURLToFile(url, filename, show_progress=True):
    try:
        print("Downloading from {url}".format(url=url))
        response = urllib.urlopen(url)
        size = int(response.info().get('Content-Length').strip())
        chunk = min(size, 8192)
        print("Writing to {filename}".format(filename=filename))
        if show_progress:
            downloaded_size = 0
            progressBar(0)
        with open(filename, "wb") as local_file:
            while True:
                data_chunk = response.read(chunk)
                if not data_chunk:
                    break
                local_file.write(data_chunk)
                if show_progress:
                    downloaded_size += len(data_chunk)
                    progressBar(int(100 * downloaded_size / size))
        print("")  # New line to fix for progress bar
    except HTTPError as e:
        raise Exception("Could not download model. [HTTP Error] {code}: {reason}."
                        .format(code=e.code, reason=e.reason))
    except URLError as e:
        raise Exception("Could not download model. [URL Error] {reason}."
                        .format(reason=e.reason))
    except Exception as e:
        raise e


def getURLFromName(name, filename):
    return "{base_url}{name}/{filename}".format(base_url=DOWNLOAD_BASE_URL,
                                                name=name, filename=filename)


def downloadModel(model, args):
    # Figure out where to store the model
    model_folder = '{folder}'.format(folder=model)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args.install:
        model_folder = '{dir_path}/{folder}'.format(dir_path=dir_path,
                                                    folder=model)

    # Check if that folder is already there
    if os.path.exists(model_folder) and not os.path.isdir(model_folder):
        if not args.force:
            raise Exception("Cannot create folder for storing the model,\
                            there exists a file of the same name.")
        else:
            print("Overwriting existing file! ({filename})"
                  .format(filename=model_folder))
            os.remove(model_folder)
    if os.path.isdir(model_folder):
        if not args.force:
            response = ""
            query = "Model already exists, continue? [y/N] "
            try:
                response = raw_input(query)
            except NameError:
                response = input(query)
            if response.upper() == 'N' or not response:
                print("Cancelling download...")
                exit(0)
        print("Overwriting existing folder! ({filename})".format(filename=model_folder))
        deleteDirectory(model_folder)

    # Now we can safely create the folder and download the model
    os.makedirs(model_folder)
    for f in ['predict_net.pb', 'init_net.pb']:
        try:
            downloadFromURLToFile(getURLFromName(model, f),
                                  '{folder}/{f}'.format(folder=model_folder,
                                                        f=f))
        except Exception as e:
            print("Abort: {reason}".format(reason=str(e)))
            print("Cleaning up...")
            deleteDirectory(model_folder)
            exit(0)

    if args.install:
        os.symlink("{folder}/__sym_init__.py".format(folder=dir_path),
                   "{folder}/__init__.py".format(folder=model_folder))


def validModelName(name):
    invalid_names = ['__init__']
    if name in invalid_names:
        return False
    if not re.match("^[/0-9a-zA-Z_-]+$", name):
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download or install pretrained models.')
    parser.add_argument('model', nargs='+',
                        help='Model to download/install.')
    parser.add_argument('-i', '--install', action='store_true',
                        help='Install the model.')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force a download/installation.')
    args = parser.parse_args()
    for model in args.model:
        if validModelName(model):
            downloadModel(model, args)
        else:
            print("'{}' is not a valid model name.".format(model))
