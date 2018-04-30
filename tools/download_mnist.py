from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import urllib

try:
    from urllib.error import URLError
    from urllib.request import urlretrieve
except ImportError:
    from urllib2 import URLError
    from urllib import urlretrieve

RESOURCES = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
]


def report_download_progress(chunk_number, chunk_size, file_size):
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = '#' * int(64 * percent)
        sys.stdout.write('\r0% |{:<64}| {}%'.format(bar, int(percent * 100)))


def download(destination_path, url):
    if os.path.exists(destination_path):
        print('{} already exists, skipping ...'.format(destination_path))
    else:
        print('Downloading {} ...'.format(url))
        try:
            urlretrieve(
                url, destination_path, reporthook=report_download_progress)
        except URLError:
            raise RuntimeError('Error downloading resource!')
        finally:
            # Just a newline.
            print()


def unzip(zipped_path):
    unzipped_path = os.path.splitext(zipped_path)[0]
    if os.path.exists(unzipped_path):
        print('{} already exists, skipping ... '.format(unzipped_path))
        return
    with gzip.open(zipped_path, 'rb') as zipped_file:
        with open(unzipped_path, 'wb') as unzipped_file:
            unzipped_file.write(zipped_file.read())
            print('Unzipped {} ...'.format(zipped_path))


def main():
    parser = argparse.ArgumentParser(
        description='Download the MNIST dataset from the internet')
    parser.add_argument(
        '-d', '--destination', default='.', help='Destination directory')
    options = parser.parse_args()

    if not os.path.exists(options.destination):
        os.makedirs(options.destination)

    try:
        for resource in RESOURCES:
            path = os.path.join(options.destination, resource)
            url = 'http://yann.lecun.com/exdb/mnist/{}'.format(resource)
            download(path, url)
            unzip(path)
    except KeyboardInterrupt:
        print('Interrupted')

    print('Done')


if __name__ == '__main__':
    main()
