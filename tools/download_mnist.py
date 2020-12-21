import argparse
import gzip
import os
from urllib.error import URLError
from urllib.request import urlretrieve
import sys

MIRRORS = [
    'http://yann.lecun.com/exdb/mnist/',
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
]

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


def download(destination_path, resource, quiet):
    if os.path.exists(destination_path):
        if not quiet:
            print('{} already exists, skipping ...'.format(destination_path))
    else:
        for mirror in MIRRORS:
            url = mirror + resource
            print('Downloading {} ...'.format(url))
            try:
                hook = None if quiet else report_download_progress
                urlretrieve(url, destination_path, reporthook=hook)
            except URLError as e:
                print('Failed to download (trying next):\n{}'.format(e))
                continue
            finally:
                if not quiet:
                    # Just a newline.
                    print()
            break
        else:
            raise RuntimeError('Error downloading resource!')


def unzip(zipped_path, quiet):
    unzipped_path = os.path.splitext(zipped_path)[0]
    if os.path.exists(unzipped_path):
        if not quiet:
            print('{} already exists, skipping ... '.format(unzipped_path))
        return
    with gzip.open(zipped_path, 'rb') as zipped_file:
        with open(unzipped_path, 'wb') as unzipped_file:
            unzipped_file.write(zipped_file.read())
            if not quiet:
                print('Unzipped {} ...'.format(zipped_path))


def main():
    parser = argparse.ArgumentParser(
        description='Download the MNIST dataset from the internet')
    parser.add_argument(
        '-d', '--destination', default='.', help='Destination directory')
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help="Don't report about progress")
    options = parser.parse_args()

    if not os.path.exists(options.destination):
        os.makedirs(options.destination)

    try:
        for resource in RESOURCES:
            path = os.path.join(options.destination, resource)
            download(path, resource, options.quiet)
            unzip(path, options.quiet)
    except KeyboardInterrupt:
        print('Interrupted')


if __name__ == '__main__':
    main()
