"""
To write tf_record into file. Here we use it for tensorboard's event writting.
The code was borrowed from https://github.com/TeamHG-Memex/tensorboard_logger
"""

import copy
import io
import os.path
import re
import struct
try:
    import boto3
    S3_ENABLED = True
except ImportError:
    S3_ENABLED = False

from .crc32c import crc32c


_VALID_OP_NAME_START = re.compile('^[A-Za-z0-9.]')
_VALID_OP_NAME_PART = re.compile('[A-Za-z0-9_.\\-/]+')

# Registry of writer factories by prefix backends.
#
# Currently supports "s3://" URLs for S3 based on boto and falls
# back to local filesystem.
REGISTERED_FACTORIES = {}


def register_writer_factory(prefix, factory):
    if ':' in prefix:
        raise ValueError('prefix cannot contain a :')
    REGISTERED_FACTORIES[prefix] = factory


def directory_check(path):
    '''Initialize the directory for log files.'''
    try:
        prefix = path.split(':')[0]
        factory = REGISTERED_FACTORIES[prefix]
        return factory.directory_check(path)
    except KeyError:
        if not os.path.exists(path):
            os.makedirs(path)


def open_file(path):
    '''Open a writer for outputting event files.'''
    try:
        prefix = path.split(':')[0]
        factory = REGISTERED_FACTORIES[prefix]
        return factory.open(path)
    except KeyError:
        return open(path, 'wb')


class S3RecordWriter(object):
    """Writes tensorboard protocol buffer files to S3."""

    def __init__(self, path):
        if not S3_ENABLED:
            raise ImportError("boto3 must be installed for S3 support.")
        self.path = path
        self.buffer = io.BytesIO()

    def __del__(self):
        self.close()

    def bucket_and_path(self):
        path = self.path
        if path.startswith("s3://"):
            path = path[len("s3://"):]
        bp = path.split("/")
        bucket = bp[0]
        path = path[1 + len(bucket):]
        return bucket, path

    def write(self, val):
        self.buffer.write(val)

    def flush(self):
        s3 = boto3.client('s3')
        bucket, path = self.bucket_and_path()
        upload_buffer = copy.copy(self.buffer)
        upload_buffer.seek(0)
        s3.upload_fileobj(upload_buffer, bucket, path)

    def close(self):
        self.flush()


class S3RecordWriterFactory(object):
    """Factory for event protocol buffer files to S3."""

    def open(self, path):
        return S3RecordWriter(path)

    def directory_check(self, path):
        # S3 doesn't need directories created before files are added
        # so we can just skip this check
        pass


register_writer_factory("s3", S3RecordWriterFactory())


class RecordWriter(object):
    def __init__(self, path):
        self._name_to_tf_name = {}
        self._tf_names = set()
        self.path = path
        self._writer = None
        self._writer = open_file(path)

    def write(self, event_str):
        w = self._writer.write
        header = struct.pack('Q', len(event_str))
        w(header)
        w(struct.pack('I', masked_crc32c(header)))
        w(event_str)
        w(struct.pack('I', masked_crc32c(event_str)))

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()


def masked_crc32c(data):
    x = u32(crc32c(data))
    return u32(((x >> 15) | u32(x << 17)) + 0xa282ead8)


def u32(x):
    return x & 0xffffffff


def make_valid_tf_name(name):
    if not _VALID_OP_NAME_START.match(name):
        # Must make it valid somehow, but don't want to remove stuff
        name = '.' + name
    return '_'.join(_VALID_OP_NAME_PART.findall(name))
