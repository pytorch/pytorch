#!/usr/bin/env python

"""Downloads the clang-format binary appropriate for the host platform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import platform
import sys
import urllib.request
import urllib.error


def report_download_progress(chunk_number, chunk_size, file_size):
    """Pretty printer for file download progress."""
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write("\r0% |{:<64}| {}%".format(bar, int(percent * 100)))


def download_clang_format(path):
    """Downloads a clang-format binary appropriate for the host platform and stores it at the given location."""
    # This dictionary maps each platform to the S3 object URL for its clang-format binary.
    PLATFORM_TO_URL = {
        "Darwin": "https://oss-clang-format.s3.us-east-2.amazonaws.com/mac/clang-format-mojave",
        "Linux": "https://oss-clang-format.s3.us-east-2.amazonaws.com/linux64/clang-format-linux64",
    }

    plat = platform.system()

    if plat not in PLATFORM_TO_URL:
        print("Unsupported platform: {}".format(plat))
        return False

    cf_url = PLATFORM_TO_URL[plat]
    filename = "{}{}clang-format".format(path, os.sep)

    # Try to download clang-format.
    try:
        urllib.request.urlretrieve(
            cf_url, filename, reporthook=report_download_progress
        )
    except urllib.error.URLError as e:
        print("Error downloading {}: {}".format(filename, str(e)))
        return False
    finally:
        print()

    return True


def parse_args(args):
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Install a platform-appropriate clang-format binary for use during PyTorch development."
    )
    parser.add_argument(
        "--path", required=True, help="Path where clang-format should be stored"
    )

    return parser.parse_args(args)


def sanitize_path(path):
    """Sanitize a given path by expanding ~ and converting a relative path to an absolute path."""
    expand_user = os.path.expanduser(path)
    abs_path = os.path.abspath(expand_user)

    return abs_path


def main(args):
    # Parse arguments.
    options = parse_args(args)
    # Make sure the path is absolute.
    path = sanitize_path(options.path)
    # Try downloading clang-format.
    ok = download_clang_format(path)

    if ok:
        print("Successfully downloaded clang-format to {}".format(path))
        print("Remember to add {} to your PATH".format(path))

    return not ok


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
