#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test helper binary for the configurable shutdown timeout tests.

With ``--fail`` the process exits immediately with exit code 1 (simulating a
failed worker). Otherwise it ignores SIGTERM and sleeps "forever", so that
the parent (PContext/agent) has to escalate to SIGKILL after the configured
shutdown timeout.
"""

import argparse
import signal
import sys
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--fail-delay", type=float, default=2.0)
    args = parser.parse_args()

    if args.fail:
        # Delay the failure slightly so that the sibling worker has time to
        # install its SIGTERM handler before it gets terminated.
        time.sleep(args.fail_delay)
        sys.exit(1)

    # Ignore SIGTERM so that the launcher must wait the full shutdown
    # timeout before sending SIGKILL.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(600)
