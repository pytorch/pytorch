# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Dict, Optional, Tuple


def _parse_rendezvous_config(config_str: str) -> Dict[str, str]:
    """
    Extracts key-value pairs from a configuration string that has the format
    <key1>=<value1>,...,<keyN>=<valueN>.
    """
    config: Dict[str, str] = {}

    if not config_str:
        return config

    key_values = config_str.split(",")
    for kv in key_values:
        key, *values = kv.split("=", 1)
        if not values:
            raise ValueError(f"The '{key}' rendezvous config has no value specified.")
        config[key] = values[0]
    return config


def _parse_hostname_and_port(
    endpoint: Optional[str], default_port: int
) -> Tuple[str, int]:
    """
    Extracts the hostname and the port number from an endpoint string that has
    the format <hostname>:<port>.

    If no hostname can be found, defaults to the loopback address 127.0.0.1.
    """
    if not endpoint:
        return ("127.0.0.1", default_port)

    hostname, *rest = endpoint.rsplit(":", 1)
    if len(rest) == 1:
        if re.match(r"^[0-9]{1,5}$", rest[0]):
            port = int(rest[0])
        else:
            port = 0
        if port <= 80 or port >= 2 ** 16:
            raise ValueError(
                f"The rendezvous endpoint '{endpoint}' has an invalid port number '{rest[0]}'."
            )
    else:
        port = default_port

    if not re.match(r"^[\w\.:-]+$", hostname):
        raise ValueError(
            f"The rendezvous enpoint '{endpoint}' has an invalid hostname '{hostname}'."
        )

    return hostname, port
