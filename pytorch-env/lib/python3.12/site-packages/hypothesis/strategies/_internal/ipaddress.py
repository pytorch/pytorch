# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network, ip_network
from typing import Literal, Optional, Union

from hypothesis.errors import InvalidArgument
from hypothesis.internal.validation import check_type
from hypothesis.strategies._internal.core import binary, sampled_from
from hypothesis.strategies._internal.numbers import integers
from hypothesis.strategies._internal.strategies import SearchStrategy
from hypothesis.strategies._internal.utils import defines_strategy

# See https://www.iana.org/assignments/iana-ipv4-special-registry/
SPECIAL_IPv4_RANGES = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.0.0/29",
    "192.0.0.8/32",
    "192.0.0.9/32",
    "192.0.0.10/32",
    "192.0.0.170/32",
    "192.0.0.171/32",
    "192.0.2.0/24",
    "192.31.196.0/24",
    "192.52.193.0/24",
    "192.88.99.0/24",
    "192.168.0.0/16",
    "192.175.48.0/24",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "240.0.0.0/4",
    "255.255.255.255/32",
)
# and https://www.iana.org/assignments/iana-ipv6-special-registry/
SPECIAL_IPv6_RANGES = (
    "::1/128",
    "::/128",
    "::ffff:0:0/96",
    "64:ff9b::/96",
    "64:ff9b:1::/48",
    "100::/64",
    "2001::/23",
    "2001::/32",
    "2001:1::1/128",
    "2001:1::2/128",
    "2001:2::/48",
    "2001:3::/32",
    "2001:4:112::/48",
    "2001:10::/28",
    "2001:20::/28",
    "2001:db8::/32",
    "2002::/16",
    "2620:4f:8000::/48",
    "fc00::/7",
    "fe80::/10",
)


@defines_strategy(force_reusable_values=True)
def ip_addresses(
    *,
    v: Optional[Literal[4, 6]] = None,
    network: Optional[Union[str, IPv4Network, IPv6Network]] = None,
) -> SearchStrategy[Union[IPv4Address, IPv6Address]]:
    r"""Generate IP addresses - ``v=4`` for :class:`~python:ipaddress.IPv4Address`\ es,
    ``v=6`` for :class:`~python:ipaddress.IPv6Address`\ es, or leave unspecified
    to allow both versions.

    ``network`` may be an :class:`~python:ipaddress.IPv4Network` or
    :class:`~python:ipaddress.IPv6Network`, or a string representing a network such as
    ``"127.0.0.0/24"`` or ``"2001:db8::/32"``.  As well as generating addresses within
    a particular routable network, this can be used to generate addresses from a
    reserved range listed in the
    `IANA <https://www.iana.org/assignments/iana-ipv4-special-registry/>`__
    `registries <https://www.iana.org/assignments/iana-ipv6-special-registry/>`__.

    If you pass both ``v`` and ``network``, they must be for the same version.
    """
    if v is not None:
        check_type(int, v, "v")
        if v not in (4, 6):
            raise InvalidArgument(f"{v=}, but only v=4 or v=6 are valid")
    if network is None:
        # We use the reserved-address registries to boost the chance
        # of generating one of the various special types of address.
        four = binary(min_size=4, max_size=4).map(IPv4Address) | sampled_from(
            SPECIAL_IPv4_RANGES
        ).flatmap(lambda network: ip_addresses(network=network))
        six = binary(min_size=16, max_size=16).map(IPv6Address) | sampled_from(
            SPECIAL_IPv6_RANGES
        ).flatmap(lambda network: ip_addresses(network=network))
        if v == 4:
            return four
        if v == 6:
            return six
        return four | six
    if isinstance(network, str):
        network = ip_network(network)
    check_type((IPv4Network, IPv6Network), network, "network")
    assert isinstance(network, (IPv4Network, IPv6Network))  # for Mypy
    if v not in (None, network.version):
        raise InvalidArgument(f"{v=} is incompatible with {network=}")
    addr_type = IPv4Address if network.version == 4 else IPv6Address
    return integers(int(network[0]), int(network[-1])).map(addr_type)  # type: ignore
