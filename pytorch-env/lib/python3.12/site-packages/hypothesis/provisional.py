# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""This module contains various provisional APIs and strategies.

It is intended for internal use, to ease code reuse, and is not stable.
Point releases may move or break the contents at any time!

Internet strategies should conform to :rfc:`3986` or the authoritative
definitions it links to.  If not, report the bug!
"""
# https://tools.ietf.org/html/rfc3696

import string
from functools import lru_cache
from importlib import resources
from typing import Optional

from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture import utils as cu
from hypothesis.strategies._internal.utils import defines_strategy

URL_SAFE_CHARACTERS = frozenset(string.ascii_letters + string.digits + "$-_.+!*'(),~")
FRAGMENT_SAFE_CHARACTERS = URL_SAFE_CHARACTERS | {"?", "/"}


@lru_cache(maxsize=1)
def get_top_level_domains() -> tuple[str, ...]:
    # This file is sourced from http://data.iana.org/TLD/tlds-alpha-by-domain.txt
    # The file contains additional information about the date that it was last updated.
    traversable = resources.files("hypothesis.vendor") / "tlds-alpha-by-domain.txt"
    _comment, *_tlds = traversable.read_text(encoding="utf-8").splitlines()
    assert _comment.startswith("#")

    # Remove special-use domain names from the list. For more discussion
    # see https://github.com/HypothesisWorks/hypothesis/pull/3572
    return ("COM", *sorted((d for d in _tlds if d != "ARPA"), key=len))


class DomainNameStrategy(st.SearchStrategy):
    @staticmethod
    def clean_inputs(
        minimum: int, maximum: int, value: Optional[int], variable_name: str
    ) -> int:
        if value is None:
            value = maximum
        elif not isinstance(value, int):
            raise InvalidArgument(
                f"Expected integer but {variable_name} is a {type(value).__name__}"
            )
        elif not minimum <= value <= maximum:
            raise InvalidArgument(
                f"Invalid value {minimum!r} < {variable_name}={value!r} < {maximum!r}"
            )
        return value

    def __init__(
        self, max_length: Optional[int] = None, max_element_length: Optional[int] = None
    ) -> None:
        """
        A strategy for :rfc:`1035` fully qualified domain names.

        The upper limit for max_length is 255 in accordance with :rfc:`1035#section-2.3.4`
        The lower limit for max_length is 4, corresponding to a two letter domain
        with a single letter subdomain.
        The upper limit for max_element_length is 63 in accordance with :rfc:`1035#section-2.3.4`
        The lower limit for max_element_length is 1 in accordance with :rfc:`1035#section-2.3.4`
        """
        # https://tools.ietf.org/html/rfc1035#section-2.3.4

        max_length = self.clean_inputs(4, 255, max_length, "max_length")
        max_element_length = self.clean_inputs(
            1, 63, max_element_length, "max_element_length"
        )

        super().__init__()
        self.max_length = max_length
        self.max_element_length = max_element_length

        # These regular expressions are constructed to match the documented
        # information in https://tools.ietf.org/html/rfc1035#section-2.3.1
        # which defines the allowed syntax of a subdomain string.
        if self.max_element_length == 1:
            self.label_regex = r"[a-zA-Z]"
        elif self.max_element_length == 2:
            self.label_regex = r"[a-zA-Z][a-zA-Z0-9]?"
        else:
            maximum_center_character_pattern_repetitions = self.max_element_length - 2
            self.label_regex = r"[a-zA-Z]([a-zA-Z0-9\-]{0,%d}[a-zA-Z0-9])?" % (
                maximum_center_character_pattern_repetitions,
            )

    def do_draw(self, data):
        # 1 - Select a valid top-level domain (TLD) name
        # 2 - Check that the number of characters in our selected TLD won't
        # prevent us from generating at least a 1 character subdomain.
        # 3 - Randomize the TLD between upper and lower case characters.
        domain = data.draw(
            st.sampled_from(get_top_level_domains())
            .filter(lambda tld: len(tld) + 2 <= self.max_length)
            .flatmap(
                lambda tld: st.tuples(
                    *(st.sampled_from([c.lower(), c.upper()]) for c in tld)
                ).map("".join)
            )
        )
        # RFC-5890 s2.3.1 says such labels are reserved, and since we don't
        # want to bother with xn-- punycode labels we'll exclude them all.
        elem_st = st.from_regex(self.label_regex, fullmatch=True).filter(
            lambda label: len(label) < 4 or label[2:4] != "--"
        )
        # The maximum possible number of subdomains is 126,
        # 1 character subdomain + 1 '.' character, * 126 = 252,
        # with a max of 255, that leaves 3 characters for a TLD.
        # Allowing any more subdomains would not leave enough
        # characters for even the shortest possible TLDs.
        elements = cu.many(data, min_size=1, average_size=3, max_size=126)
        while elements.more():
            # Generate a new valid subdomain using the regex strategy.
            sub_domain = data.draw(elem_st)
            if len(domain) + len(sub_domain) >= self.max_length:
                data.stop_example(discard=True)
                break
            domain = sub_domain + "." + domain
        return domain


@defines_strategy(force_reusable_values=True)
def domains(
    *, max_length: int = 255, max_element_length: int = 63
) -> st.SearchStrategy[str]:
    """Generate :rfc:`1035` compliant fully qualified domain names."""
    return DomainNameStrategy(
        max_length=max_length, max_element_length=max_element_length
    )


# The `urls()` strategy uses this to generate URL fragments (e.g. "#foo").
# It has been extracted to top-level so that we can test it independently
# of `urls()`, which helps with getting non-flaky coverage of the lambda.
_url_fragments_strategy = (
    st.lists(
        st.builds(
            lambda char, encode: (
                f"%{ord(char):02X}"
                if (encode or char not in FRAGMENT_SAFE_CHARACTERS)
                else char
            ),
            st.characters(min_codepoint=0, max_codepoint=255),
            st.booleans(),
        ),
        min_size=1,
    )
    .map("".join)
    .map("#{}".format)
)


@defines_strategy(force_reusable_values=True)
def urls() -> st.SearchStrategy[str]:
    """A strategy for :rfc:`3986`, generating http/https URLs."""

    def url_encode(s: str) -> str:
        return "".join(c if c in URL_SAFE_CHARACTERS else "%%%02X" % ord(c) for c in s)

    schemes = st.sampled_from(["http", "https"])
    ports = st.integers(min_value=0, max_value=2**16 - 1).map(":{}".format)
    paths = st.lists(st.text(string.printable).map(url_encode)).map("/".join)

    return st.builds(
        "{}://{}{}/{}{}".format,
        schemes,
        domains(),
        st.just("") | ports,
        paths,
        st.just("") | _url_fragments_strategy,
    )
