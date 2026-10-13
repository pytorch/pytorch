"""Compile-time rounding variants for MXFP8 block-scaled TMA kernels."""

from enum import IntEnum


class RoundingVariant(IntEnum):
    RTNE = 0
    STATELESS_SR = 1
    STATEFUL_SR_EAGER = 2
    STATEFUL_SR_CAPTURE = 3
