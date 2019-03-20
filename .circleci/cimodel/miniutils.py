#!/usr/bin/env python3


def quote(s):
    return '"' + s + '"'


def override(word, substitutions):
    return substitutions.get(word, word)
