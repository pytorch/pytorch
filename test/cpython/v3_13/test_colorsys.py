# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_colorsys.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import unittest
import colorsys

def frange(start, stop, step):
    while start <= stop:
        yield start
        start += step

class ColorsysTest(CPythonTestCase):

    def assertTripleEqual(self, tr1, tr2):
        self.assertEqual(len(tr1), 3)
        self.assertEqual(len(tr2), 3)
        self.assertAlmostEqual(tr1[0], tr2[0])
        self.assertAlmostEqual(tr1[1], tr2[1])
        self.assertAlmostEqual(tr1[2], tr2[2])

    def test_hsv_roundtrip(self):
        for r in frange(0.0, 1.0, 0.2):
            for g in frange(0.0, 1.0, 0.2):
                for b in frange(0.0, 1.0, 0.2):
                    rgb = (r, g, b)
                    self.assertTripleEqual(
                        rgb,
                        colorsys.hsv_to_rgb(*colorsys.rgb_to_hsv(*rgb))
                    )

    def test_hsv_values(self):
        values = [
            # rgb, hsv
            ((0.0, 0.0, 0.0), (  0  , 0.0, 0.0)), # black
            ((0.0, 0.0, 1.0), (4./6., 1.0, 1.0)), # blue
            ((0.0, 1.0, 0.0), (2./6., 1.0, 1.0)), # green
            ((0.0, 1.0, 1.0), (3./6., 1.0, 1.0)), # cyan
            ((1.0, 0.0, 0.0), (  0  , 1.0, 1.0)), # red
            ((1.0, 0.0, 1.0), (5./6., 1.0, 1.0)), # purple
            ((1.0, 1.0, 0.0), (1./6., 1.0, 1.0)), # yellow
            ((1.0, 1.0, 1.0), (  0  , 0.0, 1.0)), # white
            ((0.5, 0.5, 0.5), (  0  , 0.0, 0.5)), # grey
        ]
        for (rgb, hsv) in values:
            self.assertTripleEqual(hsv, colorsys.rgb_to_hsv(*rgb))
            self.assertTripleEqual(rgb, colorsys.hsv_to_rgb(*hsv))

    def test_hls_roundtrip(self):
        for r in frange(0.0, 1.0, 0.2):
            for g in frange(0.0, 1.0, 0.2):
                for b in frange(0.0, 1.0, 0.2):
                    rgb = (r, g, b)
                    self.assertTripleEqual(
                        rgb,
                        colorsys.hls_to_rgb(*colorsys.rgb_to_hls(*rgb))
                    )

    def test_hls_values(self):
        values = [
            # rgb, hls
            ((0.0, 0.0, 0.0), (  0  , 0.0, 0.0)), # black
            ((0.0, 0.0, 1.0), (4./6., 0.5, 1.0)), # blue
            ((0.0, 1.0, 0.0), (2./6., 0.5, 1.0)), # green
            ((0.0, 1.0, 1.0), (3./6., 0.5, 1.0)), # cyan
            ((1.0, 0.0, 0.0), (  0  , 0.5, 1.0)), # red
            ((1.0, 0.0, 1.0), (5./6., 0.5, 1.0)), # purple
            ((1.0, 1.0, 0.0), (1./6., 0.5, 1.0)), # yellow
            ((1.0, 1.0, 1.0), (  0  , 1.0, 0.0)), # white
            ((0.5, 0.5, 0.5), (  0  , 0.5, 0.0)), # grey
        ]
        for (rgb, hls) in values:
            self.assertTripleEqual(hls, colorsys.rgb_to_hls(*rgb))
            self.assertTripleEqual(rgb, colorsys.hls_to_rgb(*hls))

    def test_hls_nearwhite(self):  # gh-106498
        values = (
            # rgb, hls: these do not work in reverse
            ((0.9999999999999999, 1, 1), (0.5, 1.0, 1.0)),
            ((1, 0.9999999999999999, 0.9999999999999999), (0.0, 1.0, 1.0)),
        )
        for rgb, hls in values:
            self.assertTripleEqual(hls, colorsys.rgb_to_hls(*rgb))
            self.assertTripleEqual((1.0, 1.0, 1.0), colorsys.hls_to_rgb(*hls))

    def test_yiq_roundtrip(self):
        for r in frange(0.0, 1.0, 0.2):
            for g in frange(0.0, 1.0, 0.2):
                for b in frange(0.0, 1.0, 0.2):
                    rgb = (r, g, b)
                    self.assertTripleEqual(
                        rgb,
                        colorsys.yiq_to_rgb(*colorsys.rgb_to_yiq(*rgb))
                    )

    def test_yiq_values(self):
        values = [
            # rgb, yiq
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), # black
            ((0.0, 0.0, 1.0), (0.11, -0.3217, 0.3121)), # blue
            ((0.0, 1.0, 0.0), (0.59, -0.2773, -0.5251)), # green
            ((0.0, 1.0, 1.0), (0.7, -0.599, -0.213)), # cyan
            ((1.0, 0.0, 0.0), (0.3, 0.599, 0.213)), # red
            ((1.0, 0.0, 1.0), (0.41, 0.2773, 0.5251)), # purple
            ((1.0, 1.0, 0.0), (0.89, 0.3217, -0.3121)), # yellow
            ((1.0, 1.0, 1.0), (1.0, 0.0, 0.0)), # white
            ((0.5, 0.5, 0.5), (0.5, 0.0, 0.0)), # grey
        ]
        for (rgb, yiq) in values:
            self.assertTripleEqual(yiq, colorsys.rgb_to_yiq(*rgb))
            self.assertTripleEqual(rgb, colorsys.yiq_to_rgb(*yiq))

if __name__ == "__main__":
    run_tests()
