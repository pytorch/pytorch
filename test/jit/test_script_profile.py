# Owner(s): ["oncall: jit"]

import os
import sys

import torch
from torch import nn

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class Sequence(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input):
        outputs = []
        h_t = torch.zeros(input.size(0), 51)
        c_t = torch.zeros(input.size(0), 51)
        h_t2 = torch.zeros(input.size(0), 51)
        c_t2 = torch.zeros(input.size(0), 51)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class TestScriptProfile(JitTestCase):

    def test_basic(self):
        seq = torch.jit.script(Sequence())
        p = torch.jit._ScriptProfile()
        p.enable()
        seq(torch.rand((10, 100)))
        p.disable()
        self.assertNotEqual(p.dump_string(), "")

    def test_script(self):
        seq = Sequence()

        @torch.jit.script
        def fn():
            p = torch.jit._ScriptProfile()
            p.enable()
            _ = seq(torch.rand((10, 100)))
            p.disable()
            return p

        self.assertNotEqual(fn().dump_string(), "")

    def test_multi(self):
        seq = torch.jit.script(Sequence())
        profiles = [torch.jit._ScriptProfile() for _ in range(5)]
        for p in profiles:
            p.enable()

        last = None
        while len(profiles) > 0:
            seq(torch.rand((10, 10)))
            p = profiles.pop()
            p.disable()
            stats = p.dump_string()
            self.assertNotEqual(stats, "")
            if last:
                self.assertNotEqual(stats, last)
            last = stats

    def test_section(self):
        seq = Sequence()

        @torch.jit.script
        def fn():
            p = torch.jit._ScriptProfile()
            p.enable()
            _ = seq(torch.rand((10, 100)))
            p.disable()
            stats0 = p.dump_string()

            _ = seq(torch.rand((10, 10)))
            stats1 = p.dump_string()

            p.enable()
            _ = seq(torch.rand((10, 10)))
            p.disable()
            stats2 = p.dump_string()

            p.enable()
            return stats0, stats1, stats2

        s0, s1, s2 = fn()
        self.assertEqual(s0, s1)
        self.assertNotEqual(s1, s2)

    def test_empty(self):
        p = torch.jit._ScriptProfile()
        p.enable()
        p.disable()
        self.assertEqual(p.dump_string(), "")
