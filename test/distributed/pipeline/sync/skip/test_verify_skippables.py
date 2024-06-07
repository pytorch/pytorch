# Owner(s): ["oncall: distributed"]

# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from torch import nn

from torch.distributed.pipeline.sync.skip import Namespace, skippable, verify_skippables
from torch.testing._internal.common_utils import run_tests


def test_matching():
    @skippable(stash=["foo"])
    class Layer1(nn.Module):
        pass

    @skippable(pop=["foo"])
    class Layer2(nn.Module):
        pass

    verify_skippables(nn.Sequential(Layer1(), Layer2()))


def test_stash_not_pop():
    @skippable(stash=["foo"])
    class Layer1(nn.Module):
        pass

    with pytest.raises(TypeError) as e:
        verify_skippables(nn.Sequential(Layer1()))
    assert "no module declared 'foo' as poppable but stashed" in str(e.value)


def test_pop_unknown():
    @skippable(pop=["foo"])
    class Layer1(nn.Module):
        pass

    with pytest.raises(TypeError) as e:
        verify_skippables(nn.Sequential(Layer1()))
    assert "'0' declared 'foo' as poppable but it was not stashed" in str(e.value)


def test_stash_again():
    @skippable(stash=["foo"])
    class Layer1(nn.Module):
        pass

    @skippable(stash=["foo"])
    class Layer2(nn.Module):
        pass

    @skippable(pop=["foo"])
    class Layer3(nn.Module):
        pass

    with pytest.raises(TypeError) as e:
        verify_skippables(nn.Sequential(Layer1(), Layer2(), Layer3()))
    assert "'1' redeclared 'foo' as stashable" in str(e.value)


def test_pop_again():
    @skippable(stash=["foo"])
    class Layer1(nn.Module):
        pass

    @skippable(pop=["foo"])
    class Layer2(nn.Module):
        pass

    @skippable(pop=["foo"])
    class Layer3(nn.Module):
        pass

    with pytest.raises(TypeError) as e:
        verify_skippables(nn.Sequential(Layer1(), Layer2(), Layer3()))
    assert "'2' redeclared 'foo' as poppable" in str(e.value)


def test_stash_pop_together_different_names():
    @skippable(stash=["foo"])
    class Layer1(nn.Module):
        pass

    @skippable(pop=["foo"], stash=["bar"])
    class Layer2(nn.Module):
        pass

    @skippable(pop=["bar"])
    class Layer3(nn.Module):
        pass

    verify_skippables(nn.Sequential(Layer1(), Layer2(), Layer3()))


def test_stash_pop_together_same_name():
    @skippable(stash=["foo"], pop=["foo"])
    class Layer1(nn.Module):
        pass

    with pytest.raises(TypeError) as e:
        verify_skippables(nn.Sequential(Layer1()))
    assert "'0' declared 'foo' both as stashable and as poppable" in str(e.value)


def test_double_stash_pop():
    @skippable(stash=["foo"])
    class Layer1(nn.Module):
        pass

    @skippable(pop=["foo"])
    class Layer2(nn.Module):
        pass

    @skippable(stash=["foo"])
    class Layer3(nn.Module):
        pass

    @skippable(pop=["foo"])
    class Layer4(nn.Module):
        pass

    with pytest.raises(TypeError) as e:
        verify_skippables(nn.Sequential(Layer1(), Layer2(), Layer3(), Layer4()))
    assert "'2' redeclared 'foo' as stashable" in str(e.value)
    assert "'3' redeclared 'foo' as poppable" in str(e.value)


def test_double_stash_pop_but_isolated():
    @skippable(stash=["foo"])
    class Layer1(nn.Module):
        pass

    @skippable(pop=["foo"])
    class Layer2(nn.Module):
        pass

    @skippable(stash=["foo"])
    class Layer3(nn.Module):
        pass

    @skippable(pop=["foo"])
    class Layer4(nn.Module):
        pass

    ns1 = Namespace()
    ns2 = Namespace()

    verify_skippables(
        nn.Sequential(
            Layer1().isolate(ns1),
            Layer2().isolate(ns1),
            Layer3().isolate(ns2),
            Layer4().isolate(ns2),
        )
    )


if __name__ == "__main__":
    run_tests()
