# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import pytest
from torch import nn

from torch.distributed._pipeline.sync.skip import Namespace, skippable, verify_skippables


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
        nn.Sequential(Layer1().isolate(ns1), Layer2().isolate(ns1), Layer3().isolate(ns2), Layer4().isolate(ns2),)
    )
