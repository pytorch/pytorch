# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Integration with third-party libraries."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from types import ModuleType

    from optree.integration import jax, numpy, torch


SUBMODULES: frozenset[str] = frozenset({'jax', 'numpy', 'torch'})


def __getattr__(name: str) -> ModuleType:
    if name in SUBMODULES:
        import importlib  # pylint: disable=import-outside-toplevel
        import sys  # pylint: disable=import-outside-toplevel

        module = sys.modules[__name__]

        submodule = importlib.import_module(f'{__name__}.{name}')
        setattr(module, name, submodule)
        return submodule

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


del TYPE_CHECKING
