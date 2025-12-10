# Copyright 2022-2025 MetaOPT Team. All Rights Reserved.
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
"""The :mod:`optree.treespec` namespace contains constructors for class :class:`optree.PyTreeSpec`.

>>> import optree.treespec as treespec
>>> treespec.leaf()
PyTreeSpec(*)
>>> treespec.none()
PyTreeSpec(None)
>>> treespec.dict({'a': treespec.leaf(), 'b': treespec.leaf()})
PyTreeSpec({'a': *, 'b': *})

.. versionadded:: 0.14.1
"""

from __future__ import annotations

from optree.ops import treespec_defaultdict as defaultdict
from optree.ops import treespec_deque as deque
from optree.ops import treespec_dict as dict  # pylint: disable=redefined-builtin
from optree.ops import treespec_from_collection as from_collection
from optree.ops import treespec_leaf as leaf
from optree.ops import treespec_list as list  # pylint: disable=redefined-builtin
from optree.ops import treespec_namedtuple as namedtuple
from optree.ops import treespec_none as none
from optree.ops import treespec_ordereddict as ordereddict
from optree.ops import treespec_structseq as structseq
from optree.ops import treespec_tuple as tuple  # pylint: disable=redefined-builtin


__all__ = [
    'leaf',
    'none',
    'tuple',
    'list',
    'dict',
    'namedtuple',
    'ordereddict',
    'defaultdict',
    'deque',
    'structseq',
    'from_collection',
]
