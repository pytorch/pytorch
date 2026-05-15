# Copyright (c) Meta Platforms, Inc. and affiliates.
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

"""CuTeDSL cache setup shared by Blackwell kernel modules.

CuTeDSL cache bytecode is version-sensitive.  The default global cache
(``/tmp/$USER/cutlass_python_cache``) can be shared across environments with
incompatible ``nvidia-cutlass-dsl`` versions, producing noisy warnings and
lost cache reuse.  Call this helper before importing ``cutlass`` in modules
that compile CuTeDSL kernels.
"""

from __future__ import annotations

import importlib.metadata
import os
import re


def ensure_versioned_cutedsl_cache_dir() -> None:
    """Set a version-scoped CuTeDSL cache directory when the user did not.

    The path format intentionally matches the historical per-module logic:
    ``$TMPDIR/$USER/cutlass_python_cache_<nvidia-cutlass-dsl-version>``.
    If ``CUTE_DSL_CACHE_DIR`` is already set, leave it untouched.
    """
    if "CUTE_DSL_CACHE_DIR" in os.environ:
        return
    try:
        dsl_ver = importlib.metadata.version("nvidia-cutlass-dsl")
    except Exception:
        dsl_ver = "unknown"
    dsl_ver = re.sub(r"[^0-9A-Za-z]+", "_", dsl_ver)
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    tmp = os.environ.get("TMPDIR") or "/tmp"
    os.environ["CUTE_DSL_CACHE_DIR"] = os.path.join(
        tmp, user, f"cutlass_python_cache_{dsl_ver}"
    )
