from __future__ import annotations

import os
import sys
import warnings


def which(thefile: str) -> str | None:
    warnings.warn(
        "tools.setup_helpers.which is deprecated and will be removed in a future version. "
        "Use shutil.which instead.",
        FutureWarning,
        stacklevel=2,
    )

    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == "win32":
            exts = os.environ.get("PATHEXT", "").split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None
