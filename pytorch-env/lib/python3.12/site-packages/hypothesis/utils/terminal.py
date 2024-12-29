# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import os


def guess_background_color():
    """Returns one of "dark", "light", or "unknown".

    This is basically just guessing, but better than always guessing "dark"!
    See also https://stackoverflow.com/questions/2507337/ and
    https://unix.stackexchange.com/questions/245378/
    """
    django_colors = os.getenv("DJANGO_COLORS", "")
    for theme in ("light", "dark"):
        if theme in django_colors.split(";"):
            return theme
    # Guessing based on the $COLORFGBG environment variable
    try:
        fg, *_, bg = os.getenv("COLORFGBG").split(";")
    except Exception:
        pass
    else:
        # 0=black, 7=light-grey, 15=white ; we don't interpret other colors
        if fg in ("7", "15") and bg == "0":
            return "dark"
        elif fg == "0" and bg in ("7", "15"):
            return "light"
    # TODO: Guessing based on the xterm control sequence
    return "unknown"
