#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from argparse import Action


class env(Action):
    """
    Get argument values from ``PET_{dest}`` before defaultingto the given ``default`` value.

    For flags (e.g. ``--standalone``)
    use ``check_env`` instead.

    .. note:: when multiple option strings are specified, ``dest`` is
              the longest option string (e.g. for ``"-f", "--foo"``
              the env var to set is ``PET_FOO`` not ``PET_F``)

    Example:
    ::

     parser.add_argument("-f", "--foo", action=env, default="bar")

     ./program                                      -> args.foo="bar"
     ./program -f baz                               -> args.foo="baz"
     ./program --foo baz                            -> args.foo="baz"
     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"
     PET_FOO="env_bar" ./program --foo baz -> args.foo="baz"
     PET_FOO="env_bar" ./program           -> args.foo="env_bar"

     parser.add_argument("-f", "--foo", action=env, required=True)

     ./program                                      -> fails
     ./program -f baz                               -> args.foo="baz"
     PET_FOO="env_bar" ./program           -> args.foo="env_bar"
     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"
    """

    def __init__(self, dest, default=None, required=False, **kwargs) -> None:
        env_name = f"PET_{dest.upper()}"
        default = os.environ.get(env_name, default)

        # ``required`` means that it NEEDS to be present  in the command-line args
        # rather than "this option requires a value (either set explicitly or default"
        # so if we found default then we don't "require" it to be in the command-line
        # so set it to False
        if default:
            required = False

        super().__init__(dest=dest, default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class check_env(Action):
    """
    Check whether the env var ``PET_{dest}`` exists before defaulting to the given ``default`` value.

    Equivalent to
    ``store_true`` argparse built-in action except that the argument can
    be omitted from the commandline if the env var is present and has a
    non-zero value.

    .. note:: it is redundant to pass ``default=True`` for arguments
              that use this action because a flag should be ``True``
              when present and ``False`` otherwise.

    Example:
    ::

     parser.add_argument("--verbose", action=check_env)

     ./program                                  -> args.verbose=False
     ./program --verbose                        -> args.verbose=True
     PET_VERBOSE=1 ./program           -> args.verbose=True
     PET_VERBOSE=0 ./program           -> args.verbose=False
     PET_VERBOSE=0 ./program --verbose -> args.verbose=True

    Anti-pattern (don't do this):

    ::

     parser.add_argument("--verbose", action=check_env, default=True)

     ./program                                  -> args.verbose=True
     ./program --verbose                        -> args.verbose=True
     PET_VERBOSE=1 ./program           -> args.verbose=True
     PET_VERBOSE=0 ./program           -> args.verbose=False

    """

    def __init__(self, dest, default=False, **kwargs) -> None:
        env_name = f"PET_{dest.upper()}"
        default = bool(int(os.environ.get(env_name, "1" if default else "0")))
        super().__init__(dest=dest, const=True, default=default, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)
