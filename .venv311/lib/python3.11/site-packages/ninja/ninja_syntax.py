#!/usr/bin/python

# Copyright 2011 Google Inc. All Rights Reserved.
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

"""Python module for generating .ninja files.

Note that this is emphatically not a required piece of Ninja; it's
just a helpful utility for build-file-generation systems that already
use Python.
"""

import re
import textwrap
from io import TextIOWrapper
from typing import Dict, List, Match, Optional, Tuple, Union

def escape_path(word: str) -> str:
    return word.replace('$ ', '$$ ').replace(' ', '$ ').replace(':', '$:')

class Writer(object):
    def __init__(self, output: TextIOWrapper, width: int = 78) -> None:
        self.output = output
        self.width = width

    def newline(self) -> None:
        self.output.write('\n')

    def comment(self, text: str) -> None:
        for line in textwrap.wrap(text, self.width - 2, break_long_words=False,
                                  break_on_hyphens=False):
            self.output.write('# ' + line + '\n')

    def variable(
        self,
        key: str,
        value: Optional[Union[bool, int, float, str, List[str]]],
        indent: int = 0,
    ) -> None:
        if value is None:
            return
        if isinstance(value, list):
            value = ' '.join(filter(None, value))  # Filter out empty strings.
        self._line('%s = %s' % (key, value), indent)

    def pool(self, name: str, depth: int) -> None:
        self._line('pool %s' % name)
        self.variable('depth', depth, indent=1)

    def rule(
        self,
        name: str,
        command: str,
        description: Optional[str] = None,
        depfile: Optional[str] = None,
        generator: bool = False,
        pool: Optional[str] = None,
        restat: bool = False,
        rspfile: Optional[str] = None,
        rspfile_content: Optional[str] = None,
        deps: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self._line('rule %s' % name)
        self.variable('command', command, indent=1)
        if description:
            self.variable('description', description, indent=1)
        if depfile:
            self.variable('depfile', depfile, indent=1)
        if generator:
            self.variable('generator', '1', indent=1)
        if pool:
            self.variable('pool', pool, indent=1)
        if restat:
            self.variable('restat', '1', indent=1)
        if rspfile:
            self.variable('rspfile', rspfile, indent=1)
        if rspfile_content:
            self.variable('rspfile_content', rspfile_content, indent=1)
        if deps:
            self.variable('deps', deps, indent=1)

    def build(
        self,
        outputs: Union[str, List[str]],
        rule: str,
        inputs: Optional[Union[str, List[str]]] = None,
        implicit: Optional[Union[str, List[str]]] = None,
        order_only: Optional[Union[str, List[str]]] = None,
        variables: Optional[
            Union[
                List[Tuple[str, Optional[Union[str, List[str]]]]],
                Dict[str, Optional[Union[str, List[str]]]],
            ]
        ] = None,
        implicit_outputs: Optional[Union[str, List[str]]] = None,
        pool: Optional[str] = None,
        dyndep: Optional[str] = None,
    ) -> List[str]:
        outputs = as_list(outputs)
        out_outputs = [escape_path(x) for x in outputs]
        all_inputs = [escape_path(x) for x in as_list(inputs)]

        if implicit:
            implicit = [escape_path(x) for x in as_list(implicit)]
            all_inputs.append('|')
            all_inputs.extend(implicit)
        if order_only:
            order_only = [escape_path(x) for x in as_list(order_only)]
            all_inputs.append('||')
            all_inputs.extend(order_only)
        if implicit_outputs:
            implicit_outputs = [escape_path(x)
                                for x in as_list(implicit_outputs)]
            out_outputs.append('|')
            out_outputs.extend(implicit_outputs)

        self._line('build %s: %s' % (' '.join(out_outputs),
                                     ' '.join([rule] + all_inputs)))
        if pool is not None:
            self._line('  pool = %s' % pool)
        if dyndep is not None:
            self._line('  dyndep = %s' % dyndep)

        if variables:
            if isinstance(variables, dict):
                iterator = iter(variables.items())
            else:
                iterator = iter(variables)

            for key, val in iterator:
                self.variable(key, val, indent=1)

        return outputs

    def include(self, path: str) -> None:
        self._line('include %s' % path)

    def subninja(self, path: str) -> None:
        self._line('subninja %s' % path)

    def default(self, paths: Union[str, List[str]]) -> None:
        self._line('default %s' % ' '.join(as_list(paths)))

    def _count_dollars_before_index(self, s: str, i: int) -> int:
        """Returns the number of '$' characters right in front of s[i]."""
        dollar_count = 0
        dollar_index = i - 1
        while dollar_index > 0 and s[dollar_index] == '$':
            dollar_count += 1
            dollar_index -= 1
        return dollar_count

    def _line(self, text: str, indent: int = 0) -> None:
        """Write 'text' word-wrapped at self.width characters."""
        leading_space = '  ' * indent
        while len(leading_space) + len(text) > self.width:
            # The text is too wide; wrap if possible.

            # Find the rightmost space that would obey our width constraint and
            # that's not an escaped space.
            available_space = self.width - len(leading_space) - len(' $')
            space = available_space
            while True:
                space = text.rfind(' ', 0, space)
                if (space < 0 or
                    self._count_dollars_before_index(text, space) % 2 == 0):
                    break

            if space < 0:
                # No such space; just use the first unescaped space we can find.
                space = available_space - 1
                while True:
                    space = text.find(' ', space + 1)
                    if (space < 0 or
                        self._count_dollars_before_index(text, space) % 2 == 0):
                        break
            if space < 0:
                # Give up on breaking.
                break

            self.output.write(leading_space + text[0:space] + ' $\n')
            text = text[space+1:]

            # Subsequent lines are continuations, so indent them.
            leading_space = '  ' * (indent+2)

        self.output.write(leading_space + text + '\n')

    def close(self) -> None:
        self.output.close()


def as_list(input: Optional[Union[str, List[str]]]) -> List[str]:
    if input is None:
        return []
    if isinstance(input, list):
        return input
    return [input]


def escape(string: str) -> str:
    """Escape a string such that it can be embedded into a Ninja file without
    further interpretation."""
    assert '\n' not in string, 'Ninja syntax does not allow newlines'
    # We only have one special metacharacter: '$'.
    return string.replace('$', '$$')


def expand(string: str, vars: Dict[str, str], local_vars: Dict[str, str] = {}) -> str:
    """Expand a string containing $vars as Ninja would.

    Note: doesn't handle the full Ninja variable syntax, but it's enough
    to make configure.py's use of it work.
    """
    def exp(m: Match[str]) -> str:
        var = m.group(1)
        if var == '$':
            return '$'
        return local_vars.get(var, vars.get(var, ''))
    return re.sub(r'\$(\$|\w*)', exp, string)
