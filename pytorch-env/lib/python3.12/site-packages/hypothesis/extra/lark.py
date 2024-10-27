# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
----------------
hypothesis[lark]
----------------

This extra can be used to generate strings matching any context-free grammar,
using the `Lark parser library <https://github.com/lark-parser/lark>`_.

It currently only supports Lark's native EBNF syntax, but we plan to extend
this to support other common syntaxes such as ANTLR and :rfc:`5234` ABNF.
Lark already `supports loading grammars
<https://lark-parser.readthedocs.io/en/latest/nearley.html>`_
from `nearley.js <https://nearley.js.org/>`_, so you may not have to write
your own at all.
"""

from inspect import signature
from typing import Optional

import lark
from lark.grammar import NonTerminal, Terminal

from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture.utils import calc_label_from_name
from hypothesis.internal.validation import check_type
from hypothesis.strategies._internal.regex import IncompatibleWithAlphabet
from hypothesis.strategies._internal.utils import cacheable, defines_strategy

__all__ = ["from_lark"]


def get_terminal_names(terminals, rules, ignore_names):
    """Get names of all terminals in the grammar.

    The arguments are the results of calling ``Lark.grammar.compile()``,
    so you would think that the ``terminals`` and ``ignore_names`` would
    have it all... but they omit terminals created with ``@declare``,
    which appear only in the expansion(s) of nonterminals.
    """
    names = {t.name for t in terminals} | set(ignore_names)
    for rule in rules:
        names |= {t.name for t in rule.expansion if isinstance(t, Terminal)}
    return names


class LarkStrategy(st.SearchStrategy):
    """Low-level strategy implementation wrapping a Lark grammar.

    See ``from_lark`` for details.
    """

    def __init__(self, grammar, start, explicit, alphabet):
        assert isinstance(grammar, lark.lark.Lark)
        if start is None:
            start = grammar.options.start
        if not isinstance(start, list):
            start = [start]
        self.grammar = grammar

        # This is a total hack, but working around the changes is a nicer user
        # experience than breaking for anyone who doesn't instantly update their
        # installation of Lark alongside Hypothesis.
        compile_args = signature(grammar.grammar.compile).parameters
        if "terminals_to_keep" in compile_args:
            terminals, rules, ignore_names = grammar.grammar.compile(start, ())
        elif "start" in compile_args:  # pragma: no cover
            # Support lark <= 0.10.0, without the terminals_to_keep argument.
            terminals, rules, ignore_names = grammar.grammar.compile(start)
        else:  # pragma: no cover
            # This branch is to support lark <= 0.7.1, without the start argument.
            terminals, rules, ignore_names = grammar.grammar.compile()

        self.names_to_symbols = {}

        for r in rules:
            t = r.origin
            self.names_to_symbols[t.name] = t

        disallowed = set()
        self.terminal_strategies = {}
        for t in terminals:
            self.names_to_symbols[t.name] = Terminal(t.name)
            s = st.from_regex(t.pattern.to_regexp(), fullmatch=True, alphabet=alphabet)
            try:
                s.validate()
            except IncompatibleWithAlphabet:
                disallowed.add(t.name)
            else:
                self.terminal_strategies[t.name] = s

        self.ignored_symbols = tuple(self.names_to_symbols[n] for n in ignore_names)

        all_terminals = get_terminal_names(terminals, rules, ignore_names)
        if unknown_explicit := sorted(set(explicit) - all_terminals):
            raise InvalidArgument(
                "The following arguments were passed as explicit_strategies, but "
                f"there is no {unknown_explicit} terminal production in this grammar."
            )
        if missing_declared := sorted(
            all_terminals - {t.name for t in terminals} - set(explicit)
        ):
            raise InvalidArgument(
                f"Undefined terminal{'s' * (len(missing_declared) > 1)} "
                f"{sorted(missing_declared)!r}. Generation does not currently "
                "support use of %declare unless you pass `explicit`, a dict of "
                f"names-to-strategies, such as `{{{missing_declared[0]!r}: "
                'st.just("")}}`'
            )
        self.terminal_strategies.update(explicit)

        nonterminals = {}

        for rule in rules:
            if disallowed.isdisjoint(r.name for r in rule.expansion):
                nonterminals.setdefault(rule.origin.name, []).append(
                    tuple(rule.expansion)
                )

        allowed_rules = {*self.terminal_strategies, *nonterminals}
        while dict(nonterminals) != (
            nonterminals := {
                k: clean
                for k, v in nonterminals.items()
                if (clean := [x for x in v if all(r.name in allowed_rules for r in x)])
            }
        ):
            allowed_rules = {*self.terminal_strategies, *nonterminals}

        if set(start).isdisjoint(allowed_rules):
            raise InvalidArgument(
                f"No start rule {tuple(start)} is allowed by {alphabet=}"
            )
        self.start = st.sampled_from(
            [self.names_to_symbols[s] for s in start if s in allowed_rules]
        )

        self.nonterminal_strategies = {
            k: st.sampled_from(sorted(v, key=len)) for k, v in nonterminals.items()
        }

        self.__rule_labels = {}

    def do_draw(self, data):
        state = []
        start = data.draw(self.start)
        self.draw_symbol(data, start, state)
        return "".join(state)

    def rule_label(self, name):
        try:
            return self.__rule_labels[name]
        except KeyError:
            return self.__rule_labels.setdefault(
                name, calc_label_from_name(f"LARK:{name}")
            )

    def draw_symbol(self, data, symbol, draw_state):
        if isinstance(symbol, Terminal):
            strategy = self.terminal_strategies[symbol.name]
            draw_state.append(data.draw(strategy))
        else:
            assert isinstance(symbol, NonTerminal)
            data.start_example(self.rule_label(symbol.name))
            expansion = data.draw(self.nonterminal_strategies[symbol.name])
            for e in expansion:
                self.draw_symbol(data, e, draw_state)
                self.gen_ignore(data, draw_state)
            data.stop_example()

    def gen_ignore(self, data, draw_state):
        if self.ignored_symbols and data.draw_boolean(1 / 4):
            emit = data.draw(st.sampled_from(self.ignored_symbols))
            self.draw_symbol(data, emit, draw_state)

    def calc_has_reusable_values(self, recur):
        return True


def check_explicit(name):
    def inner(value):
        check_type(str, value, "value drawn from " + name)
        return value

    return inner


@cacheable
@defines_strategy(force_reusable_values=True)
def from_lark(
    grammar: lark.lark.Lark,
    *,
    start: Optional[str] = None,
    explicit: Optional[dict[str, st.SearchStrategy[str]]] = None,
    alphabet: st.SearchStrategy[str] = st.characters(codec="utf-8"),
) -> st.SearchStrategy[str]:
    """A strategy for strings accepted by the given context-free grammar.

    ``grammar`` must be a ``Lark`` object, which wraps an EBNF specification.
    The Lark EBNF grammar reference can be found
    `here <https://lark-parser.readthedocs.io/en/latest/grammar.html>`_.

    ``from_lark`` will automatically generate strings matching the
    nonterminal ``start`` symbol in the grammar, which was supplied as an
    argument to the Lark class.  To generate strings matching a different
    symbol, including terminals, you can override this by passing the
    ``start`` argument to ``from_lark``.  Note that Lark may remove unreachable
    productions when the grammar is compiled, so you should probably pass the
    same value for ``start`` to both.

    Currently ``from_lark`` does not support grammars that need custom lexing.
    Any lexers will be ignored, and any undefined terminals from the use of
    ``%declare`` will result in generation errors.  To define strategies for
    such terminals, pass a dictionary mapping their name to a corresponding
    strategy as the ``explicit`` argument.

    The :pypi:`hypothesmith` project includes a strategy for Python source,
    based on a grammar and careful post-processing.
    """
    check_type(lark.lark.Lark, grammar, "grammar")
    if explicit is None:
        explicit = {}
    else:
        check_type(dict, explicit, "explicit")
        explicit = {
            k: v.map(check_explicit(f"explicit[{k!r}]={v!r}"))
            for k, v in explicit.items()
        }
    return LarkStrategy(grammar, start, explicit, alphabet)
