import ast
import json
from pathlib import Path
import re


MSG = 'PTA100 expected should come before actual'


def is_literal(node):
    return (
        isinstance(node, ast.Constant) or
        (
            (
                isinstance(node, ast.List) or
                isinstance(node, ast.Set) or
                isinstance(node, ast.Tuple)
            ) and
            all(is_literal(elt) for elt in node.elts)
        ) or
        (
            isinstance(node, ast.Dict) and
            all(is_literal(key) for key in node.keys) and
            all(is_literal(value) for value in node.values)
        )
    )


def classify_arg(node):
    info = {'type': type(node).__name__, 'literal': is_literal(node)}
    if info['type'] == 'Name':
        for keyword in ['actual', 'expected']:
            if re.search(keyword, node.id, re.IGNORECASE):
                info[keyword] = True
    return info


def known_pattern(left, right):
    return right.get('actual') or left.get('expected') or left.get('literal')


def classify_call(node):
    info = {'args': len(node.args)}
    if info['args'] == 2:
        left, right = map(classify_arg, node.args)
        info['left'] = left
        info['right'] = right
        if known_pattern(left, right):
            info['good'] = True
        # check info.get('good') to ignore cases with two literals
        if not info.get('good') and known_pattern(right, left):
            info['flipped'] = True
    return info


class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.usages = {}
        self.problems = []

    def visit_Call(self, node):
        if (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'self' and
                node.func.attr == 'assertEqual'
        ):
            info = classify_call(node)
            pos = node.lineno, node.col_offset
            self.usages[f'{pos[0]}:{pos[1]}'] = info
            if info.get('flipped'):
                self.problems.append(pos)
        self.generic_visit(node)


class Plugin:
    name = f'pytorch-{__name__.replace("_", "-")}'
    version = '0.1.0'
    labels_dir = None

    @classmethod
    def add_options(cls, option_manager):
        option_manager.add_option(
            '--all-assert-equal-usages', metavar='DIR',
            help="Write each file's labeled self.assertEqual usages into DIR.",
        )

    @classmethod
    def parse_options(cls, options):
        cls.labels_dir = options.all_assert_equal_usages

    def __init__(self, tree, filename):
        self._tree = tree
        self._filename = filename

    def run(self):
        visitor = Visitor()
        visitor.visit(self._tree)
        if Plugin.labels_dir:
            # assumes Path.cwd() is pytorch clone dir
            rel = Path(self._filename).resolve().relative_to(Path.cwd())
            path = (Plugin.labels_dir / rel).with_name(f'{rel.name}.json')
            if not path.exists() and visitor.usages:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f'{json.dumps(visitor.usages, indent=2)}\n')
        for line, col in visitor.problems:
            yield line, col, MSG, type(self)
