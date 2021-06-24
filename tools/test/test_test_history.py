import itertools
import re
import shlex
import unittest
from typing import List, Optional

from tools import test_history
from typing_extensions import TypedDict


class Example(TypedDict):
    cmd: str
    args: List[str]
    lines: List[str]


def parse_block(block: List[str]) -> Optional[Example]:
    if block:
        match = re.match(r'^\$ ([^ ]+) (.*)$', block[0])
        if match:
            cmd, first = match.groups()
            args = []
            for i, line in enumerate([first] + block[1:]):
                if line.endswith('\\'):
                    args.append(line[:-1])
                else:
                    args.append(line)
                    break
            return {
                'cmd': cmd,
                'args': shlex.split(''.join(args)),
                'lines': block[i + 1:]
            }
    return None


def parse_description(description: str) -> List[Example]:
    examples: List[Example] = []
    for block in description.split('\n\n'):
        matches = [
            re.match(r'^    (.*)$', line)
            for line in block.splitlines()
        ]
        if all(matches):
            lines = []
            for match in matches:
                assert match
                line, = match.groups()
                lines.append(line)
            example = parse_block(lines)
            if example:
                examples.append(example)
    return examples


class TestTestHistory(unittest.TestCase):
    maxDiff = None

    def test_help_examples(self) -> None:
        examples = parse_description(test_history.description())
        self.assertEqual(len(examples), 3)
        for i, example in enumerate(examples):
            with self.subTest(i=i):
                self.assertTrue(test_history.__file__.endswith(example['cmd']))
                expected = example['lines']
                actual = list(itertools.islice(
                    test_history.run(example['args']),
                    len(expected),
                ))
                self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
