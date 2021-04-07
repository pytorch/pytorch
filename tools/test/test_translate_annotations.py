import re
import unittest

from tools.translate_annotations import parse_diff, translate, translate_all

example_regex = r'^(?P<filename>.*?):(?P<lineNumber>\d+):(?P<columnNumber>\d+): (?P<errorCode>\w+\d+) (?P<errorDesc>.*)'

# in the below example patch, note that the filenames differ, so the
# translation should reflect that as well as the line numbers

# $ git clone -b 1.0.2 https://github.com/cscorley/whatthepatch.git
# $ cd whatthepatch/tests/casefiles
# $ git diff --no-index --unified=0 lao tzu
example_diff = '''
diff --git a/lao b/tzu
index 635ef2c..5af88a8 100644
--- a/lao
+++ b/tzu
@@ -1,2 +0,0 @@
-The Way that can be told of is not the eternal Way;
-The name that can be named is not the eternal name.
@@ -4 +2,2 @@ The Nameless is the origin of Heaven and Earth;
-The Named is the mother of all things.
+The named is the mother of all things.
+
@@ -11,0 +11,3 @@ But after they are produced,
+They both may be called deep and profound.
+Deeper and more profound,
+The door of all subtleties!
'''.lstrip()


class TestTranslateAnnotations(unittest.TestCase):
    maxDiff = None

    def test_parse_diff(self) -> None:
        self.assertEqual(
            parse_diff(example_diff),
            {
                'old_filename': 'lao',
                'hunks': [
                    {
                        'old_start': 1,
                        'old_count': 2,
                        'new_start': 0,
                        'new_count': 0,
                    },
                    {
                        'old_start': 4,
                        'old_count': 1,
                        'new_start': 2,
                        'new_count': 2,
                    },
                    {
                        'old_start': 11,
                        'old_count': 0,
                        'new_start': 11,
                        'new_count': 3,
                    },
                ],
            },
        )

    def test_translate_lao_tzu(self) -> None:
        diff = parse_diff(example_diff)
        self.assertEqual(translate(diff, -1), None)  # out of bounds
        self.assertEqual(translate(diff, 0), None)  # we start at 1
        self.assertEqual(translate(diff, 1), 3)
        self.assertEqual(translate(diff, 2), None)
        self.assertEqual(translate(diff, 3), None)
        self.assertEqual(translate(diff, 4), 5)
        self.assertEqual(translate(diff, 5), 6)
        self.assertEqual(translate(diff, 6), 7)
        self.assertEqual(translate(diff, 7), 8)
        self.assertEqual(translate(diff, 8), 9)
        self.assertEqual(translate(diff, 9), 10)
        self.assertEqual(translate(diff, 10), 11)
        self.assertEqual(translate(diff, 11), None)
        self.assertEqual(translate(diff, 12), None)
        self.assertEqual(translate(diff, 13), None)
        self.assertEqual(translate(diff, 14), 12)  # keep going
        self.assertEqual(translate(diff, 15), 13)

    def test_translate_empty(self) -> None:
        diff = parse_diff('--- a/foo')
        self.assertEqual(translate(diff, -1), None)
        self.assertEqual(translate(diff, 0), None)
        self.assertEqual(translate(diff, 1), 1)
        self.assertEqual(translate(diff, 2), 2)
        self.assertEqual(translate(diff, 3), 3)
        self.assertEqual(translate(diff, 4), 4)
        self.assertEqual(translate(diff, 5), 5)
        # etc

    def test_foo(self) -> None:
        self.assertEqual(
            translate_all(
                re.compile(example_regex),
                [
                    'README.md:1:3: R100 make a better title',
                    'README.md:2:1: R200 give a better description',
                ],
            ),
            [
                {
                    'filename': 'README.md',
                    'lineNumber': 1,
                    'columnNumber': 3,
                    'errorCode': 'R100',
                    'errorDesc': 'make a better title',
                },
                {
                    'filename': 'README.md',
                    'lineNumber': 2,
                    'columnNumber': 1,
                    'errorCode': 'R200',
                    'errorDesc': 'give a better description',
                },
            ],
        )


if __name__ == '__main__':
    unittest.main()
