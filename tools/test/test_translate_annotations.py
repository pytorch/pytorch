import re
import unittest

from tools.translate_annotations import parse_annotation, parse_diff, translate

flake8_regex \
    = r'^(?P<filename>.*?):(?P<lineNumber>\d+):(?P<columnNumber>\d+): (?P<errorCode>\w+\d+) (?P<errorDesc>.*)'
clang_tidy_regex \
    = r'^(?P<filename>.*?):(?P<lineNumber>\d+):(?P<columnNumber>\d+): (?P<errorDesc>.*?) \[(?P<errorCode>.*)\]'

# in the below example patch, note that the filenames differ, so the
# translation should reflect that as well as the line numbers

# $ git clone -b 1.0.2 https://github.com/cscorley/whatthepatch.git
# $ cd whatthepatch/tests/casefiles
# $ git diff --no-index --unified=0 lao tzu
lao_tzu_diff = '''
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

sparser_diff = '''
diff --git a/foo.txt b/bar.txt
index 27a6dad..6fae323 100644
--- a/foo.txt
+++ b/bar.txt
@@ -4,3 +4,2 @@ lines
-lines
-lines
-lines
+A change!!
+Wow
@@ -10,2 +8,0 @@ more lines
-even more
-even more
'''.lstrip()

new_file_diff = '''
diff --git a/torch/csrc/jit/tensorexpr/operators/conv2d.h b/torch/csrc/jit/tensorexpr/operators/conv2d.h
new file mode 100644
index 0000000000..a81eeae346
--- /dev/null
+++ b/torch/csrc/jit/tensorexpr/operators/conv2d.h
@@ -0,0 +1,19 @@
+#pragma once
+
+#include <torch/csrc/jit/tensorexpr/tensor.h>
+
+namespace torch {
+namespace jit {
+namespace tensorexpr {
+
+TORCH_API Tensor* conv2d_depthwise(
+    BufHandle input,
+    BufHandle weight,
+    BufHandle bias,
+    int stride,
+    int pad,
+    int groups);
+
+} // namespace tensorexpr
+} // namespace jit
+} // namespace torch
'''.lstrip()

# fun fact, this example fools VS Code's diff syntax highlighter
haskell_diff = '''
diff --git a/hello.hs b/hello.hs
index ffb8d4ad14..0872ac9db6 100644
--- a/hello.hs
+++ b/hello.hs
@@ -1 +1 @@
--- a/hello/world/example
+main = putStrLn "Hello, world!"
'''.lstrip()


class TestTranslateAnnotations(unittest.TestCase):
    maxDiff = None

    def test_parse_diff_lao_tzu(self) -> None:
        self.assertEqual(
            parse_diff(lao_tzu_diff),
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

    def test_parse_diff_new_file(self) -> None:
        self.assertEqual(
            parse_diff(new_file_diff),
            {
                'old_filename': None,
                'hunks': [
                    {
                        'old_start': 0,
                        'old_count': 0,
                        'new_start': 1,
                        'new_count': 19,
                    },
                ],
            },
        )

    def test_parse_diff_haskell(self) -> None:
        self.assertEqual(
            parse_diff(haskell_diff),
            {
                'old_filename': 'hello.hs',
                'hunks': [
                    {
                        'old_start': 1,
                        'old_count': 1,
                        'new_start': 1,
                        'new_count': 1,
                    },
                ],
            },
        )

    def test_translate_lao_tzu(self) -> None:
        # we'll pretend that this diff represents the file lao being
        # renamed to tzu and also modified
        diff = parse_diff(lao_tzu_diff)

        # line numbers less than 1 are invalid so they map to None
        self.assertEqual(translate(diff, -1), None)
        self.assertEqual(translate(diff, 0), None)

        # the first two lines of the file were removed, so the first
        # line of the new version corresponds to the third line of the
        # original
        self.assertEqual(translate(diff, 1), 3)

        # the second and third lines of the new file were not present in
        # the original version, so they map to None
        self.assertEqual(translate(diff, 2), None)
        self.assertEqual(translate(diff, 3), None)

        # at this point, we have a stretch of lines that are identical
        # in both versions of the file, but the original version of the
        # file had 4 lines before this section whereas the new version
        # has only 3 lines before this section
        self.assertEqual(translate(diff, 4), 5)
        self.assertEqual(translate(diff, 5), 6)
        self.assertEqual(translate(diff, 6), 7)
        self.assertEqual(translate(diff, 7), 8)
        self.assertEqual(translate(diff, 8), 9)
        self.assertEqual(translate(diff, 9), 10)
        self.assertEqual(translate(diff, 10), 11)

        # these three lines were added in the new version of the file,
        # so they map to None
        self.assertEqual(translate(diff, 11), None)
        self.assertEqual(translate(diff, 12), None)
        self.assertEqual(translate(diff, 13), None)

        # the diff doesn't say how long the file is, so we keep mapping
        # line numbers back; since we can look back at the original
        # files, though, we can see that the original is two lines
        # shorter than the new version, which explains why we are
        # subtracting 2 here
        self.assertEqual(translate(diff, 14), 12)
        self.assertEqual(translate(diff, 15), 13)

    def test_translate_empty(self) -> None:
        diff = parse_diff('--- a/foo')

        # again, we start numbering at 1
        self.assertEqual(translate(diff, -1), None)
        self.assertEqual(translate(diff, 0), None)

        # this diff says there are no changes, so all line numbers
        # greater than zero map to themselves
        self.assertEqual(translate(diff, 1), 1)
        self.assertEqual(translate(diff, 2), 2)
        self.assertEqual(translate(diff, 3), 3)
        self.assertEqual(translate(diff, 4), 4)
        self.assertEqual(translate(diff, 5), 5)

    def test_translate_sparser(self) -> None:
        diff = parse_diff(sparser_diff)

        # again, we start numbering at 1
        self.assertEqual(translate(diff, -1), None)
        self.assertEqual(translate(diff, 0), None)

        # the first three lines are unchanged
        self.assertEqual(translate(diff, 1), 1)
        self.assertEqual(translate(diff, 2), 2)
        self.assertEqual(translate(diff, 3), 3)

        # we removed three lines here and added two, so the two lines we
        # added don't map back to anything in the original file
        self.assertEqual(translate(diff, 4), None)
        self.assertEqual(translate(diff, 5), None)

        # we have some unchanged lines here, but in the preceding hunk
        # we removed 3 and added only 2, so we have an offset of 1
        self.assertEqual(translate(diff, 6), 7)
        self.assertEqual(translate(diff, 7), 8)

        # since the unified diff format essentially subtracts 1 from the
        # starting line number when the count is 0, and since we use
        # bisect.bisect_right to decide which hunk to look at, an
        # earlier version of translate had a bug that caused it to get
        # confused because it would look at the second hunk (which lists
        # 8 as its start line number) rather than the first hunk
        self.assertEqual(translate(diff, 8), 9)

        # after the two lines that we removed in the second hunk, we've
        # reduced the total length of the file by 3 lines, so once we
        # reach the end of the diff, we just add 3 to every line number
        self.assertEqual(translate(diff, 9), 12)
        self.assertEqual(translate(diff, 10), 13)
        self.assertEqual(translate(diff, 11), 14)
        self.assertEqual(translate(diff, 12), 15)

    def test_parse_annotation_flake8(self) -> None:
        regex = re.compile(flake8_regex)
        self.assertEqual(
            parse_annotation(regex, 'README.md:1:3: R100 make a better title'),
            {
                'filename': 'README.md',
                'lineNumber': 1,
                'columnNumber': 3,
                'errorCode': 'R100',
                'errorDesc': 'make a better title',
            },
        )

    def test_parse_annotation_clang_tidy(self) -> None:
        regex = re.compile(clang_tidy_regex)
        self.assertEqual(
            parse_annotation(regex, 'README.md:2:1: improve description [R200]'),
            {
                'filename': 'README.md',
                'lineNumber': 2,
                'columnNumber': 1,
                'errorCode': 'R200',
                'errorDesc': 'improve description',
            },
        )


if __name__ == '__main__':
    unittest.main()
