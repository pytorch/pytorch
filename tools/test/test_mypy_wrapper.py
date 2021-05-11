import unittest

from tools import mypy_wrapper


class TestMypyWrapper(unittest.TestCase):
    configs = {
        'foo.ini': {
            'file1.abc',
            'dir2',
            'dir3/file4.xyz',
        },
        'bar/baz.ini': {
            'file1.abc',
            'dir2/dir5/file6.def',
            'dir3/file7.abc',
        },
    }

    trie: mypy_wrapper.Trie = {
        'file1.abc': {None: {'foo.ini', 'bar/baz.ini'}},
        'dir2': {
            None: {'foo.ini'},
            'dir5': {'file6.def': {None: {'bar/baz.ini'}}},
        },
        'dir3': {
            'file4.xyz': {None: {'foo.ini'}},
            'file7.abc': {None: {'bar/baz.ini'}},
        },
    }

    def test_config_files(self) -> None:
        self.assertEqual(mypy_wrapper.config_files().keys(), {
            'mypy.ini',
            'mypy-strict.ini',
        })

    def test_split_path(self) -> None:
        self.assertEqual(mypy_wrapper.split_path('file1.abc'), ['file1.abc'])
        self.assertEqual(
            mypy_wrapper.split_path('dir3/file4.xyz'),
            ['dir3', 'file4.xyz'],
        )
        self.assertEqual(
            mypy_wrapper.split_path('dir2/dir5/file6.def'),
            ['dir2', 'dir5', 'file6.def'],
        )

    def test_make_trie(self) -> None:
        self.assertEqual(mypy_wrapper.make_trie(self.configs), self.trie)

    def test_lookup(self) -> None:
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'file1.abc'),
            {'foo.ini', 'bar/baz.ini'},
        )
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'dir2/dir5/file6.def'),
            {'foo.ini', 'bar/baz.ini'},
        )
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'dir3/file4.xyz'),
            {'foo.ini'},
        )
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'dir3/file7.abc'),
            {'bar/baz.ini'},
        )
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'file8.xyz'),
            set(),
        )
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'dir2/dir9/file10.abc'),
            {'foo.ini'},
        )
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'dir3/file11.abc'),
            set(),
        )

        # non-leaves shouldn't ever be passed to lookup in practice, but
        # still, good to consider/test these cases
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'dir2'),
            {'foo.ini'},
        )
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'dir2/dir5'),
            {'foo.ini'},
        )
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'dir3'),
            set(),
        )
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'dir2/dir9'),
            {'foo.ini'},
        )
        self.assertEqual(
            mypy_wrapper.lookup(self.trie, 'dir4'),
            set(),
        )

    def test_make_plan(self) -> None:
        self.assertEqual(
            mypy_wrapper.make_plan(configs=self.configs, files=[
                'file8.xyz',
                'dir3/file11.abc',
            ]),
            {}
        )
        self.assertEqual(
            mypy_wrapper.make_plan(configs=self.configs, files=[
                'file8.xyz',
                'dir2/dir9/file10.abc',
                'dir3/file4.xyz',
                'dir3/file11.abc',
            ]),
            {
                'foo.ini': ['dir2/dir9/file10.abc', 'dir3/file4.xyz'],
            }
        )
        self.assertEqual(
            mypy_wrapper.make_plan(configs=self.configs, files=[
                'file8.xyz',
                'dir3/file11.abc',
                'dir3/file7.abc',
            ]),
            {
                'bar/baz.ini': ['dir3/file7.abc'],
            }
        )
        self.assertEqual(
            mypy_wrapper.make_plan(configs=self.configs, files=[
                'dir2/dir9/file10.abc',
                'dir2/dir5/file6.def',
                'dir3/file7.abc',
                'file1.abc',
                'dir3/file11.abc',
            ]),
            {
                'foo.ini': [
                    'dir2/dir9/file10.abc',
                    'dir2/dir5/file6.def',
                    'file1.abc',
                ],
                'bar/baz.ini': [
                    'dir2/dir5/file6.def',
                    'dir3/file7.abc',
                    'file1.abc',
                ],
            }
        )


if __name__ == '__main__':
    unittest.main()
