import unittest
import os
import contextlib

try:
    from test.support.warnings_helper import ignore_warnings, check_warnings
except ImportError:
    # older Python versions
    from test.support import ignore_warnings, check_warnings

import importlib_resources as resources

# Since the functional API forwards to Traversable, we only test
# filesystem resources here -- not zip files, namespace packages etc.
# We do test for two kinds of Anchor, though.


class StringAnchorMixin:
    anchor01 = 'importlib_resources.tests.data01'
    anchor02 = 'importlib_resources.tests.data02'


class ModuleAnchorMixin:
    from . import data01 as anchor01
    from . import data02 as anchor02


class FunctionalAPIBase:
    def _gen_resourcetxt_path_parts(self):
        """Yield various names of a text file in anchor02, each in a subTest"""
        for path_parts in (
            ('subdirectory', 'subsubdir', 'resource.txt'),
            ('subdirectory/subsubdir/resource.txt',),
            ('subdirectory/subsubdir', 'resource.txt'),
        ):
            with self.subTest(path_parts=path_parts):
                yield path_parts

    def test_read_text(self):
        self.assertEqual(
            resources.read_text(self.anchor01, 'utf-8.file'),
            'Hello, UTF-8 world!\n',
        )
        self.assertEqual(
            resources.read_text(
                self.anchor02,
                'subdirectory',
                'subsubdir',
                'resource.txt',
                encoding='utf-8',
            ),
            'a resource',
        )
        for path_parts in self._gen_resourcetxt_path_parts():
            self.assertEqual(
                resources.read_text(
                    self.anchor02,
                    *path_parts,
                    encoding='utf-8',
                ),
                'a resource',
            )
        # Use generic OSError, since e.g. attempting to read a directory can
        # fail with PermissionError rather than IsADirectoryError
        with self.assertRaises(OSError):
            resources.read_text(self.anchor01)
        with self.assertRaises(OSError):
            resources.read_text(self.anchor01, 'no-such-file')
        with self.assertRaises(UnicodeDecodeError):
            resources.read_text(self.anchor01, 'utf-16.file')
        self.assertEqual(
            resources.read_text(
                self.anchor01,
                'binary.file',
                encoding='latin1',
            ),
            '\x00\x01\x02\x03',
        )
        self.assertEqual(
            resources.read_text(
                self.anchor01,
                'utf-16.file',
                errors='backslashreplace',
            ),
            'Hello, UTF-16 world!\n'.encode('utf-16').decode(
                errors='backslashreplace',
            ),
        )

    def test_read_binary(self):
        self.assertEqual(
            resources.read_binary(self.anchor01, 'utf-8.file'),
            b'Hello, UTF-8 world!\n',
        )
        for path_parts in self._gen_resourcetxt_path_parts():
            self.assertEqual(
                resources.read_binary(self.anchor02, *path_parts),
                b'a resource',
            )

    def test_open_text(self):
        with resources.open_text(self.anchor01, 'utf-8.file') as f:
            self.assertEqual(f.read(), 'Hello, UTF-8 world!\n')
        for path_parts in self._gen_resourcetxt_path_parts():
            with resources.open_text(
                self.anchor02,
                *path_parts,
                encoding='utf-8',
            ) as f:
                self.assertEqual(f.read(), 'a resource')
        # Use generic OSError, since e.g. attempting to read a directory can
        # fail with PermissionError rather than IsADirectoryError
        with self.assertRaises(OSError):
            resources.open_text(self.anchor01)
        with self.assertRaises(OSError):
            resources.open_text(self.anchor01, 'no-such-file')
        with resources.open_text(self.anchor01, 'utf-16.file') as f:
            with self.assertRaises(UnicodeDecodeError):
                f.read()
        with resources.open_text(
            self.anchor01,
            'binary.file',
            encoding='latin1',
        ) as f:
            self.assertEqual(f.read(), '\x00\x01\x02\x03')
        with resources.open_text(
            self.anchor01,
            'utf-16.file',
            errors='backslashreplace',
        ) as f:
            self.assertEqual(
                f.read(),
                'Hello, UTF-16 world!\n'.encode('utf-16').decode(
                    errors='backslashreplace',
                ),
            )

    def test_open_binary(self):
        with resources.open_binary(self.anchor01, 'utf-8.file') as f:
            self.assertEqual(f.read(), b'Hello, UTF-8 world!\n')
        for path_parts in self._gen_resourcetxt_path_parts():
            with resources.open_binary(
                self.anchor02,
                *path_parts,
            ) as f:
                self.assertEqual(f.read(), b'a resource')

    def test_path(self):
        with resources.path(self.anchor01, 'utf-8.file') as path:
            with open(str(path), encoding='utf-8') as f:
                self.assertEqual(f.read(), 'Hello, UTF-8 world!\n')
        with resources.path(self.anchor01) as path:
            with open(os.path.join(path, 'utf-8.file'), encoding='utf-8') as f:
                self.assertEqual(f.read(), 'Hello, UTF-8 world!\n')

    def test_is_resource(self):
        is_resource = resources.is_resource
        self.assertTrue(is_resource(self.anchor01, 'utf-8.file'))
        self.assertFalse(is_resource(self.anchor01, 'no_such_file'))
        self.assertFalse(is_resource(self.anchor01))
        self.assertFalse(is_resource(self.anchor01, 'subdirectory'))
        for path_parts in self._gen_resourcetxt_path_parts():
            self.assertTrue(is_resource(self.anchor02, *path_parts))

    def test_contents(self):
        with check_warnings((".*contents.*", DeprecationWarning)):
            c = resources.contents(self.anchor01)
        self.assertGreaterEqual(
            set(c),
            {'utf-8.file', 'utf-16.file', 'binary.file', 'subdirectory'},
        )
        with contextlib.ExitStack() as cm:
            cm.enter_context(self.assertRaises(OSError))
            cm.enter_context(check_warnings((".*contents.*", DeprecationWarning)))

            list(resources.contents(self.anchor01, 'utf-8.file'))

        for path_parts in self._gen_resourcetxt_path_parts():
            with contextlib.ExitStack() as cm:
                cm.enter_context(self.assertRaises(OSError))
                cm.enter_context(check_warnings((".*contents.*", DeprecationWarning)))

                list(resources.contents(self.anchor01, *path_parts))
        with check_warnings((".*contents.*", DeprecationWarning)):
            c = resources.contents(self.anchor01, 'subdirectory')
        self.assertGreaterEqual(
            set(c),
            {'binary.file'},
        )

    @ignore_warnings(category=DeprecationWarning)
    def test_common_errors(self):
        for func in (
            resources.read_text,
            resources.read_binary,
            resources.open_text,
            resources.open_binary,
            resources.path,
            resources.is_resource,
            resources.contents,
        ):
            with self.subTest(func=func):
                # Rejecting None anchor
                with self.assertRaises(TypeError):
                    func(None)
                # Rejecting invalid anchor type
                with self.assertRaises((TypeError, AttributeError)):
                    func(1234)
                # Unknown module
                with self.assertRaises(ModuleNotFoundError):
                    func('$missing module$')

    def test_text_errors(self):
        for func in (
            resources.read_text,
            resources.open_text,
        ):
            with self.subTest(func=func):
                # Multiple path arguments need explicit encoding argument.
                with self.assertRaises(TypeError):
                    func(
                        self.anchor02,
                        'subdirectory',
                        'subsubdir',
                        'resource.txt',
                    )


class FunctionalAPITest_StringAnchor(
    unittest.TestCase,
    FunctionalAPIBase,
    StringAnchorMixin,
):
    pass


class FunctionalAPITest_ModuleAnchor(
    unittest.TestCase,
    FunctionalAPIBase,
    ModuleAnchorMixin,
):
    pass
