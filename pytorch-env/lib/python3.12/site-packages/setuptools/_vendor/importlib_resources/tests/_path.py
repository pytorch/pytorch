import pathlib
import functools

from typing import Dict, Union


####
# from jaraco.path 3.4.1

FilesSpec = Dict[str, Union[str, bytes, 'FilesSpec']]  # type: ignore


def build(spec: FilesSpec, prefix=pathlib.Path()):
    """
    Build a set of files/directories, as described by the spec.

    Each key represents a pathname, and the value represents
    the content. Content may be a nested directory.

    >>> spec = {
    ...     'README.txt': "A README file",
    ...     "foo": {
    ...         "__init__.py": "",
    ...         "bar": {
    ...             "__init__.py": "",
    ...         },
    ...         "baz.py": "# Some code",
    ...     }
    ... }
    >>> target = getfixture('tmp_path')
    >>> build(spec, target)
    >>> target.joinpath('foo/baz.py').read_text(encoding='utf-8')
    '# Some code'
    """
    for name, contents in spec.items():
        create(contents, pathlib.Path(prefix) / name)


@functools.singledispatch
def create(content: Union[str, bytes, FilesSpec], path):
    path.mkdir(exist_ok=True)
    build(content, prefix=path)  # type: ignore


@create.register
def _(content: bytes, path):
    path.write_bytes(content)


@create.register
def _(content: str, path):
    path.write_text(content, encoding='utf-8')


# end from jaraco.path
####
