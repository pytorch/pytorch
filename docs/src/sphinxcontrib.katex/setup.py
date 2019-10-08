# -*- coding: utf-8 -*-

import io
from setuptools import setup, find_packages

__version__ = 'unknown'
with open('sphinxcontrib/katex.py') as katex_py:
    for line in katex_py:
        if line.startswith('__version__'):
            exec(line)
            break


def readfile(filename):
    with io.open(filename, encoding="utf-8") as stream:
        return stream.read().split("\n")


readme = readfile("README.rst")[3:]  # skip title
requires = readfile("requirements.txt")

setup(
    name='sphinxcontrib-katex',
    version=__version__,
    url='https://github.com/hagenw/sphinxcontrib-katex',
    download_url='https://pypi.python.org/pypi/sphinxcontrib-katex',
    license='MIT',
    author='Hagen Wierstorf',
    author_email='hagenw@posteo.de',
    description=readme[0],
    long_description="\n".join(readme[2:]),
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Documentation',
        'Topic :: Utilities',
    ],
    platforms='any',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires,
    namespace_packages=['sphinxcontrib'],
)
