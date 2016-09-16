#! /usr/bin/env python
# encoding: utf-8
"""
Very lightweight docstring to Markdown converter. Modified for use in pytorch


### License

Copyright © 2013 Thomas Gläßle <t_glaessle@gmx.de>

This work  is free. You can  redistribute it and/or modify  it under the
terms of the Do What The Fuck  You Want To Public License, Version 2, as
published by Sam Hocevar. See the COPYING file for more details.

This program  is free software.  It comes  without any warranty,  to the
extent permitted by applicable law.


### Description

Little convenience tool to extract docstrings from a module or class and
convert them to GitHub Flavoured Markdown:

https://help.github.com/articles/github-flavored-markdown

Its purpose is to quickly generate `README.md` files for small projects.


### API

The interface consists of the following functions:

 - `doctrim(docstring)`
 - `doc2md(docstring, title)`

You can run this script from the command line like:

$ doc2md.py [-a] [--no-toc] [-t title] module-name [class-name] > README.md


### Limitations

At the moment  this is suited only  for a very specific use  case. It is
hardly forseeable, if I will decide to improve on it in the near future.

"""
import re
import sys
import inspect

__all__ = ['doctrim', 'doc2md']

doctrim = inspect.cleandoc

def unindent(lines):
    """
    Remove common indentation from string.

    Unlike doctrim there is no special treatment of the first line.

    """
    try:
        # Determine minimum indentation:
        indent = min(len(line) - len(line.lstrip())
                     for line in lines if line)
    except ValueError:
        return lines
    else:
        return [line[indent:] for line in lines]

def code_block(lines, language=''):
    """
    Mark the code segment for syntax highlighting.
    """
    return ['```' + language] + lines + ['```']

def doctest2md(lines):
    """
    Convert the given doctest to a syntax highlighted markdown segment.
    """
    is_only_code = True
    lines = unindent(lines)
    for line in lines:
        if not line.startswith('>>> ') and not line.startswith('... ') and line not in ['>>>', '...']:
            is_only_code = False
            break
    if is_only_code:
        orig = lines
        lines = []
        for line in orig:
            lines.append(line[4:])
    return lines

def doc_code_block(lines, language):
    if language == 'python':
        lines = doctest2md(lines)
    return code_block(lines, language)

_args_section = re.compile('^\s*Args:\s*')
def is_args_check(line):
    return _args_section.match(line)

def args_block(lines):
    out = ['']
    out += ['Parameter | Default | Description']
    out += ['--------- | ------- | -----------']
    for line in lines:
        matches = re.findall(r'\s*([^:]+):\s*(.*?)\s*(Default:\s(.*))?\s*$', line)
        assert matches != None
        name = matches[0][0]
        description = matches[0][1]
        default = matches[0][3]
        out += [name + ' | ' + default + ' | ' + description]
    return out

_returns_section = re.compile('^\s*Returns:\s*')
def is_returns_check(line):
    return _returns_section.match(line)

_image_section = re.compile('^\s*Image:\s*')
def is_image_check(line):
    return _image_section.match(line)

_example_section = re.compile('^\s*Returns:\s*|^\s*Example:\s*')
def is_example_check(line):
    return _example_section.match(line)

_inputshape_section = re.compile('^\s*Returns:\s*|^\s*Input Shape:\s*')
def is_inputshape_check(line):
    return _inputshape_section.match(line)

_outputshape_section = re.compile('^\s*Returns:\s*|^\s*Output Shape:\s*')
def is_outputshape_check(line):
    return _outputshape_section.match(line)


#def get_docargs(line)

_reg_section = re.compile('^#+ ')
def is_heading(line):
    return _reg_section.match(line)

def get_heading(line):
    assert is_heading(line)
    part = line.partition(' ')
    return len(part[0]), part[2]

def make_heading(level, title):
    return '#'*max(level, 1) + ' ' + title

def find_sections(lines):
    """
    Find all section names and return a list with their names.
    """
    sections = []
    for line in lines:
        if is_heading(line):
            sections.append(get_heading(line))
    return sections

def make_toc(sections):
    """
    Generate table of contents for array of section names.
    """
    if not sections:
        return []
    outer = min(n for n,t in sections)
    refs = []
    for ind,sec in sections:
        ref = sec.lower()
        ref = ref.replace(' ', '-')
        ref = ref.replace('?', '')
        refs.append("    "*(ind-outer) + "- [%s](#%s)" % (sec, ref))
    return refs

def _doc2md(lines, shiftlevel=0):
    _doc2md.md = []
    md = _doc2md.md
    _doc2md.is_code = False
    _doc2md.is_args = False
    _doc2md.is_returns = False
    _doc2md.is_inputshape = False
    _doc2md.is_outputshape = False
    def reset():
        if _doc2md.is_code:
            _doc2md.is_code = False
            _doc2md.md += ['']
            _doc2md.md += doc_code_block(code, 'python')

        if _doc2md.is_args:
            _doc2md.is_args = False
            _doc2md.md += args_block(args)

        if _doc2md.is_returns:
            _doc2md.is_returns = False
            _doc2md.md += returns

        _doc2md.is_inputshape = False
        _doc2md.is_outputshape = False

    for line in lines:
        trimmed = line.lstrip()
        if is_args_check(line):
            reset()
            _doc2md.is_args = True
            md += ['']
            md += ['#' * (shiftlevel+2) + ' Constructor Arguments']
            args = []
        elif is_returns_check(line):
            reset()
            _doc2md.is_returns = True
            md += ['']
            md += ['#' * (shiftlevel+2) + ' Returns']
            returns = []
        elif is_example_check(line):
            reset()
            #md += [line]
        elif is_inputshape_check(line):
            reset()
            md += ['']
            md += ['#' * (shiftlevel+2) + ' Input Shape']
            shape = re.findall(r'\s*Input\sShape:\s*(.*?)\s*$', line)
            md += [shape[0].replace('[', '\[').replace(']', '\]').replace('(', '\(').replace(')', '\)')]
        elif is_outputshape_check(line):
            reset()
            md += ['']
            md += ['#' * (shiftlevel+2) + ' Output Shape']
            shape = re.findall(r'\s*Output\sShape:\s*(.*?)\s*$', line)
            md += [shape[0].replace('[', '\[').replace(']', '\]').replace('(', '\(').replace(')', '\)')]
        elif is_image_check(line):
            reset()
            md += ['']
            filename = re.findall(r'\s*Image:\s*(.*?)\s*$', line)
            md += ['<img src="' + filename[0] + '" >']
        elif _doc2md.is_code == False and trimmed.startswith('>>> '):
            reset()
            _doc2md.is_code = True
            code = [line]
        elif shiftlevel != 0 and is_heading(line):
            reset()
            level, title = get_heading(line)
            md += [make_heading(level + shiftlevel, title)]
        elif _doc2md.is_args:
            if line:
                args.append(line)
            else:
                reset()
        elif _doc2md.is_returns:
            if line:
                returns.append(line)
            else:
                reset()
        elif _doc2md.is_code:
            if line:
                code.append(line)
            else:
                reset()
        else:
            reset()
            md += [line]
    reset()
    return md

def doc2md(docstr, title, min_level=1, more_info=False, toc=True):
    """
    Convert a docstring to a markdown text.
    """
    text = doctrim(docstr)
    lines = text.split('\n')

    sections = find_sections(lines)
    if sections:
        level = min(n for n,t in sections) - 1
    else:
        level = 1

    shiftlevel = 0
    if level < min_level:
        shiftlevel = min_level - level
        level = min_level
        sections = [(lev+shiftlevel, tit) for lev,tit in sections]

    md = [
        make_heading(level, title),
        "",
        lines.pop(0),
        ""
    ]
    if toc:
        md += make_toc(sections)
    md += _doc2md(lines, shiftlevel)
    if more_info:
        return (md, sections)
    else:
        return "\n".join(md)

def mod2md(module, title, title_api_section, toc=True):
    """
    Generate markdown document from module, including API section.
    """
    docstr = module.__doc__  or " "

    text = doctrim(docstr)
    lines = text.split('\n')

    sections = find_sections(lines)
    if sections:
        level = min(n for n,t in sections) - 1
    else:
        level = 1

    api_md = []
    api_sec = []
    if title_api_section :
        # sections.append((level+1, title_api_section))
        for name, entry in module.__dict__.items():
            if name[0] != '_' and entry.__doc__:
                #api_sec.append((level+1, name))
                #api_md += ['', '']
                if entry.__doc__:
                    md, sec = doc2md(entry.__doc__, name,
                                     min_level=level+1, more_info=True, toc=False)
                    api_sec += sec
                    api_md += md

    sections += api_sec

    # headline
    md = [
        make_heading(level, title),
        "",
        lines.pop(0),
        ""
    ]

    # main sections
    if toc:
        md += make_toc(sections)
    md += _doc2md(lines)

    if toc:
        md += ['']
        md += make_toc(api_sec)
    md += api_md

    return "\n".join(md)

def main(args=None):
    # parse the program arguments
    import argparse
    parser = argparse.ArgumentParser(
            description='Convert docstrings to markdown.')

    parser.add_argument(
            'module', help='The module containing the docstring.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
            'entry', nargs='?',
            help='Convert only docstring of this entry in module.')
    group.add_argument(
            '-a', '--all', dest='all', action='store_true',
            help='Create an API section with the contents of module.__all__.')
    parser.add_argument(
            '-t', '--title', dest='title',
            help='Document title (default is module name)')
    parser.add_argument(
            '--no-toc', dest='toc', action='store_false', default=True,
            help='Do not automatically generate the TOC')
    args = parser.parse_args(args)

    import importlib
    import inspect
    import os

    def add_path(*pathes):
        for path in reversed(pathes):
            if path not in sys.path:
                sys.path.insert(0, path)

    file = inspect.getfile(inspect.currentframe())
    add_path(os.path.realpath(os.path.abspath(os.path.dirname(file))))
    add_path(os.getcwd())

    mod_name = args.module
    if mod_name.endswith('.py'):
        mod_name = mod_name.rsplit('.py', 1)[0]
    title = args.title or mod_name.replace('_', '-')

    module = importlib.import_module(mod_name)

    if args.all:
        print(mod2md(module, title, 'API', toc=args.toc))

    else:
        if args.entry:
            docstr = module.__dict__[args.entry].__doc__ or ''
        else:
            docstr = module.__doc__ or ''

        print(doc2md(docstr, title, toc=args.toc))

if __name__ == "__main__":
    main()
