#!/usr/bin/env python3
#
#  Syntax: mkdoc.py [-I<path> ..] [.. a list of header files ..]
#
#  Extract documentation from C++ header files to use it in Python bindings
#

import os
import sys
import platform
import re
import textwrap

from clang import cindex
from clang.cindex import CursorKind
from collections import OrderedDict
from threading import Thread, Semaphore
from multiprocessing import cpu_count

RECURSE_LIST = [
    CursorKind.TRANSLATION_UNIT,
    CursorKind.NAMESPACE,
    CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL,
    CursorKind.ENUM_DECL,
    CursorKind.CLASS_TEMPLATE
]

PRINT_LIST = [
    CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL,
    CursorKind.ENUM_DECL,
    CursorKind.ENUM_CONSTANT_DECL,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.FUNCTION_DECL,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.CONVERSION_FUNCTION,
    CursorKind.CXX_METHOD,
    CursorKind.CONSTRUCTOR,
    CursorKind.FIELD_DECL
]

CPP_OPERATORS = {
    '<=': 'le', '>=': 'ge', '==': 'eq', '!=': 'ne', '[]': 'array',
    '+=': 'iadd', '-=': 'isub', '*=': 'imul', '/=': 'idiv', '%=':
    'imod', '&=': 'iand', '|=': 'ior', '^=': 'ixor', '<<=': 'ilshift',
    '>>=': 'irshift', '++': 'inc', '--': 'dec', '<<': 'lshift', '>>':
    'rshift', '&&': 'land', '||': 'lor', '!': 'lnot', '~': 'bnot',
    '&': 'band', '|': 'bor', '+': 'add', '-': 'sub', '*': 'mul', '/':
    'div', '%': 'mod', '<': 'lt', '>': 'gt', '=': 'assign', '()': 'call'
}

CPP_OPERATORS = OrderedDict(
    sorted(CPP_OPERATORS.items(), key=lambda t: -len(t[0])))

job_count = cpu_count()
job_semaphore = Semaphore(job_count)

output = []

def d(s):
    return s.decode('utf8')


def sanitize_name(name):
    name = re.sub(r'type-parameter-0-([0-9]+)', r'T\1', name)
    for k, v in CPP_OPERATORS.items():
        name = name.replace('operator%s' % k, 'operator_%s' % v)
    name = re.sub('<.*>', '', name)
    name = ''.join([ch if ch.isalnum() else '_' for ch in name])
    name = re.sub('_$', '', re.sub('_+', '_', name))
    return '__doc_' + name


def process_comment(comment):
    result = ''

    # Remove C++ comment syntax
    leading_spaces = float('inf')
    for s in comment.expandtabs(tabsize=4).splitlines():
        s = s.strip()
        if s.startswith('/*'):
            s = s[2:].lstrip('*')
        elif s.endswith('*/'):
            s = s[:-2].rstrip('*')
        elif s.startswith('///'):
            s = s[3:]
        if s.startswith('*'):
            s = s[1:]
        if len(s) > 0:
            leading_spaces = min(leading_spaces, len(s) - len(s.lstrip()))
        result += s + '\n'

    if leading_spaces != float('inf'):
        result2 = ""
        for s in result.splitlines():
            result2 += s[leading_spaces:] + '\n'
        result = result2

    # Doxygen tags
    cpp_group = '([\w:]+)'
    param_group = '([\[\w:\]]+)'

    s = result
    s = re.sub(r'\\c\s+%s' % cpp_group, r'``\1``', s)
    s = re.sub(r'\\a\s+%s' % cpp_group, r'*\1*', s)
    s = re.sub(r'\\e\s+%s' % cpp_group, r'*\1*', s)
    s = re.sub(r'\\em\s+%s' % cpp_group, r'*\1*', s)
    s = re.sub(r'\\b\s+%s' % cpp_group, r'**\1**', s)
    s = re.sub(r'\\ingroup\s+%s' % cpp_group, r'', s)
    s = re.sub(r'\\param%s?\s+%s' % (param_group, cpp_group),
               r'\n\n$Parameter ``\2``:\n\n', s)
    s = re.sub(r'\\tparam%s?\s+%s' % (param_group, cpp_group),
               r'\n\n$Template parameter ``\2``:\n\n', s)

    for in_, out_ in {
        'return': 'Returns',
        'author': 'Author',
        'authors': 'Authors',
        'copyright': 'Copyright',
        'date': 'Date',
        'remark': 'Remark',
        'sa': 'See also',
        'see': 'See also',
        'extends': 'Extends',
        'throw': 'Throws',
        'throws': 'Throws'
    }.items():
        s = re.sub(r'\\%s\s*' % in_, r'\n\n$%s:\n\n' % out_, s)

    s = re.sub(r'\\details\s*', r'\n\n', s)
    s = re.sub(r'\\brief\s*', r'', s)
    s = re.sub(r'\\short\s*', r'', s)
    s = re.sub(r'\\ref\s*', r'', s)

    s = re.sub(r'\\code\s?(.*?)\s?\\endcode',
               r"```\n\1\n```\n", s, flags=re.DOTALL)

    # HTML/TeX tags
    s = re.sub(r'<tt>(.*?)</tt>', r'``\1``', s, flags=re.DOTALL)
    s = re.sub(r'<pre>(.*?)</pre>', r"```\n\1\n```\n", s, flags=re.DOTALL)
    s = re.sub(r'<em>(.*?)</em>', r'*\1*', s, flags=re.DOTALL)
    s = re.sub(r'<b>(.*?)</b>', r'**\1**', s, flags=re.DOTALL)
    s = re.sub(r'\\f\$(.*?)\\f\$', r'$\1$', s, flags=re.DOTALL)
    s = re.sub(r'<li>', r'\n\n* ', s)
    s = re.sub(r'</?ul>', r'', s)
    s = re.sub(r'</li>', r'\n\n', s)

    s = s.replace('``true``', '``True``')
    s = s.replace('``false``', '``False``')

    # Re-flow text
    wrapper = textwrap.TextWrapper()
    wrapper.expand_tabs = True
    wrapper.replace_whitespace = True
    wrapper.drop_whitespace = True
    wrapper.width = 70
    wrapper.initial_indent = wrapper.subsequent_indent = ''

    result = ''
    in_code_segment = False
    for x in re.split(r'(```)', s):
        if x == '```':
            if not in_code_segment:
                result += '```\n'
            else:
                result += '\n```\n\n'
            in_code_segment = not in_code_segment
        elif in_code_segment:
            result += x.strip()
        else:
            for y in re.split(r'(?: *\n *){2,}', x):
                wrapped = wrapper.fill(re.sub(r'\s+', ' ', y).strip())
                if len(wrapped) > 0 and wrapped[0] == '$':
                    result += wrapped[1:] + '\n'
                    wrapper.initial_indent = \
                        wrapper.subsequent_indent = ' ' * 4
                else:
                    if len(wrapped) > 0:
                        result += wrapped + '\n\n'
                    wrapper.initial_indent = wrapper.subsequent_indent = ''
    return result.rstrip().lstrip('\n')


def extract(filename, node, prefix):
    if not (node.location.file is None or
            os.path.samefile(d(node.location.file.name), filename)):
        return 0
    if node.kind in RECURSE_LIST:
        sub_prefix = prefix
        if node.kind != CursorKind.TRANSLATION_UNIT:
            if len(sub_prefix) > 0:
                sub_prefix += '_'
            sub_prefix += d(node.spelling)
        for i in node.get_children():
            extract(filename, i, sub_prefix)
    if node.kind in PRINT_LIST:
        comment = d(node.raw_comment) if node.raw_comment is not None else ''
        comment = process_comment(comment)
        sub_prefix = prefix
        if len(sub_prefix) > 0:
            sub_prefix += '_'
        if len(node.spelling) > 0:
            name = sanitize_name(sub_prefix + d(node.spelling))
            global output
            output.append((name, filename, comment))


class ExtractionThread(Thread):
    def __init__(self, filename, parameters):
        Thread.__init__(self)
        self.filename = filename
        self.parameters = parameters
        job_semaphore.acquire()

    def run(self):
        print('Processing "%s" ..' % self.filename, file=sys.stderr)
        try:
            index = cindex.Index(
                cindex.conf.lib.clang_createIndex(False, True))
            tu = index.parse(self.filename, self.parameters)
            extract(self.filename, tu.cursor, '')
        finally:
            job_semaphore.release()

if __name__ == '__main__':
    parameters = ['-x', 'c++', '-std=c++11']
    filenames = []

    if platform.system() == 'Darwin':
        dev_path = '/Applications/Xcode.app/Contents/Developer/'
        lib_dir = dev_path + 'Toolchains/XcodeDefault.xctoolchain/usr/lib/'
        sdk_dir = dev_path + 'Platforms/MacOSX.platform/Developer/SDKs'
        libclang = lib_dir + 'libclang.dylib'

        if os.path.exists(libclang):
            cindex.Config.set_library_path(os.path.dirname(libclang))

        if os.path.exists(sdk_dir):
            sysroot_dir = os.path.join(sdk_dir, next(os.walk(sdk_dir))[1][0])
            parameters.append('-isysroot')
            parameters.append(sysroot_dir)

    for item in sys.argv[1:]:
        if item.startswith('-'):
            parameters.append(item)
        else:
            filenames.append(item)

    if len(filenames) == 0:
        print('Syntax: %s [.. a list of header files ..]' % sys.argv[0])
        exit(-1)

    print('''/*
  This file contains docstrings for the Python bindings.
  Do not edit! These were automatically extracted by mkdoc.py
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
''')

    output.clear()
    for filename in filenames:
        thr = ExtractionThread(filename, parameters)
        thr.start()

    print('Waiting for jobs to finish ..', file=sys.stderr)
    for i in range(job_count):
        job_semaphore.acquire()

    name_ctr = 1
    name_prev = None
    for name, _, comment in list(sorted(output, key=lambda x: (x[0], x[1]))):
        if name == name_prev:
            name_ctr += 1
            name = name + "_%i" % name_ctr
        else:
            name_prev = name
            name_ctr = 1
        print('\nstatic const char *%s =%sR"doc(%s)doc";' %
              (name, '\n' if '\n' in comment else ' ', comment))

    print('''
#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
''')
