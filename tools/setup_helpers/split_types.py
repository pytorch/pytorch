import os

this_file = os.path.dirname(os.path.abspath(__file__))
generated_dir = os.path.abspath(os.path.join(this_file, '..', '..', 'torch', 'csrc', 'generated'))

line_start = '//generic_include '

types = [
    'Double',
    'Float',
    'Half',
    'Long',
    'Int',
    'Short',
    'Char',
    'Byte'
]

generic_include = '#define {lib}_GENERIC_FILE "{path}"'
generate_include = '#include "{lib}/{lib}Generate{type}Type.h"'


def split_types(file_name):
    assert file_name.startswith('torch/csrc/')
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)

    with open(file_name, 'r') as f:
        lines = f.read().split('\n')

    # Find //generic_include
    for i, l in enumerate(lines):
        if l.startswith(line_start):
            args = l[len(line_start):]
            lib_prefix, generic_file = filter(bool, args.split())
            break
    else:
        raise RuntimeError("generic include not found")

    gen_name_prefix = file_name[len('torch/csrc/'):].replace('/', '_').replace('.cpp', '')
    gen_path_prefix = os.path.join(generated_dir, gen_name_prefix)

    prefix = '\n'.join(lines[:i])
    suffix = '\n'.join(lines[i + 1:])

    to_build = []

    g_include = generic_include.format(lib=lib_prefix, path=generic_file)
    for t in types:
        t_include = generate_include.format(lib=lib_prefix, type=t)
        gen_path = gen_path_prefix + t + '.cpp'
        to_build.append(gen_path)
        with open(gen_path, 'w') as f:
            f.write(prefix + '\n' +
                    g_include + '\n' +
                    t_include + '\n' +
                    suffix)
    return to_build
