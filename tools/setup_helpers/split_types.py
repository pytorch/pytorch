import os
import sys

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


def get_gen_path_prefix(file_name):
    gen_name_prefix = file_name[len('torch/csrc/'):].replace('/', '_').replace('.cpp', '')
    gen_path_prefix = os.path.join(generated_dir, gen_name_prefix)
    return gen_path_prefix


def split_types_ninja(file_name, w):
    gen_path_prefix = get_gen_path_prefix(file_name)
    to_build = [gen_path_prefix + t + '.cpp' for t in types]
    myself = 'tools/setup_helpers/split_types.py'
    cmd = "{} {} '{}'".format(sys.executable, myself, file_name)
    w.writer.build(
        to_build, 'do_cmd', [file_name, myself],
        variables={
            'cmd': cmd,
        })
    return to_build


def split_types(file_name, ninja_global):
    # when ninja is enabled we just generate the build rule here
    if ninja_global is not None:
        return split_types_ninja(file_name, ninja_global)

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

# when called from ninja
if __name__ == '__main__':
    file_name = sys.argv[1].strip("'")
    split_types(file_name, None)
