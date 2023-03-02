
# Generates a C++ header files embedding the original input as a string literal

import argparse
import pathlib
from datetime import datetime

arg_parser = argparse.ArgumentParser(
    description='Converts source files to C++ string literals', allow_abbrev=False)

arg_parser.add_argument('-i', '--input', required=True,
                        help='Input source file')

arg_parser.add_argument('-o', '--output', required=True,
                        help='Name of the generated header file')

args = arg_parser.parse_args()

# msvc string literal maximum length 16380
# https://docs.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2026?view=msvc-170
MAX_STRING_LITERAL = 16000
# https://docs.microsoft.com/en-us/cpp/c-language/maximum-string-length?view=msvc-170
MAX_STRING_CONCATENATED = 65535

with open(args.input, 'r') as fin:
    with open(args.output, 'w') as fout:
        literal_name = f'{pathlib.Path(args.input).stem}_cu'
        fout.write(f'// Generated from "{args.input}"\n')
        fout.write(f'// {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        fout.write('namespace nvfuser_resources {\n\n')
        fout.write(f'constexpr const char* {literal_name} = R"(\n')
        accumulated_chars = 0
        accumulated_chars_per_literal = 0
        for line in fin:
            accumulated_chars = accumulated_chars + len(line) + 1
            accumulated_chars_per_literal = accumulated_chars_per_literal + len(line) + 1
            if accumulated_chars_per_literal >= MAX_STRING_LITERAL:
                fout.write(')"\n')
                fout.write('R"(\n')
                fout.write(line)
                accumulated_chars_per_literal = len(line) + 1
            else:
                fout.write(line)
        fout.write(')";\n')
        fout.write('\n} // namespace nvfuser_resources\n')
        if accumulated_chars >= MAX_STRING_CONCATENATED:
            raise Exception("runtime header file exceeds size limit of 65535 for MSVC")
