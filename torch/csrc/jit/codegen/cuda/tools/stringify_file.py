
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

with open(args.input, 'r') as fin:
    with open(args.output, 'w') as fout:
        literal_name = f'{pathlib.Path(args.input).stem}_cu'
        fout.write(f'// Generated from "{args.input}"\n')
        fout.write(f'// {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        fout.write('namespace nvfuser_resources {\n\n')
        fout.write(f'constexpr const char* {literal_name} = R"(\n')
        for line in fin:
            fout.write(line)
        fout.write(')";\n')
        fout.write('\n} // namespace nvfuser_resources\n')
