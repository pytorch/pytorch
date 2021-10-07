import json
import tempfile
from pathlib import Path
import os


def merge_ltc_compile_commands():
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    pytorch_build_path = base_dir.parent / Path('build')
    src1 = pytorch_build_path / Path('compile_commands.json')
    print(src1)
    build_ninja = list(path for path in Path('.').rglob('build.ninja'))[0]
    print(build_ninja)

    temp_name = next(tempfile._get_candidate_names())
    src2 = Path('/tmp') / temp_name
    print(src2)
    os.system(f"ninja -f {build_ninja} -t compdb > {src2}")

    out1 = base_dir.parent / Path('compile_commands.json')
    print(out1)

    def merge(src1, src2, out1):

        with open(src1) as db1:
            jdb1 = json.load(db1)

        with open(src2) as db2:
            jdb2 = json.load(db2)

        with open(out1, "w") as odb1:
            json.dump(jdb1 + jdb2, odb1)

    merge(src1, src2, out1)


if __name__ == '__main__':
    merge_ltc_compile_commands()
