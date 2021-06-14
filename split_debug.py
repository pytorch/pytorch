import argparse
import subprocess
import tempfile
import pathlib


def cmd(s, **kwargs):
    s = [str(x) for x in s]
    print("$", " ".join(s))
    subprocess.run(s, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="split a library/executable into a release version / debug version with a .gnu_debuglink between the two")
    parser.add_argument("--elf", help="elf file path", required=True)
    parser.add_argument("--crc-only", action="store_true", help="skip debug fission, only output crc32")
    args = parser.parse_args()

    elf_path = pathlib.Path(args.elf)
    debug_path = elf_path.parent.joinpath(f"{elf_path.name}.dbg")

    if not args.crc_only:
        cmd(["cp", elf_path, debug_path])
        cmd(["strip", "--only-keep-debug", debug_path])
        cmd(["strip", "--strip-debug", elf_path])
        cmd(["objcopy", elf_path.name, f"--add-gnu-debuglink", debug_path.name], cwd=elf_path.parent)


    with tempfile.NamedTemporaryFile() as f:
        cmd(["objcopy", "--dump-section", f".gnu_debuglink={f.name}", elf_path])
        with open(f.name, "rb") as fp:
            crc_bytes = fp.read()
            # get last 4 bytes (the CRC), flip byte order since that's what
            # we already do for CRC names and this keeps it consistent
            crc_bytes = list(reversed(crc_bytes[-4:]))
            crc_bytes = bytes(crc_bytes)
            print(crc_bytes.hex())