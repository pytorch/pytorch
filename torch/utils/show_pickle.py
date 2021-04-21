#!/usr/bin/env python3
import sys
import pickle
import pprint
import zipfile
import fnmatch
from typing import IO, BinaryIO, Union


class FakeObject(object):
    def __init__(self, module, name, args):
        self.module = module
        self.name = name
        self.args = args
        # NOTE: We don't distinguish between state never set and state set to None.
        self.state = None

    def __repr__(self):
        state_str = "" if self.state is None else f"(state={self.state!r})"
        return f"{self.module}.{self.name}{self.args!r}{state_str}"

    def __setstate__(self, state):
        self.state = state

    @staticmethod
    def pp_format(printer, obj, stream, indent, allowance, context, level):
        if not obj.args and obj.state is None:
            stream.write(repr(obj))
            return
        if obj.state is None:
            stream.write(f"{obj.module}.{obj.name}")
            printer._format(obj.args, stream, indent + 1, allowance + 1, context, level)
            return
        if not obj.args:
            stream.write(f"{obj.module}.{obj.name}()(state=\n")
            indent += printer._indent_per_level
            stream.write(" " * indent)
            printer._format(obj.state, stream, indent, allowance + 1, context, level + 1)
            stream.write(")")
            return
        raise Exception("Need to implement")


class FakeClass(object):
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.__new__ = self.fake_new  # type: ignore[assignment]

    def __repr__(self):
        return f"{self.module}.{self.name}"

    def __call__(self, *args):
        return FakeObject(self.module, self.name, args)

    def fake_new(self, *args):
        return FakeObject(self.module, self.name, args[1:])


class DumpUnpickler(pickle._Unpickler):  # type: ignore[name-defined]
    def find_class(self, module, name):
        return FakeClass(module, name)

    def persistent_load(self, pid):
        return FakeObject("pers", "obj", (pid,))

    @classmethod
    def dump(cls, in_stream, out_stream):
        value = cls(in_stream).load()
        pprint.pprint(value, stream=out_stream)
        return value


def main(argv, output_stream=None):
    if len(argv) != 2:
        # Don't spam stderr if not using stdout.
        if output_stream is not None:
            raise Exception("Pass argv of length 2.")
        sys.stderr.write("usage: show_pickle PICKLE_FILE\n")
        sys.stderr.write("  PICKLE_FILE can be any of:\n")
        sys.stderr.write("    path to a pickle file\n")
        sys.stderr.write("    file.zip@member.pkl\n")
        sys.stderr.write("    file.zip@*/pattern.*\n")
        sys.stderr.write("      (shell glob pattern for members)\n")
        sys.stderr.write("      (only first match will be shown)\n")
        return 2

    fname = argv[1]
    handle: Union[IO[bytes], BinaryIO]
    if "@" not in fname:
        with open(fname, "rb") as handle:
            DumpUnpickler.dump(handle, output_stream)
    else:
        zfname, mname = fname.split("@", 1)
        with zipfile.ZipFile(zfname) as zf:
            if "*" not in mname:
                with zf.open(mname) as handle:
                    DumpUnpickler.dump(handle, output_stream)
            else:
                found = False
                for info in zf.infolist():
                    if fnmatch.fnmatch(info.filename, mname):
                        with zf.open(info) as handle:
                            DumpUnpickler.dump(handle, output_stream)
                        found = True
                        break
                if not found:
                    raise Exception(f"Could not find member matching {mname} in {zfname}")


if __name__ == "__main__":
    # This hack works on every version of Python I've tested.
    # I've tested on the following versions:
    #   3.7.4
    if True:
        pprint.PrettyPrinter._dispatch[FakeObject.__repr__] = FakeObject.pp_format  # type: ignore[attr-defined]

    sys.exit(main(sys.argv))
