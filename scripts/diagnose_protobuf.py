## @package diagnose_protobuf
# Module scripts.diagnose_protobuf
"""Diagnoses the current protobuf situation.

Protocol buffer needs to be properly installed for Caffe2 to work, and
sometimes it is rather tricky. Specifically, we will need to have a
consistent version between C++ and python simultaneously. This is a
convenience script for one to quickly check if this is so on one's local
machine.

Usage:
    [set your environmental variables like PATH and PYTHONPATH]
    python scripts/diagnose_protobuf.py
"""

import os
import re
from subprocess import PIPE, Popen

# Get python protobuf version.
try:
    import google.protobuf

    python_version = google.protobuf.__version__
    python_protobuf_installed = True
except ImportError:
    print("DEBUG: cannot find python protobuf install.")
    python_protobuf_installed = False

if os.name == "nt":
    protoc_name = "protoc.exe"
else:
    protoc_name = "protoc"

try:
    p = Popen([protoc_name, "--version"], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
except:
    print("DEBUG: did not find protoc binary.")
    print("DEBUG: out: " + out)
    print("DEBUG: err: " + err)
    native_protobuf_installed = False
else:
    if p.returncode:
        print("DEBUG: protoc returned a non-zero return code.")
        print("DEBUG: out: " + out)
        print("DEBUG: err: " + err)
        native_protobuf_installed = False
    else:
        tmp = re.search(r"\d\.\d\.\d", out)
        if tmp:
            native_version = tmp.group(0)
            native_protobuf_installed = True
        else:
            print("DEBUG: cannot parse protoc version string.")
            print("DEBUG: out: " + out)
            native_protobuf_installed = False

PYTHON_PROTOBUF_NOT_INSTALLED = """
You have not installed python protobuf. Protobuf is needed to run caffe2. You
can install protobuf via pip or conda (if you are using anaconda python).
"""

NATIVE_PROTOBUF_NOT_INSTALLED = """
You have not installed the protoc binary. Protoc is needed to compile Caffe2
protobuf source files. Depending on the platform you are on, you can install
protobuf via:
    (1) Mac: using homebrew and do brew install protobuf.
    (2) Linux: use apt and do apt-get install libprotobuf-dev
    (3) Windows: install from source, or from the releases here:
        https://github.com/google/protobuf/releases/
"""

VERSION_MISMATCH = """
Your python protobuf is of version {py_ver} but your native protoc version is of
version {native_ver}. This will cause the installation to produce incompatible
protobuf files. This is bad in general - consider installing the same version.
""".format(
    py_ver=python_version, native_ver=native_version
)

# Now, give actual recommendations
if not python_protobuf_installed:
    print(PYTHON_PROTOBUF_NOT_INSTALLED)

if not native_protobuf_installed:
    print(NATIVE_PROTOBUF_NOT_INSTALLED)

if python_protobuf_installed and native_protobuf_installed:
    if python_version != native_version:
        print(VERSION_MISMATCH)
    else:
        print("All looks good.")
