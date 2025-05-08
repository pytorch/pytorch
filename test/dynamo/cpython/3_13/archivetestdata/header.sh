#!/bin/bash
INTERPRETER_UNDER_TEST="$1"
if [[ ! -x "${INTERPRETER_UNDER_TEST}" ]]; then
    echo "Interpreter must be the command line argument."
    exit 4
fi
EXECUTABLE="$0" exec "${INTERPRETER_UNDER_TEST}" -E - <<END_OF_PYTHON
import os
import zipfile

namespace = {}

filename = os.environ['EXECUTABLE']
print(f'Opening {filename} as a zipfile.')
with zipfile.ZipFile(filename, mode='r') as exe_zip:
  for file_info in exe_zip.infolist():
    data = exe_zip.read(file_info)
    exec(data, namespace, namespace)
    break  # Only use the first file in the archive.

print('Favorite number in executable:', namespace["FAVORITE_NUMBER"])

### Archive contents will be appended after this file. ###
END_OF_PYTHON
