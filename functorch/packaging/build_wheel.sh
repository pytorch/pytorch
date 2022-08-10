#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE=wheel
setup_env 0.2.0
setup_wheel_python
pip_install numpy pyyaml future ninja
pip_install --upgrade setuptools
setup_pip_pytorch_version
python setup.py clean

if [[ "$OSTYPE" == "msys" ]]; then
    "$script_dir/windows/internal/vc_env_helper.bat" python setup.py bdist_wheel
else
    python setup.py bdist_wheel
fi
