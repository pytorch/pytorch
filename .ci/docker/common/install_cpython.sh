#!/bin/bash
# Script used only in CD pipeline
set -uex -o pipefail

PYTHON_DOWNLOAD_URL=https://www.python.org/ftp/python
GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py

# Python versions to be installed in /opt/$VERSION_NO
CPYTHON_VERSIONS=${CPYTHON_VERSIONS:-"3.9.0 3.10.1 3.11.0 3.12.0 3.13.0 3.13.0t 3.14.0 3.14.0t"}

function check_var {
    if [ -z "$1" ]; then
        echo "required variable not defined"
        exit 1
    fi
}

function do_cpython_build {
    local py_ver=$1
    local py_folder=$2
    check_var $py_ver
    check_var $py_folder
    tar -xzf Python-$py_ver.tgz

    local additional_flags=""
    if [[ "$py_ver" == *"t" ]]; then
        additional_flags=" --disable-gil"
    fi

    pushd $py_folder

    local prefix="/opt/_internal/cpython-${py_ver}"
    mkdir -p ${prefix}/lib
    if [[ -n $(which patchelf) ]]; then
        local shared_flags="--enable-shared"
    else
        local shared_flags="--disable-shared"
    fi
    if [[ -z  "${WITH_OPENSSL+x}" ]]; then
        local openssl_flags=""
    else
        local openssl_flags="--with-openssl=${WITH_OPENSSL} --with-openssl-rpath=auto"
    fi



    # -Wformat added for https://bugs.python.org/issue17547 on Python 2.6
    CFLAGS="-Wformat" ./configure --prefix=${prefix} ${openssl_flags} ${shared_flags} ${additional_flags} > /dev/null

    make -j40 > /dev/null
    make install > /dev/null

    if [[ "${shared_flags}" == "--enable-shared" ]]; then
        patchelf --set-rpath '$ORIGIN/../lib' ${prefix}/bin/python3
    fi

    popd
    rm -rf $py_folder
    # Some python's install as bin/python3. Make them available as
    # bin/python.
    if [ -e ${prefix}/bin/python3 ]; then
        ln -s python3 ${prefix}/bin/python
    fi
    ${prefix}/bin/python get-pip.py
    if [ -e ${prefix}/bin/pip3 ] && [ ! -e ${prefix}/bin/pip ]; then
        ln -s pip3 ${prefix}/bin/pip
    fi
    # install setuptools since python 3.12 is required to use distutils
    # packaging is needed to create symlink since wheel no longer provides needed information
    ${prefix}/bin/pip install packaging==25.0 wheel==0.45.1 setuptools==80.9.0
    local abi_tag=$(${prefix}/bin/python -c "from packaging.tags import interpreter_name, interpreter_version; import sysconfig ; from sysconfig import get_config_var; print('{0}{1}-{0}{1}{2}'.format(interpreter_name(), interpreter_version(), 't' if sysconfig.get_config_var('Py_GIL_DISABLED') else ''))")
    ln -sf ${prefix} /opt/python/${abi_tag}
}

function build_cpython {
    local py_ver=$1
    check_var $py_ver
    local py_suffix=$py_ver
    local py_folder=$py_ver

    # Special handling for nogil
    if [[ "${py_ver}" == *"t" ]]; then
        py_suffix=${py_ver::-1}
        py_folder=$py_suffix
    fi
    # Update to rc2 due to https://github.com/python/cpython/commit/c72699086fe4
    if [ "$py_suffix" == "3.14.0" ]; then
        py_suffix="3.14.0rc2"
    fi
    wget -q $PYTHON_DOWNLOAD_URL/$py_folder/Python-$py_suffix.tgz -O Python-$py_ver.tgz
    do_cpython_build $py_ver Python-$py_suffix

    rm -f Python-$py_ver.tgz
}

function build_cpythons {
    check_var $GET_PIP_URL
    curl -sLO $GET_PIP_URL
    for py_ver in $@; do
        build_cpython $py_ver
    done
    rm -f get-pip.py
}

mkdir -p /opt/python
mkdir -p /opt/_internal
build_cpythons $CPYTHON_VERSIONS
