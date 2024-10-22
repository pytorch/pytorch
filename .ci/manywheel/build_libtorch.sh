#!/usr/bin/env bash
# meant to be called only from the neighboring build.sh and build_cpu.sh scripts

set -e pipefail
SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# Require only one python installation
if [[ -z "$DESIRED_PYTHON" ]]; then
    echo "Need to set DESIRED_PYTHON env variable"
    exit 1
fi
if [[ -n "$BUILD_PYTHONLESS" && -z "$LIBTORCH_VARIANT" ]]; then
    echo "BUILD_PYTHONLESS is set, so need LIBTORCH_VARIANT to also be set"
    echo "LIBTORCH_VARIANT should be one of shared-with-deps shared-without-deps static-with-deps static-without-deps"
    exit 1
fi

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# TODO move this into the Docker images
OS_NAME=`awk -F= '/^NAME/{print $2}' /etc/os-release`
if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
    retry yum install -q -y zip openssl
elif [[ "$OS_NAME" == *"AlmaLinux"* ]]; then
    retry yum install -q -y zip openssl
elif [[ "$OS_NAME" == *"Red Hat Enterprise Linux"* ]]; then
    retry dnf install -q -y zip openssl
elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
    # TODO: Remove this once nvidia package repos are back online
    # Comment out nvidia repositories to prevent them from getting apt-get updated, see https://github.com/pytorch/pytorch/issues/74968
    # shellcheck disable=SC2046
    sed -i 's/.*nvidia.*/# &/' $(find /etc/apt/ -type f -name "*.list")
    retry apt-get update
    retry apt-get -y install zip openssl
fi

# Version: setup.py uses $PYTORCH_BUILD_VERSION.post$PYTORCH_BUILD_NUMBER if
# PYTORCH_BUILD_NUMBER > 1
build_version="$PYTORCH_BUILD_VERSION"
build_number="$PYTORCH_BUILD_NUMBER"
if [[ -n "$OVERRIDE_PACKAGE_VERSION" ]]; then
    # This will be the *exact* version, since build_number<1
    build_version="$OVERRIDE_PACKAGE_VERSION"
    build_number=0
fi
if [[ -z "$build_version" ]]; then
    build_version=1.0.0
fi
if [[ -z "$build_number" ]]; then
    build_number=1
fi
export PYTORCH_BUILD_VERSION=$build_version
export PYTORCH_BUILD_NUMBER=$build_number

export CMAKE_LIBRARY_PATH="/opt/intel/lib:/lib:$CMAKE_LIBRARY_PATH"
export CMAKE_INCLUDE_PATH="/opt/intel/include:$CMAKE_INCLUDE_PATH"

# set OPENSSL_ROOT_DIR=/opt/openssl if it exists
if [[ -e /opt/openssl ]]; then
    export OPENSSL_ROOT_DIR=/opt/openssl
    export CMAKE_INCLUDE_PATH="/opt/openssl/include":$CMAKE_INCLUDE_PATH
fi

# If given a python version like 3.6m or 2.7mu, convert this to the format we
# expect. The binary CI jobs pass in python versions like this; they also only
# ever pass one python version, so we assume that DESIRED_PYTHON is not a list
# in this case
if [[ -n "$DESIRED_PYTHON" && "$DESIRED_PYTHON" != cp* ]]; then
    python_nodot="$(echo $DESIRED_PYTHON | tr -d m.u)"
    DESIRED_PYTHON="cp${python_nodot}-cp${python_nodot}"
fi
pydir="/opt/python/$DESIRED_PYTHON"
export PATH="$pydir/bin:$PATH"

export PATCHELF_BIN=/usr/local/bin/patchelf
patchelf_version=`$PATCHELF_BIN --version`
echo "patchelf version: " $patchelf_version
if [[ "$patchelf_version" == "patchelf 0.9" ]]; then
    echo "Your patchelf version is too old. Please use version >= 0.10."
    exit 1
fi

########################################################
# Compile wheels as well as libtorch
#######################################################
if [[ -z "$PYTORCH_ROOT" ]]; then
    echo "Need to set PYTORCH_ROOT env variable"
    exit 1
fi
pushd "$PYTORCH_ROOT"
python setup.py clean
retry pip install -qr requirements.txt
retry pip install -q numpy==2.0.1

if [[ "$DESIRED_DEVTOOLSET" == *"cxx11-abi"* ]]; then
    export _GLIBCXX_USE_CXX11_ABI=1
else
    export _GLIBCXX_USE_CXX11_ABI=0
fi

if [[ "$DESIRED_CUDA" == *"rocm"* ]]; then
    echo "Calling build_amd.py at $(date)"
    python tools/amd_build/build_amd.py
    # TODO remove this work-around once pytorch sources are updated
    export ROCclr_DIR=/opt/rocm/rocclr/lib/cmake/rocclr
fi

echo "Calling setup.py install at $(date)"

if [[ $LIBTORCH_VARIANT = *"static"* ]]; then
    STATIC_CMAKE_FLAG="-DTORCH_STATIC=1"
fi

(
    set -x

    mkdir -p build

    time CMAKE_ARGS=${CMAKE_ARGS[@]} \
        EXTRA_CAFFE2_CMAKE_FLAGS="${EXTRA_CAFFE2_CMAKE_FLAGS[@]} $STATIC_CMAKE_FLAG" \
        # TODO: Remove this flag once https://github.com/pytorch/pytorch/issues/55952 is closed
        CFLAGS='-Wno-deprecated-declarations' \
        BUILD_LIBTORCH_CPU_WITH_DEBUG=1 \
        python setup.py install

    mkdir -p libtorch/{lib,bin,include,share}

    # Make debug folder separate so it doesn't get zipped up with the rest of
    # libtorch
    mkdir debug

    # Copy over all lib files
    cp -rv build/lib/*                libtorch/lib/
    cp -rv build/lib*/torch/lib/*     libtorch/lib/

    # Copy over all include files
    cp -rv build/include/*            libtorch/include/
    cp -rv build/lib*/torch/include/* libtorch/include/

    # Copy over all of the cmake files
    cp -rv build/lib*/torch/share/*   libtorch/share/

    # Split libtorch into debug / release version
    cp libtorch/lib/libtorch_cpu.so libtorch/lib/libtorch_cpu.so.dbg

    # Keep debug symbols on debug lib
    strip --only-keep-debug libtorch/lib/libtorch_cpu.so.dbg

    # Remove debug info from release lib
    strip --strip-debug libtorch/lib/libtorch_cpu.so

    # Add a debug link to the release lib to the debug lib (debuggers will then
    # search for symbols in a file called libtorch_cpu.so.dbg in some
    # predetermined locations) and embed a CRC32 of the debug library into the .so
    cd libtorch/lib

    objcopy libtorch_cpu.so --add-gnu-debuglink=libtorch_cpu.so.dbg
    cd ../..

    # Move the debug symbols to its own directory so it doesn't get processed /
    # zipped with all the other libraries
    mv libtorch/lib/libtorch_cpu.so.dbg debug/libtorch_cpu.so.dbg

    echo "${PYTORCH_BUILD_VERSION}" > libtorch/build-version
    echo "$(pushd $PYTORCH_ROOT && git rev-parse HEAD)" > libtorch/build-hash

)

if [[ "$DESIRED_DEVTOOLSET" == *"cxx11-abi"* ]]; then
    LIBTORCH_ABI="cxx11-abi-"
else
    LIBTORCH_ABI=
fi

(
    set -x

    mkdir -p /tmp/$LIBTORCH_HOUSE_DIR

    # objcopy installs a CRC32 into libtorch_cpu above so, so add that to the name here
    CRC32=$(objcopy --dump-section .gnu_debuglink=>(tail -c4 | od -t x4 -An | xargs echo) libtorch/lib/libtorch_cpu.so)

    # Zip debug symbols
    zip /tmp/$LIBTORCH_HOUSE_DIR/debug-libtorch-$LIBTORCH_ABI$LIBTORCH_VARIANT-$PYTORCH_BUILD_VERSION-$CRC32.zip debug/libtorch_cpu.so.dbg

    # Zip and copy libtorch
    zip -rq /tmp/$LIBTORCH_HOUSE_DIR/libtorch-$LIBTORCH_ABI$LIBTORCH_VARIANT-$PYTORCH_BUILD_VERSION.zip libtorch
    cp /tmp/$LIBTORCH_HOUSE_DIR/libtorch-$LIBTORCH_ABI$LIBTORCH_VARIANT-$PYTORCH_BUILD_VERSION.zip \
       /tmp/$LIBTORCH_HOUSE_DIR/libtorch-$LIBTORCH_ABI$LIBTORCH_VARIANT-latest.zip
)


popd

#######################################################################
# ADD DEPENDENCIES INTO THE WHEEL
#
# auditwheel repair doesn't work correctly and is buggy
# so manually do the work of copying dependency libs and patchelfing
# and fixing RECORDS entries correctly
######################################################################

fname_with_sha256() {
    HASH=$(sha256sum $1 | cut -c1-8)
    DIRNAME=$(dirname $1)
    BASENAME=$(basename $1)
    if [[ $BASENAME == "libnvrtc-builtins.so" || $BASENAME == "libcudnn"* ]]; then
        echo $1
    else
        INITNAME=$(echo $BASENAME | cut -f1 -d".")
        ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
        echo "$DIRNAME/$INITNAME-$HASH.$ENDNAME"
    fi
}

fname_without_so_number() {
    LINKNAME=$(echo $1 | sed -e 's/\.so.*/.so/g')
    echo "$LINKNAME"
}

make_wheel_record() {
    FPATH=$1
    if echo $FPATH | grep RECORD >/dev/null 2>&1; then
        # if the RECORD file, then
        echo "$FPATH,,"
    else
        HASH=$(openssl dgst -sha256 -binary $FPATH | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
        FSIZE=$(ls -nl $FPATH | awk '{print $5}')
        echo "$FPATH,sha256=$HASH,$FSIZE"
    fi
}

echo 'Built this package:'
(
    set -x
    mkdir -p /$LIBTORCH_HOUSE_DIR
    mv /tmp/$LIBTORCH_HOUSE_DIR/*.zip /$LIBTORCH_HOUSE_DIR
    rm -rf /tmp/$LIBTORCH_HOUSE_DIR
)
TMP_DIR=$(mktemp -d)
trap "rm -rf ${TMP_DIR}" EXIT
pushd "${TMP_DIR}"

for pkg in /$LIBTORCH_HOUSE_DIR/libtorch*.zip; do

    # if the glob didn't match anything
    if [[ ! -e $pkg ]]; then
        continue
    fi

    rm -rf tmp
    mkdir -p tmp
    cd tmp
    cp $pkg .

    unzip -q $(basename $pkg)
    rm -f $(basename $pkg)

    PREFIX=libtorch

    if [[ $pkg != *"without-deps"* ]]; then
        # copy over needed dependent .so files over and tag them with their hash
        patched=()
        for filepath in "${DEPS_LIST[@]}"; do
            filename=$(basename $filepath)
            destpath=$PREFIX/lib/$filename
            if [[ "$filepath" != "$destpath" ]]; then
                cp $filepath $destpath
            fi

            if [[ "$DESIRED_CUDA" == *"rocm"* ]]; then
                patchedpath=$(fname_without_so_number $destpath)
            else
                patchedpath=$(fname_with_sha256 $destpath)
            fi
            patchedname=$(basename $patchedpath)
            if [[ "$destpath" != "$patchedpath" ]]; then
                mv $destpath $patchedpath
            fi
            patched+=("$patchedname")
            echo "Copied $filepath to $patchedpath"
        done

        echo "patching to fix the so names to the hashed names"
        for ((i=0;i<${#DEPS_LIST[@]};++i)); do
            find $PREFIX -name '*.so*' | while read sofile; do
                origname=${DEPS_SONAME[i]}
                patchedname=${patched[i]}
                if [[ "$origname" != "$patchedname" ]] || [[ "$DESIRED_CUDA" == *"rocm"* ]]; then
                    set +e
                    origname=$($PATCHELF_BIN --print-needed $sofile | grep "$origname.*")
                    ERRCODE=$?
                    set -e
                    if [ "$ERRCODE" -eq "0" ]; then
                        echo "patching $sofile entry $origname to $patchedname"
                        $PATCHELF_BIN --replace-needed $origname $patchedname $sofile
                    fi
                fi
            done
        done

        # copy over needed auxiliary files
        for ((i=0;i<${#DEPS_AUX_SRCLIST[@]};++i)); do
            srcpath=${DEPS_AUX_SRCLIST[i]}
            dstpath=$PREFIX/${DEPS_AUX_DSTLIST[i]}
            mkdir -p $(dirname $dstpath)
            cp $srcpath $dstpath
        done
    fi

    # set RPATH of _C.so and similar to $ORIGIN, $ORIGIN/lib
    find $PREFIX -maxdepth 1 -type f -name "*.so*" | while read sofile; do
        echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/lib'
        $PATCHELF_BIN --set-rpath '$ORIGIN:$ORIGIN/lib' $sofile
        $PATCHELF_BIN --print-rpath $sofile
    done

    # set RPATH of lib/ files to $ORIGIN
    find $PREFIX/lib -maxdepth 1 -type f -name "*.so*" | while read sofile; do
        echo "Setting rpath of $sofile to " '$ORIGIN'
        $PATCHELF_BIN --set-rpath '$ORIGIN' $sofile
        $PATCHELF_BIN --print-rpath $sofile
    done

    # regenerate the RECORD file with new hashes
    record_file=`echo $(basename $pkg) | sed -e 's/-cp.*$/.dist-info\/RECORD/g'`
    if [[ -e $record_file ]]; then
        echo "Generating new record file $record_file"
        rm -f $record_file
        # generate records for folders in wheel
        find * -type f | while read fname; do
            echo $(make_wheel_record $fname) >>$record_file
        done
    fi

    # zip up the wheel back
    zip -rq $(basename $pkg) $PREFIX*

    # replace original wheel
    rm -f $pkg
    mv $(basename $pkg) $pkg
    cd ..
    rm -rf tmp
done

# Copy wheels to host machine for persistence before testing
if [[ -n "$PYTORCH_FINAL_PACKAGE_DIR" ]]; then
    cp /$LIBTORCH_HOUSE_DIR/libtorch*.zip "$PYTORCH_FINAL_PACKAGE_DIR"
    cp /$LIBTORCH_HOUSE_DIR/debug-libtorch*.zip "$PYTORCH_FINAL_PACKAGE_DIR"
fi
