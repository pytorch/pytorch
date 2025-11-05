#!/usr/bin/env bash
# meant to be called only from the neighboring build.sh and build_cpu.sh scripts

set -ex
SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source ${SOURCE_DIR}/set_desired_python.sh


if [[ -n "$BUILD_PYTHONLESS" && -z "$LIBTORCH_VARIANT" ]]; then
    echo "BUILD_PYTHONLESS is set, so need LIBTORCH_VARIANT to also be set"
    echo "LIBTORCH_VARIANT should be one of shared-with-deps shared-without-deps static-with-deps static-without-deps"
    exit 1
fi

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

MANYLINUX_TAG=""
# TODO move this into the Docker images
OS_NAME=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
if [[ "$OS_NAME" == *"AlmaLinux"* ]]; then
    retry yum install -q -y zip openssl
    MANYLINUX_TAG="manylinux_2_28"
elif [[ "$OS_NAME" == *"Red Hat Enterprise Linux"* ]]; then
    retry dnf install -q -y zip openssl
elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
    # TODO: Remove this once nvidia package repos are back online
    # Comment out nvidia repositories to prevent them from getting apt-get updated, see https://github.com/pytorch/pytorch/issues/74968
    # shellcheck disable=SC2046
    sed -i 's/.*nvidia.*/# &/' $(find /etc/apt/ -type f -name "*.list")
    retry apt-get update
    retry apt-get -y install zip openssl
else
    echo "Unknown OS: '$OS_NAME'"
    exit 1
fi

# We use the package name to test the package by passing this to 'pip install'
# This is the env variable that setup.py uses to name the package. Note that
# pip 'normalizes' the name first by changing all - to _
if [[ -z "$TORCH_PACKAGE_NAME" ]]; then
    TORCH_PACKAGE_NAME='torch'
fi

if [[ -z "$TORCH_NO_PYTHON_PACKAGE_NAME" ]]; then
    TORCH_NO_PYTHON_PACKAGE_NAME='torch_no_python'
fi

TORCH_PACKAGE_NAME="$(echo $TORCH_PACKAGE_NAME | tr '-' '_')"
TORCH_NO_PYTHON_PACKAGE_NAME="$(echo $TORCH_NO_PYTHON_PACKAGE_NAME | tr '-' '_')"
echo "Expecting the built wheels to all be called '$TORCH_PACKAGE_NAME' or '$TORCH_NO_PYTHON_PACKAGE_NAME'"

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

if [[ -e /opt/openssl ]]; then
    export OPENSSL_ROOT_DIR=/opt/openssl
    export CMAKE_INCLUDE_PATH="/opt/openssl/include":$CMAKE_INCLUDE_PATH
fi

# Set AArch64 variables
if [[ $GPU_ARCH_TYPE == *"aarch64"* ]]; then
    export USE_MKLDNN=ON
    export USE_MKLDNN_ACL=ON
    export ACL_ROOT_DIR="/acl"
fi

mkdir -p /tmp/$WHEELHOUSE_DIR

export PATCHELF_BIN=/usr/local/bin/patchelf
patchelf_version=$($PATCHELF_BIN --version)
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
retry pip install -qUr requirements-build.txt
python setup.py clean
retry pip install -qr requirements.txt
case ${DESIRED_PYTHON} in
  cp31*)
    retry pip install -q --pre numpy==2.1.0
    ;;
  # Should catch 3.9+
  *)
    retry pip install -q --pre numpy==2.0.2
    ;;
esac

if [[ "$DESIRED_CUDA" == *"rocm"* ]]; then
    echo "Calling build_amd.py at $(date)"
    python tools/amd_build/build_amd.py
fi

# This value comes from binary_linux_build.sh (and should only be set to true
# for master / release branches)
BUILD_DEBUG_INFO=${BUILD_DEBUG_INFO:=0}

if [[ $BUILD_DEBUG_INFO == "1" ]]; then
    echo "Building wheel and debug info"
else
    echo "BUILD_DEBUG_INFO was not set, skipping debug info"
fi

if [[ "$DISABLE_RCCL" = 1 ]]; then
    echo "Disabling NCCL/RCCL in pyTorch"
    USE_RCCL=0
    USE_NCCL=0
    USE_KINETO=0
else
    USE_RCCL=1
    USE_NCCL=1
    USE_KINETO=1
fi

echo "Calling setup.py bdist at $(date)"

time CMAKE_ARGS=${CMAKE_ARGS[@]} \
    EXTRA_CAFFE2_CMAKE_FLAGS=${EXTRA_CAFFE2_CMAKE_FLAGS[@]} \
    BUILD_LIBTORCH_CPU_WITH_DEBUG=$BUILD_DEBUG_INFO \
    USE_NCCL=${USE_NCCL} USE_RCCL=${USE_RCCL} USE_KINETO=${USE_KINETO} \
    python -m build --wheel --no-isolation --outdir /tmp/$WHEELHOUSE_DIR
echo "Finished setup.py bdist at $(date)"

# Build libtorch packages
if [[ -n "$BUILD_PYTHONLESS" ]]; then
    # Now build pythonless libtorch
    # Note - just use whichever python we happen to be on
    python setup.py clean

    if [[ $LIBTORCH_VARIANT = *"static"* ]]; then
        STATIC_CMAKE_FLAG="-DTORCH_STATIC=1"
    fi

    mkdir -p build
    pushd build
    echo "Calling tools/build_libtorch.py at $(date)"
    time CMAKE_ARGS=${CMAKE_ARGS[@]} \
         EXTRA_CAFFE2_CMAKE_FLAGS="${EXTRA_CAFFE2_CMAKE_FLAGS[@]} $STATIC_CMAKE_FLAG" \
         python ../tools/build_libtorch.py
    echo "Finished tools/build_libtorch.py at $(date)"
    popd

    mkdir -p libtorch/{lib,bin,include,share}
    cp -r build/build/lib libtorch/

    # for now, the headers for the libtorch package will just be copied in
    # from one of the wheels (this is from when this script built multiple
    # wheels at once)
    ANY_WHEEL=$(ls /tmp/$WHEELHOUSE_DIR/torch*.whl | head -n1)
    unzip -d any_wheel $ANY_WHEEL
    if [[ -d any_wheel/torch/include ]]; then
        cp -r any_wheel/torch/include libtorch/
    else
        cp -r any_wheel/torch/lib/include libtorch/
    fi
    cp -r any_wheel/torch/share/cmake libtorch/share/
    rm -rf any_wheel

    echo $PYTORCH_BUILD_VERSION > libtorch/build-version
    echo "$(pushd $PYTORCH_ROOT && git rev-parse HEAD)" > libtorch/build-hash

    mkdir -p /tmp/$LIBTORCH_HOUSE_DIR

    zip -rq /tmp/$LIBTORCH_HOUSE_DIR/libtorch-$LIBTORCH_ABI$LIBTORCH_VARIANT-$PYTORCH_BUILD_VERSION.zip libtorch
    cp /tmp/$LIBTORCH_HOUSE_DIR/libtorch-$LIBTORCH_ABI$LIBTORCH_VARIANT-$PYTORCH_BUILD_VERSION.zip \
       /tmp/$LIBTORCH_HOUSE_DIR/libtorch-$LIBTORCH_ABI$LIBTORCH_VARIANT-latest.zip
fi

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
    # Do not rename nvrtc-builtins.so as they are dynamically loaded
    # by libnvrtc.so
    # Similarly don't mangle libcudnn and libcublas library names
    if [[ $BASENAME == "libnvrtc-builtins.s"* || $BASENAME == "libcudnn"* || $BASENAME == "libcublas"*  ]]; then
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
        echo "\"$FPATH\",,"
    else
        HASH=$(openssl dgst -sha256 -binary $FPATH | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
        FSIZE=$(ls -nl $FPATH | awk '{print $5}')
        echo "\"$FPATH\",sha256=$HASH,$FSIZE"
    fi
}

replace_needed_sofiles() {
    find $1 -name '*.so*' | while read sofile; do
        origname=$2
        patchedname=$3
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
}

echo 'Built this wheel:'
ls /tmp/$WHEELHOUSE_DIR
mkdir -p "/$WHEELHOUSE_DIR"
mv /tmp/$WHEELHOUSE_DIR/torch*linux*.whl /$WHEELHOUSE_DIR/

if [[ -n "$BUILD_PYTHONLESS" ]]; then
    mkdir -p /$LIBTORCH_HOUSE_DIR
    mv /tmp/$LIBTORCH_HOUSE_DIR/*.zip /$LIBTORCH_HOUSE_DIR
    rm -rf /tmp/$LIBTORCH_HOUSE_DIR
fi
rm -rf /tmp/$WHEELHOUSE_DIR
rm -rf /tmp_dir
mkdir /tmp_dir
pushd /tmp_dir

for pkg in /$WHEELHOUSE_DIR/torch_no_python*.whl /$WHEELHOUSE_DIR/torch*linux*.whl /$LIBTORCH_HOUSE_DIR/libtorch*.zip; do

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

    if [[ -d torch ]]; then
        PREFIX=torch
    else
        PREFIX=libtorch
    fi

    if [[ $pkg != *"without-deps"* ]]; then
        # copy over needed dependent .so files over and tag them with their hash
        patched=()
        for filepath in "${DEPS_LIST[@]}"; do
            filename=$(basename $filepath)
            destpath=$PREFIX/lib/$filename
            if [[ "$filepath" != "$destpath" ]]; then
                cp $filepath $destpath
            fi

            # ROCm workaround for roctracer dlopens
            if [[ "$DESIRED_CUDA" == *"rocm"* ]]; then
                patchedpath=$(fname_without_so_number $destpath)
            # Keep the so number for XPU dependencies and libgomp.so.1 to avoid twice load
            elif [[ "$DESIRED_CUDA" == *"xpu"* || "$filename" == "libgomp.so.1" ]]; then
                patchedpath=$destpath
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
            replace_needed_sofiles $PREFIX ${DEPS_SONAME[i]} ${patched[i]}
            # do the same for caffe2, if it exists
            if [[ -d caffe2 ]]; then
                replace_needed_sofiles caffe2 ${DEPS_SONAME[i]} ${patched[i]}
            fi
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
        echo "Setting rpath of $sofile to ${C_SO_RPATH:-'$ORIGIN:$ORIGIN/lib'}"
        $PATCHELF_BIN --set-rpath ${C_SO_RPATH:-'$ORIGIN:$ORIGIN/lib'} ${FORCE_RPATH:-} $sofile
        $PATCHELF_BIN --print-rpath $sofile
    done

    # set RPATH of lib/ files to $ORIGIN
    find $PREFIX/lib -maxdepth 1 -type f -name "*.so*" | while read sofile; do
        echo "Setting rpath of $sofile to ${LIB_SO_RPATH:-'$ORIGIN'}"
        $PATCHELF_BIN --set-rpath ${LIB_SO_RPATH:-'$ORIGIN'} ${FORCE_RPATH:-} $sofile
        $PATCHELF_BIN --print-rpath $sofile
    done

    # create Manylinux 2_28 tag this needs to happen before regenerate the RECORD
    if [[ $MANYLINUX_TAG == "manylinux_2_28" && $GPU_ARCH_TYPE != "cpu-s390x" && $GPU_ARCH_TYPE != "xpu" ]]; then
        wheel_file=$(echo $(basename $pkg) | sed -e 's/-cp.*$/.dist-info\/WHEEL/g')
        sed -i -e s#linux_#"${MANYLINUX_TAG}"# $wheel_file;
    fi

    # regenerate the RECORD file with new hashes
    record_file=$(echo $(basename $pkg) | sed -e 's/-cp.*$/.dist-info\/RECORD/g')
    if [[ -e $record_file ]]; then
        echo "Generating new record file $record_file"
        : > "$record_file"
        # generate records for folders in wheel
        find * -type f | while read fname; do
            make_wheel_record "$fname" >>"$record_file"
        done
    fi

    if [[ $BUILD_DEBUG_INFO == "1" ]]; then
        pushd "$PREFIX/lib"

        # Duplicate library into debug lib
        cp libtorch_cpu.so libtorch_cpu.so.dbg

        # Keep debug symbols on debug lib
        strip --only-keep-debug libtorch_cpu.so.dbg

        # Remove debug info from release lib
        strip --strip-debug libtorch_cpu.so

        objcopy libtorch_cpu.so --add-gnu-debuglink=libtorch_cpu.so.dbg

        # Zip up debug info
        mkdir -p /tmp/debug
        mv libtorch_cpu.so.dbg /tmp/debug/libtorch_cpu.so.dbg
        CRC32=$(objcopy --dump-section .gnu_debuglink=>(tail -c4 | od -t x4 -An | xargs echo) libtorch_cpu.so)

        pushd /tmp
        PKG_NAME=$(basename "$pkg" | sed 's/\.whl$//g')
        zip /tmp/debug-whl-libtorch-"$PKG_NAME"-"$CRC32".zip /tmp/debug/libtorch_cpu.so.dbg
        cp /tmp/debug-whl-libtorch-"$PKG_NAME"-"$CRC32".zip "$PYTORCH_FINAL_PACKAGE_DIR"
        popd

        popd
    fi

    # Rename wheel for Manylinux 2_28
    if [[ $MANYLINUX_TAG == "manylinux_2_28" && $GPU_ARCH_TYPE != "cpu-s390x" && $GPU_ARCH_TYPE != "xpu" ]]; then
        pkg_name=$(echo $(basename $pkg) | sed -e s#linux_#"${MANYLINUX_TAG}_"#)
        zip -rq $pkg_name $PREIX*
        rm -f $pkg
        mv $pkg_name $(dirname $pkg)/$pkg_name
    else
        # zip up the wheel back
        zip -rq $(basename $pkg) $PREIX*
        # remove original wheel
        rm -f $pkg
        mv $(basename $pkg) $pkg
    fi

    cd ..
    rm -rf tmp
done

# Copy wheels to host machine for persistence before testing
if [[ -n "$PYTORCH_FINAL_PACKAGE_DIR" ]]; then
    mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR" || true
    if [[ -n "$BUILD_PYTHONLESS" ]]; then
        cp /$LIBTORCH_HOUSE_DIR/libtorch*.zip "$PYTORCH_FINAL_PACKAGE_DIR"
    else
        cp /$WHEELHOUSE_DIR/torch*.whl "$PYTORCH_FINAL_PACKAGE_DIR"
    fi
fi

# remove stuff before testing
rm -rf /opt/rh
if ls /usr/local/cuda* >/dev/null 2>&1; then
    rm -rf /usr/local/cuda*
fi


# Test that all the wheels work
if [[ -z "$BUILD_PYTHONLESS" ]]; then
  export OMP_NUM_THREADS=4 # on NUMA machines this takes too long
  pushd $PYTORCH_ROOT/test

  # Install the wheel for this Python version
  pip uninstall -y "$TORCH_PACKAGE_NAME"

  pip install "$TORCH_PACKAGE_NAME" --no-index -f /$WHEELHOUSE_DIR --no-dependencies -v

  # Print info on the libraries installed in this wheel
  # Rather than adjust find command to skip non-library files with an embedded *.so* in their name,
  # since this is only for reporting purposes, we add the || true to the ldd command.
  installed_libraries=($(find "$pydir/lib/python${py_majmin}/site-packages/torch/" -name '*.so*'))
  echo "The wheel installed all of the libraries: ${installed_libraries[@]}"
  for installed_lib in "${installed_libraries[@]}"; do
      ldd "$installed_lib" || true
  done

  # Run the tests
  echo "$(date) :: Running tests"
  pushd "$PYTORCH_ROOT"


  LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
          "${PYTORCH_ROOT}/.ci/pytorch/run_tests.sh" manywheel "${py_majmin}" "$DESIRED_CUDA"
  popd
  echo "$(date) :: Finished tests"
fi
