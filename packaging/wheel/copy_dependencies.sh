#!/usr/bin/env bash

set -eou pipefail

fname_with_sha256() {
    HASH=$(sha256sum $1 | cut -c1-8)
    DIRNAME=$(dirname $1)
    BASENAME=$(basename $1)
    if [[ $BASENAME == "libnvrtc-builtins.so" ]]; then
        echo $1
    else
        INITNAME=$(echo $BASENAME | cut -f1 -d".")
        ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
        echo "$DIRNAME/$INITNAME-$HASH.$ENDNAME"
    fi
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

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
OUT_DIR=${OUT_DIR:-${GIT_ROOT_DIR}/out}

LIBGOMP_PATH=$(find / -name libgomp.so.1 | head -1)
if ! find / -name libgomp.so.1 | head -1 >/dev/null 2>/dev/null; then
    echo "ERROR: libgomp.so.1 not found, exiting"
    exit 1
fi

DEPS_LIST=(
    ${LIBGOMP_PATH}
)
DEPS_SONAME=(
    "libgomp.so.1"
)

if nvcc --version >/dev/null 2>/dev/null; then
    source "${SOURCE_DIR}/cuda_helpers.sh"
fi


for pkg in "${OUT_DIR}"/*.whl; do

    # if the glob didn't match anything
    if [[ ! -e "${pkg}" ]]; then
        continue
    fi

    tmp_dir=$(mktemp -d)
    PREFIX=${tmp_dir}/torch
    trap 'rm -rf ${tmp_dir}' EXIT
    cp ${pkg} ${tmp_dir}

    unzip -q ${pkg} -d ${tmp_dir}

    # copy over needed dependent .so files over and tag them with their hash
    patched=()
    for filepath in "${DEPS_LIST[@]}"; do
        filename=$(basename ${filepath})
        destpath=${PREFIX}/lib/${filename}
        if [[ "${filepath}" != "${destpath}" ]]; then
            cp -v $filepath $destpath
        fi

        patchedpath=$(fname_with_sha256 ${destpath})
        patchedname=$(basename ${patchedpath})
        if [[ "${destpath}" != "${patchedpath}" ]]; then
            mv -v ${destpath} ${patchedpath}
        fi
        patched+=("$patchedname")
    done

    for ((i=0;i<${#DEPS_LIST[@]};++i)); do
        find $PREFIX -name '*.so*' | while read sofile; do
            origname=${DEPS_SONAME[i]}
            patchedname=${patched[i]}
            if [[ "$origname" != "$patchedname" ]]; then
                if patchelf --print-needed $sofile | grep $origname 2>&1 >/dev/null; then
                    (
                        set -x
                        patchelf --replace-needed $origname $patchedname $sofile
                    )
                fi
            fi
        done
    done

    # set RPATH of _C.so and similar to $ORIGIN, $ORIGIN/lib
    find $PREFIX -maxdepth 1 -type f -name "*.so*" | while read -r sofile; do
        echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/lib'
        (
            set -x
            patchelf --set-rpath '$ORIGIN:$ORIGIN/lib' $sofile
            patchelf --print-rpath $sofile
        )
    done

    # set RPATH of lib/ files to $ORIGIN
    find $PREFIX/lib -maxdepth 1 -type f -name "*.so*" | while read -r sofile; do
        echo "Setting rpath of $sofile to " '$ORIGIN'
        (
            set -x
            patchelf --set-rpath '$ORIGIN' $sofile
            patchelf --print-rpath $sofile
        )
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
    pushd "${tmp_dir}"
    (
        set -x
        zip -rq $(basename ${pkg}) "$(basename ${PREFIX})*"
        mv "$(basename ${pkg})" "${pkg}"

    )
done
