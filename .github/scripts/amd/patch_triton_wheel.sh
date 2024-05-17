#!/bin/bash
set -x

if [ -z "$1" ]; then
    echo "Need wheel location argument" && exit 1
fi

WHEELHOUSE_DIR=$1
PATCHELF_BIN=patchelf
ROCM_LIB=backends/amd/lib
ROCM_LD=backends/amd/llvm/bin
PREFIX=triton
fname_without_so_number() {
    LINKNAME=$(echo $1 | sed -e 's/\.so.*/.so/g')
    echo "$LINKNAME"
}

replace_needed_sofiles() {
    find $1 -name '*.so*' -o -name 'ld.lld' | while read sofile; do
        origname=$2
        patchedname=$3
        if [[ "$origname" != "$patchedname" ]]; then
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

mkdir  -p "/tmp_dir"
pushd /tmp_dir
for pkg in /$WHEELHOUSE_DIR/*triton*.whl; do
    echo "Modifying $pkg"
    rm -rf tmp
    mkdir -p tmp
    cd tmp
    cp $pkg .
    unzip -q $(basename $pkg)
    rm -f $(basename $pkg)
    $PATCHELF_BIN --set-rpath ${LD_SO_RPATH:-'$ORIGIN:$ORIGIN/../../lib'} $PREFIX/$ROCM_LD/ld.lld
    $PATCHELF_BIN --print-rpath $PREFIX/$ROCM_LD/ld.lld
    # Modify libtriton.so as it sits in _C directory apart from its dependencies
    find $PREFIX/_C -type f -name "*.so*" | while read sofile; do
        echo "Setting rpath of $sofile"
        $PATCHELF_BIN --set-rpath ${C_SO_RPATH:-'$ORIGIN:$ORIGIN/'../$ROCM_LIB} ${FORCE_RPATH:-} $sofile
        $PATCHELF_BIN --print-rpath $sofile
    done

    # All included dependencies are included in a single lib directory
    deps=()
    deps_soname=()
    while read sofile; do
        echo "Setting rpath of $sofile to ${LIB_SO_RPATH:-'$ORIGIN'}"
        $PATCHELF_BIN --set-rpath ${LIB_SO_RPATH:-'$ORIGIN'} ${FORCE_RPATH:-} $sofile
        $PATCHELF_BIN --print-rpath $sofile
        deps+=("$sofile")
        deps_soname+=("$(basename $sofile)")
    done < <(find $PREFIX/$ROCM_LIB -type f -name "*.so*")

    patched=()
    for filepath in "${deps[@]}"; do
        filename=$(basename $filepath)
        destpath=$PREFIX/$ROCM_LIB/$filename
        if [[ "$filepath" != "$destpath" ]]; then
            cp $filepath $destpath
        fi
        patchedpath=$(fname_without_so_number $destpath)
        patchedname=$(basename $patchedpath)
        if [[ "$destpath" != "$patchedpath" ]]; then
            mv $destpath $patchedpath
        fi
        patched+=("$patchedname")
        echo "Copied $filepath to $patchedpath"
    done

    # Go through all required shared objects and see if any of our other objects are dependants.  If so, replace so.ver wth so
    for ((i=0;i<${#deps[@]};++i)); do
        echo "replacing "${deps_soname[i]} ${patched[i]}
        replace_needed_sofiles $PREFIX/$ROCM_LIB ${deps_soname[i]} ${patched[i]}
        replace_needed_sofiles $PREFIX/_C ${deps_soname[i]} ${patched[i]}
        replace_needed_sofiles $PREFIX/$ROCM_LD ${deps_soname[i]} ${patched[i]}
    done

    # Re-bundle whl with so adjustments
    zip -rqy $(basename $pkg) *

    if [[ -z "${MANYLINUX_VERSION}" ]]; then
        newpkg=$pkg
    else
        newpkg=$(echo $pkg | sed -e "s/\linux_x86_64/${MANYLINUX_VERSION}/g")
    fi

    # Remove original whl
    rm -f $pkg

    # Move rebuilt whl to original location with new name.
    mv $(basename $pkg) $newpkg
done
