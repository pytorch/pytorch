#!/bin/bash
# Script used in CI and CD pipeline

set -ex

ver() {
    printf "%3d%03d%03d%03d" $(echo "$1" | tr '.' ' ');
}

# Magma build scripts need `python`
ln -sf /usr/bin/python3 /usr/bin/python

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  almalinux)
    yum install -y gcc-gfortran
    ;;
  *)
    echo "No preinstalls to build magma..."
    ;;
esac

MKLROOT=${MKLROOT:-/opt/conda/envs/py_$ANACONDA_PYTHON_VERSION}

# "install" hipMAGMA into /opt/rocm/magma by copying after build
if [[ $(ver $ROCM_VERSION) -ge $(ver 7.0) ]]; then
    git clone https://github.com/ROCm/utk-magma.git -b release/2.9.0_rocm70 magma
    pushd magma
    # version 2.9 + ROCm 7.0 related updates
    git checkout 91c4f720a17e842b364e9de41edeef76995eb9ad
else
    git clone https://bitbucket.org/icl/magma.git
    pushd magma
    # Version 2.7.2 + ROCm related updates
    git checkout a1625ff4d9bc362906bd01f805dbbe12612953f6
fi

cp make.inc-examples/make.inc.hip-gcc-mkl make.inc
echo 'LIBDIR += -L$(MKLROOT)/lib' >> make.inc
if [[ -f "${MKLROOT}/lib/libmkl_core.a" ]]; then
    echo 'LIB = -Wl,--start-group -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -Wl,--end-group -lpthread -lstdc++ -lm -lgomp -lhipblas -lhipsparse' >> make.inc
fi
echo 'LIB += -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib -Wl,--rpath,$(MKLROOT)/lib -Wl,--rpath,/opt/rocm/magma/lib -ldl' >> make.inc
echo 'DEVCCFLAGS += --gpu-max-threads-per-block=256' >> make.inc
export PATH="${PATH}:/opt/rocm/bin"
if [[ -n "$PYTORCH_ROCM_ARCH" ]]; then
  amdgpu_targets=`echo $PYTORCH_ROCM_ARCH | sed 's/;/ /g'`
else
  amdgpu_targets=`rocm_agent_enumerator | grep -v gfx000 | sort -u | xargs`
fi
for arch in $amdgpu_targets; do
  echo "DEVCCFLAGS += --offload-arch=$arch" >> make.inc
done
# hipcc with openmp flag may cause isnan() on __device__ not to be found; depending on context, compiler may attempt to match with host definition
sed -i 's/^FOPENMP/#FOPENMP/g' make.inc
make -f make.gen.hipMAGMA -j $(nproc)
LANG=C.UTF-8 make lib/libmagma.so -j $(nproc) MKLROOT="${MKLROOT}"
make testing/testing_dgemm -j $(nproc) MKLROOT="${MKLROOT}"
popd
mv magma /opt/rocm
