#!/bin/bash

ver() {
  printf "%3d%03d%03d%03d" $(echo "$1" | tr '.' ' ');
}

set -ex

# "install" hipMAGMA into /opt/rocm/magma by copying after build
if [[ $(ver $ROCM_VERSION) -ge $(ver 6.0) ]]; then
  git clone https://bitbucket.org/mpruthvi1/magma.git -b rocm_60
  pushd magma
else
  git clone https://bitbucket.org/icl/magma.git
  pushd magma
  git checkout 28592a7170e4b3707ed92644bf4a689ed600c27f
fi

cp make.inc-examples/make.inc.hip-gcc-mkl make.inc
echo 'LIBDIR += -L$(MKLROOT)/lib' >> make.inc
echo 'LIB += -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib -Wl,--rpath,$(MKLROOT)/lib -Wl,--rpath,/opt/rocm/magma/lib' >> make.inc
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
LANG=C.UTF-8 make lib/libmagma.so -j $(nproc) MKLROOT=/opt/conda/envs/py_$ANACONDA_PYTHON_VERSION
make testing/testing_dgemm -j $(nproc) MKLROOT=/opt/conda/envs/py_$ANACONDA_PYTHON_VERSION
popd
mv magma /opt/rocm
