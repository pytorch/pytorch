ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive

# Set AMD gpu targets to build for
ARG PYTORCH_ROCM_ARCH
ENV PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}

# Install common dependencies (so that this step can be cached separately)
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install katex
ARG KATEX
COPY ./common/install_docs_reqs.sh install_docs_reqs.sh
RUN bash ./install_docs_reqs.sh && rm install_docs_reqs.sh

# Install conda and other packages (e.g., numpy, pytest)
ARG ANACONDA_PYTHON_VERSION
ARG BUILD_ENVIRONMENT
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
ENV PATH /opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:/opt/conda/bin:$PATH
COPY requirements-ci.txt /opt/conda/requirements-ci.txt
COPY ./common/install_conda.sh install_conda.sh
COPY ./common/common_utils.sh common_utils.sh
RUN bash ./install_conda.sh && rm install_conda.sh common_utils.sh /opt/conda/requirements-ci.txt

# Install gcc
ARG GCC_VERSION
COPY ./common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# Install clang
ARG CLANG_VERSION
COPY ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# (optional) Install vision packages like OpenCV
ARG VISION
COPY ./common/install_vision.sh ./common/cache_vision_models.sh ./common/common_utils.sh ./
RUN if [ -n "${VISION}" ]; then bash ./install_vision.sh; fi
RUN rm install_vision.sh cache_vision_models.sh common_utils.sh
ENV INSTALLED_VISION ${VISION}

# Install rocm
ARG ROCM_VERSION
ENV ROCM_VERSION=${ROCM_VERSION}
RUN mkdir ci_commit_pins
COPY ./common/common_utils.sh common_utils.sh
COPY ./ci_commit_pins/rocm-composable-kernel.txt ci_commit_pins/rocm-composable-kernel.txt
COPY ./common/install_rocm.sh install_rocm.sh
RUN bash ./install_rocm.sh
RUN rm install_rocm.sh common_utils.sh
RUN rm -r ci_commit_pins
COPY ./common/install_rocm_magma.sh install_rocm_magma.sh
RUN if [ "${ROCM_VERSION}" != "nightly" ]; then bash ./install_rocm_magma.sh ${ROCM_VERSION}; fi
RUN rm install_rocm_magma.sh
ADD ./common/install_miopen.sh install_miopen.sh
RUN if [ "${ROCM_VERSION}" != "nightly" ]; then bash ./install_miopen.sh ${ROCM_VERSION}; fi && rm install_miopen.sh
ADD ./common/install_rocm_drm.sh install_rocm_drm.sh
RUN if [ "${ROCM_VERSION}" != "nightly" ]; then bash ./install_rocm_drm.sh /usr ; fi && rm install_rocm_drm.sh

# ROCm environment variables are set in /etc/rocm_env.sh by install_rocm.sh
# and sourced via /etc/bash.bashrc for interactive shells.
# CI scripts should source /etc/rocm_env.sh directly.
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install amdsmi
COPY ./common/install_amdsmi.sh install_amdsmi.sh
RUN bash ./install_amdsmi.sh
RUN rm install_amdsmi.sh

COPY ./common/install_openssl.sh install_openssl.sh
ENV OPENSSL_ROOT_DIR /opt/openssl
RUN bash ./install_openssl.sh
ENV OPENSSL_DIR /opt/openssl

ARG INDUCTOR_BENCHMARKS
ARG ANACONDA_PYTHON_VERSION
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
COPY ./common/install_inductor_benchmark_deps.sh install_inductor_benchmark_deps.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/huggingface-requirements.txt huggingface-requirements.txt
COPY ci_commit_pins/timm.txt timm.txt
COPY ci_commit_pins/torchbench.txt torchbench.txt
RUN if [ -n "${INDUCTOR_BENCHMARKS}" ]; then bash ./install_inductor_benchmark_deps.sh; fi
RUN rm install_inductor_benchmark_deps.sh common_utils.sh timm.txt huggingface-requirements.txt torchbench.txt

# (optional) Install non-default Ninja version
ARG NINJA_VERSION
COPY ./common/install_ninja.sh install_ninja.sh
RUN if [ -n "${NINJA_VERSION}" ]; then bash ./install_ninja.sh; fi
RUN rm install_ninja.sh

ARG TRITON
# Install triton, this needs to be done before sccache because the latter will
# try to reach out to S3, which docker build runners don't have access
COPY ./common/install_triton.sh install_triton.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/triton.txt triton.txt
COPY triton_version.txt triton_version.txt
RUN if [ -n "${TRITON}" ]; then bash ./install_triton.sh; fi
RUN rm install_triton.sh common_utils.sh triton.txt triton_version.txt


# Install ccache/sccache (do this last, so we get priority in PATH)
COPY ./common/install_cache.sh install_cache.sh
ENV PATH /opt/cache/bin:$PATH
RUN bash ./install_cache.sh && rm install_cache.sh

# Install Open MPI for ROCm
COPY ./common/install_openmpi.sh install_openmpi.sh
RUN if [ -n "${CUDA_VERSION}" ]; then bash install_openmpi.sh; fi
RUN rm install_openmpi.sh

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}

# Install LLVM dev version (Defined in the pytorch/builder github repository)
COPY --from=pytorch/llvm:9.0.1 /opt/llvm /opt/llvm

USER jenkins
CMD ["bash"]
