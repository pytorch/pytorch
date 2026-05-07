ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION} as base

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive
ENV PIP_EXTRA_INDEX_URL=https://gitlab.com/api/v4/projects/riseproject%2Fpython%2Fwheel_builder/packages/pypi/simple
ENV PIP_PREFER_BINARY=true

ARG GCC_VERSION
ARG CLANG_VERSION

# Allow APT to run as root
RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox

# Install common dependencies (so that this step can be cached separately)
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install clang
ARG LLVMDEV
COPY ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install katex
ARG KATEX
COPY ./common/install_docs_reqs.sh install_docs_reqs.sh
RUN bash ./install_docs_reqs.sh && rm install_docs_reqs.sh

# Install conda and other packages (e.g., numpy, pytest)
ARG PYTHON_VERSION
ARG PIP_CMAKE
# Put venv into the env vars so users don't need to activate it
ENV PATH /var/lib/jenkins/ci_env/bin:$PATH
ENV VIRTUAL_ENV /var/lib/jenkins/ci_env
COPY requirements-ci.txt /opt/requirements-ci.txt
COPY ./common/install_python.sh install_python.sh
RUN bash ./install_python.sh && rm install_python.sh /opt/requirements-ci.txt

# Install gcc
ARG GCC_VERSION
COPY ./common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# Install lcov for C++ code coverage
COPY ./common/install_lcov.sh install_lcov.sh
RUN  bash ./install_lcov.sh && rm install_lcov.sh

ARG INDUCTOR_BENCHMARKS
COPY ./common/install_inductor_benchmark_deps.sh install_inductor_benchmark_deps.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/huggingface-requirements.txt huggingface-requirements.txt
COPY ci_commit_pins/timm.txt timm.txt
COPY ci_commit_pins/torchbench.txt torchbench.txt
# Only build aoti cpp tests when INDUCTOR_BENCHMARKS is set to True
ENV BUILD_AOT_INDUCTOR_TEST ${INDUCTOR_BENCHMARKS}
RUN if [ -n "${INDUCTOR_BENCHMARKS}" ]; then bash ./install_inductor_benchmark_deps.sh; fi
RUN rm install_inductor_benchmark_deps.sh common_utils.sh timm.txt huggingface-requirements.txt torchbench.txt

ARG INSTALL_MINGW
COPY ./common/install_mingw.sh install_mingw.sh
RUN if [ -n "${INSTALL_MINGW}" ]; then bash ./install_mingw.sh; fi
RUN rm install_mingw.sh

FROM base as final

ARG EXECUTORCH
# Build and install executorch
COPY ./common/install_executorch.sh install_executorch.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/executorch.txt executorch.txt
RUN if [ -n "${EXECUTORCH}" ]; then bash ./install_executorch.sh; fi
RUN rm install_executorch.sh common_utils.sh executorch.txt

ARG HALIDE
# Build and install halide
COPY ./common/install_halide.sh install_halide.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/halide.txt halide.txt
RUN if [ -n "${HALIDE}" ]; then bash ./install_halide.sh; fi
RUN rm install_halide.sh common_utils.sh halide.txt

ARG PALLAS
ARG TPU
ARG CUDA_VERSION
# Install JAX (for Pallas) - with TPU/CUDA support based on args
# TPU=yes -> install JAX with TPU support
# PALLAS=yes + CUDA_VERSION -> install JAX with CUDA support
# PALLAS=yes (no CUDA) -> install JAX CPU-only
COPY ./common/requirements_tpu.txt requirements_tpu.txt
COPY ./common/install_jax.sh install_jax.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ./ci_commit_pins/jax.txt /ci_commit_pins/jax.txt
RUN if [ -n "${TPU}" ]; then \
      bash -c "source ./common_utils.sh && pip_install -r requirements_tpu.txt" && \
      bash ./install_jax.sh tpu; \
    elif [ -n "${PALLAS}" ]; then \
      bash ./install_jax.sh ${CUDA_VERSION:-cpu}; \
    fi
RUN rm -f install_jax.sh common_utils.sh /ci_commit_pins/jax.txt

ARG ONNX
# Install ONNX dependencies
COPY ./common/install_onnx.sh ./common/common_utils.sh ./
RUN if [ -n "${ONNX}" ]; then bash ./install_onnx.sh; fi
RUN rm install_onnx.sh common_utils.sh

ARG OPENBLAS
ARG OPENBLAS_VERSION
COPY ./common/install_openblas.sh install_openblas.sh
RUN if [ -n "${OPENBLAS}" ]; then bash ./install_openblas.sh; fi
RUN rm install_openblas.sh
ENV INSTALLED_OPENBLAS ${OPENBLAS}

# Install ccache/sccache (do this last, so we get priority in PATH)
ARG SKIP_SCCACHE_INSTALL
COPY ./common/install_cache.sh install_cache.sh
ENV PATH /opt/cache/bin:$PATH
RUN if [ -z "${SKIP_SCCACHE_INSTALL}" ]; then bash ./install_cache.sh; fi
RUN rm install_cache.sh

# Add jni.h for java host build
COPY ./common/install_jni.sh install_jni.sh
COPY ./java/jni.h jni.h
RUN bash ./install_jni.sh && rm install_jni.sh

RUN chown -R jenkins:jenkins /var/lib/jenkins/ci_env

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}

USER jenkins
CMD ["bash"]
