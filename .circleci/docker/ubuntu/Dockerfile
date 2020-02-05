ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies (so that this step can be cached separately)
ARG EC2
ADD ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install clang
ARG LLVMDEV
ARG CLANG_VERSION
ADD ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# (optional) Install thrift.
ARG THRIFT
ADD ./common/install_thrift.sh install_thrift.sh
RUN if [ -n "${THRIFT}" ]; then bash ./install_thrift.sh; fi
RUN rm install_thrift.sh
ENV INSTALLED_THRIFT ${THRIFT}

# Install user
ADD ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install katex
ARG KATEX
ADD ./common/install_katex.sh install_katex.sh
RUN bash ./install_katex.sh && rm install_katex.sh

# Install conda
ENV PATH /opt/conda/bin:$PATH
ARG ANACONDA_PYTHON_VERSION
ADD ./common/install_conda.sh install_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh

# Install gcc
ARG GCC_VERSION
ADD ./common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# Install non-standard Python versions (via Travis binaries)
ARG TRAVIS_PYTHON_VERSION
ARG TRAVIS_DL_URL_PREFIX
ENV PATH /opt/python/$TRAVIS_PYTHON_VERSION/bin:$PATH
ADD ./common/install_travis_python.sh install_travis_python.sh
RUN bash ./install_travis_python.sh && rm install_travis_python.sh

# (optional) Install protobuf for ONNX
ARG PROTOBUF
ADD ./common/install_protobuf.sh install_protobuf.sh
RUN if [ -n "${PROTOBUF}" ]; then bash ./install_protobuf.sh; fi
RUN rm install_protobuf.sh
ENV INSTALLED_PROTOBUF ${PROTOBUF}

# (optional) Install database packages like LMDB and LevelDB
ARG DB
ADD ./common/install_db.sh install_db.sh
RUN if [ -n "${DB}" ]; then bash ./install_db.sh; fi
RUN rm install_db.sh
ENV INSTALLED_DB ${DB}

# (optional) Install vision packages like OpenCV and ffmpeg
ARG VISION
ADD ./common/install_vision.sh install_vision.sh
RUN if [ -n "${VISION}" ]; then bash ./install_vision.sh; fi
RUN rm install_vision.sh
ENV INSTALLED_VISION ${VISION}

# (optional) Install Android NDK
ARG ANDROID
ARG ANDROID_NDK
ARG GRADLE_VERSION
ADD ./common/install_android.sh install_android.sh
ADD ./android/AndroidManifest.xml AndroidManifest.xml
ADD ./android/build.gradle build.gradle
RUN if [ -n "${ANDROID}" ]; then bash ./install_android.sh; fi
RUN rm install_android.sh
RUN rm AndroidManifest.xml
RUN rm build.gradle
ENV INSTALLED_ANDROID ${ANDROID}

# (optional) Install non-default CMake version
ARG CMAKE_VERSION
ADD ./common/install_cmake.sh install_cmake.sh
RUN if [ -n "${CMAKE_VERSION}" ]; then bash ./install_cmake.sh; fi
RUN rm install_cmake.sh

# (optional) Install non-default Ninja version
ARG NINJA_VERSION
ADD ./common/install_ninja.sh install_ninja.sh
RUN if [ -n "${NINJA_VERSION}" ]; then bash ./install_ninja.sh; fi
RUN rm install_ninja.sh

# Install ccache/sccache (do this last, so we get priority in PATH)
ADD ./common/install_cache.sh install_cache.sh
ENV PATH /opt/cache/bin:$PATH
RUN bash ./install_cache.sh && rm install_cache.sh

# Add jni.h for java host build
ADD ./common/install_jni.sh install_jni.sh
ADD ./java/jni.h jni.h
RUN bash ./install_jni.sh && rm install_jni.sh

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}

USER jenkins
CMD ["bash"]
