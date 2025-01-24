ARG IMAGE_NAME=pytorch/manylinux2_28-builder
ARG IMAGE_BASE_TAG="cuda12.6-main"

FROM ${IMAGE_NAME}:${IMAGE_BASE_TAG}

#
# Build trick #1
# Almalinux is not officially supported by aws-efa-installer
# but in a nutshell it is just an EL8 based distro, so we mimic to RockyLinux
#
# Build trick #2 (workaround for FindMPI.cmake bug)
# By unknown reason FindMPI.cmake[1] try to detect MPI library presence
# by lookup  mpi-${LANG} PKG, but openmpi standard is known to have
# ompi-${LANG} pkg naming convention. So we have to manually create a symlinks
# and privided explicit PKG_CONFIG_PATH 
#
#Footnotes
#[1]https://github.com/Kitware/CMake/blob/master/Modules/FindMPI.cmake#L1592-L1597
# Install EFA libs
ARG AWS_EFA_INSTALLER=1.34.0
RUN mkdir -p /tmp/efa \
    && cd /tmp/efa \
    && curl -sSL https://efa-installer.amazonaws.com/aws-efa-installer-${AWS_EFA_INSTALLER}.tar.gz | tar -xz \
    && cd aws-efa-installer \
    && cp /etc/os-release /etc/os-release.orig \
    && sed -i 's/AlmaLinux/Rocky Linux/g' /etc/os-release \
    && ./efa_installer.sh -y --skip-kmod --no-verify --enable-gdr \
    && mv /etc/os-release.orig /etc/os-release \
    && echo "/opt/amazon/openmpi/lib64" >> /etc/ld.so.conf.d/000_ompi.conf \
    \
    && pushd /opt/amazon/openmpi/lib64/pkgconfig \
    && ln -s ompi.pc mpi.pc \
    && ln -s ompi-c.pc mpi-c.pc \
    && ln -s ompi-cxx.pc mpi-cxx.pc \
    && popd \
    && ldconfig \
    && cd / \
    && rm -rf /tmp/efa

ENV PATH=$PATH:/opt/amazon/openmpi/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/amazon/openmpi/lib64
ENV PKG_CONFIG_PATH=/opt/amazon/openmpi/lib64/pkgconfig
