# Configuring the builder.

## Install prerequisites.

```
$ sudo dnf install docker
```

## Add services.

```
$ sudo cp self-hosted-builder/*.service /etc/systemd/system/
$ sudo systemctl daemon-reload
```

## Download qemu-user-static image

```
# sudo docker pull docker.io/multiarch/qemu-user-static
```

## Autostart the x86_64 emulation support.

```
$ sudo systemctl enable --now qemu-user-static
```

## Rebuild the image

First build s390x builder image `docker.io/pytorch/manylinuxs390x-builder`,
using following commands:

```
$ cd ~
$ git clone https://github.com/pytorch/builder
$ cd builder
$ GPU_ARCH_TYPE=cpu-s390x "$(pwd)/manywheel/build_docker.sh"
$ docker image save -o ~/manywheel-s390x.tar docker.io/pytorch/manylinuxs390x-builder:cpu-s390x
```

If no github actions runner registration is done yet, build `actions-runner` image
using:

```
$ cd self-hosted-builder
$ sudo docker build \
      --build-arg repo=<owner>/<name> \
      --build-arg token=<***> \
      --build-arg name=<runner-name> \
      --pull \
      -f actions-runner.Dockerfile \
      -t iiilinuxibmcom/actions-runner.<name> \
      .
```

If there are failures, ensure that selinux doesn't prevent it from working.
In worst case, selinux can be disabled with `setenforce 0`.

## Autostart the runner.

```
$ sudo systemctl enable --now actions-runner@$NAME
```

## Copy runner data from container.
```
$ sudo docker run --name copyrunner -it localhost/iiilinuxibmcom/actions-runner.<name> /bin/bash -c 'tar cvpf /tmp/runner.tar --exclude=./manywheel-s390x.tar --exclude=./venv -C /home/actions-runner .'
$ sudo docker cp copyrunner:/tmp/runner.tar ~/
$ sudo docker container rm copyrunner
```

## Update runner using existing registration

In order to build or update the `iiilinuxibmcom/actions-runner` image, e.g. to get the
latest OS security fixes, use the following commands:

```
$ cd self-hosted-builder
$ /bin/cp ~/runner.tar ~/manywheel-s390x.tar ./
$ sudo docker build \
      --pull \
      -f actions-runner-preregistered.Dockerfile \
      -t iiilinuxibmcom/actions-runner.<name> \
      .
$ /bin/rm -v ./runner.tar ./manywheel-s390x.tar
```
