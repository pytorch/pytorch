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
# sudo docker pull docker.io/iiilinuxibmcom/qemu-user-static:6.1.0-1
```

## Autostart the x86_64 emulation support.

```
$ sudo systemctl enable --now qemu-user-static
```

## Rebuild the image

In order to build or update the `iiilinuxibmcom/actions-runner` image, e.g. to get the
latest OS security fixes, use the following commands:

```
$ cd self-hosted-builder
$ sudo docker build \
      --build-arg repo=<owner>/<name> \
      --build-arg token=<***> \
      --pull \
      -f actions-runner.Dockerfile \
      -t iiilinuxibmcom/actions-runner \
      .
```

If it fails, ensure that selinux doesn't prevent it from working.
In worst case, selinux can be disabled with `setenforce 0`.

## Autostart the runner.

```
$ sudo systemctl enable --now actions-runner@$NAME
```
