# Configuring the builder.

## Install prerequisites.

```
$ sudo dnf install podman podman-docker jq
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

Build `actions-runner` image using:

```
$ cd self-hosted-builder
$ sudo docker build \
      --pull \
      -f actions-runner.Dockerfile \
      -t iiilinuxibmcom/actions-runner.<name> \
      .
```

If there are failures, ensure that selinux doesn't prevent it from working.
In worst case, selinux can be disabled with `setenforce 0`.

Now prepare all necessary files for runner registration:

```
$ sudo mkdir -p /etc/actions-runner/<name>
$ sudo chmod 700 /etc/actions-runner/<name>
$ sudo /bin/cp <github_app_private_key_file> /etc/actions-runner/<name>/key_private.pem
$ sudo echo <github_app_id> | sudo tee /etc/actions-runner/<name>/appid.env
$ sudo echo <github_app_install_id> | sudo tee /etc/actions-runner/<name>/installid.env
$ sudo echo NAME=<worker_name> | sudo tee    /etc/actions-runner/<name>/env
$ sudo echo ORG=<github_org>   | sudo tee -a /etc/actions-runner/<name>/env
$ cd self-hosted-builder
$ sudo /bin/cp helpers/*.sh /usr/local/bin/
$ sudo chmod 755 /usr/local/bin/app_token.sh /usr/local/bin/gh_token_generator.sh
```

## Autostart the runner.

```
$ sudo systemctl enable --now actions-runner@$NAME
```
