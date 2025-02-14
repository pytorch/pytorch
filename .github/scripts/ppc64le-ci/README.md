# Configuring the builder.

## Install prerequisites.

```
Install Docker 
```
## Clone pytorch repository

## Add services.

```
$ sudo cp self-hosted-builder/*.service /etc/systemd/system/
$ sudo systemctl daemon-reload
```
Next step is to build `actions-runner` image using:

```
## clone gaplib repo (https://github.com/anup-kodlekere/gaplib.git) and copy runner-sdk-8.ppc64le patch from gaplib/build-files into pytorch/.github\scripts\ppc64le-ci\self-hosted-builder

$ cd self-hosted-builder
$ sudo docker build \
      --pull \
      -f actions-runner.Dockerfile \
      --build-arg RUNNERPATCH="runner-sdk-8.ppc64le.patch" \
      -t iiilinuxibmcom/actions-runner.<name> \
      .
```

Now prepare all necessary files for runner registration:

```
$ sudo mkdir -p /etc/actions-runner/<name>
$ sudo chmod 755 /etc/actions-runner/<name>
$ sudo /bin/cp <github_app_private_key_file> /etc/actions-runner/<name>/key_private.pem
$ sudo echo <github_app_id> | sudo tee /etc/actions-runner/<name>/appid.env
$ sudo echo <github_app_install_id> | sudo tee /etc/actions-runner/<name>/installid.env
$ sudo echo NAME=<worker_name> | sudo tee    /etc/actions-runner/<name>/env
$ sudo echo OWNER=<github_owner>   | sudo tee -a /etc/actions-runner/<name>/env
$ sudo echo REPO=pytorch   | sudo tee -a /etc/actions-runner/<name>/env
$ cd self-hosted-builder
$ sudo /bin/cp helpers/*.sh /usr/local/bin/
$ sudo chmod 755 /usr/local/bin/app_token.sh /usr/local/bin/gh_token_generator.sh
```

## Autostart the runner.

```
$ sudo systemctl enable --now actions-runner@$NAME
```
