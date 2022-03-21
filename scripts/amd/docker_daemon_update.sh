# install json tool
sudo apt install -y jq

# check if docker daemon exists
DAEMON_JSON=/etc/docker/daemon.json
if test -f "$DAEMON_JSON"; then
    echo "$DAEMON_JSON exists."
    # rm -rf $DAEMON_JSON && echo '{}' >$DAEMON_JSON
else
    echo '{}' >$DAEMON_JSON
fi

# modify docker daemon json
jq '. + {
    "insecure-registries": [
        "compute-artifactory.amd.com:5000",
        "10.216.151.220:5000",
        "compute-artifactory.amd.com:5001"
    ]
}' \
    $DAEMON_JSON >/tmp/tmp.$$.json &&
    mv /tmp/tmp.$$.json $DAEMON_JSON

# print out dameon json
cat $DAEMON_JSON

# reload docker
sudo systemctl daemon-reload
sudo systemctl restart docker
