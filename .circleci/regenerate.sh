#!/bin/bash -e

# Allows this script to be invoked from any directory:
cd "$(dirname "$0")"

UNCOMMIT_CHANGE=$(git status -s | grep " config.yml" | wc -l | xargs)
if [[ $UNCOMMIT_CHANGE != 0 ]]; then
    OLD_FILE=$(mktemp)
    cp config.yml "$OLD_FILE"
    echo "Uncommitted change detected in .circleci/config.yml"
    echo "It has been backed up to $OLD_FILE"
fi

NEW_FILE=$(mktemp)
./generate_config_yml.py > "$NEW_FILE"
cp "$NEW_FILE" config.yml
echo "New config generated in .circleci/config.yml"
