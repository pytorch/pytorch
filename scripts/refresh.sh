#!/usr/bin/env bash

# If -c is passed, we will clean up the repo before building
if [ "$1" = "-c" ]; then
    echo "Cleaning up repo before building"
    python setup.py clean
else
    echo "Not cleaning, if you get errors, pass -c to clean before building"
fi

# Build
echo "Updating submodules"
git submodule update --init --recursive
echo "Building"
python setup.py develop
