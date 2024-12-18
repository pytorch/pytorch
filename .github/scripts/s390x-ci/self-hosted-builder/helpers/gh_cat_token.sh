#!/usr/bin/env bash

TOKEN_FILE=$1
TOKEN_PIPE=$2

mkfifo "${TOKEN_PIPE}"
cat "${TOKEN_FILE}" > "${TOKEN_PIPE}" &
