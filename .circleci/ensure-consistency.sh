#!/bin/bash -xe

CHECKED_IN_FILE=config.yml

TEMPFILE=$(mktemp)

./generate-config-yml.py > "$TEMPFILE"
diff --brief "$TEMPFILE" "$CHECKED_IN_FILE"

rm "$TEMPFILE"
