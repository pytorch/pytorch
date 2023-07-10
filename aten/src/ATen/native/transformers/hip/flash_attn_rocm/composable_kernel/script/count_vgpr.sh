#!/bin/bash
FILE=$1

for num in {0..255}
do
    base_pattern="(\[?${num}\b|\[\d*:${num}\])"
    spattern="s${base_pattern}"
    vpattern="v${base_pattern}"
    apattern="a${base_pattern}"
    scount=$(grep -P $spattern $FILE | wc -l)
    vcount=$(grep -P $vpattern $FILE | wc -l)
    acount=$(grep -P $apattern $FILE | wc -l)
    bash -c "echo -n v${num}   $vcount && \
             echo -n , s${num} $scount && \
             echo -n , a${num} $acount"
    if [[ $scount -ne 0 || $vcount -ne 0 || $acount -ne 0 ]]; then
        echo -n "  *"
    fi
    echo ""
done
