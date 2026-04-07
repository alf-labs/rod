#!/usr/bin/bash

A="$1" ; shift

if [[ "$A" == "0p0" ]]; then
    mkdir -p output
    ( set -x
    ./_run.sh -i 0 --start 0:20 --end 0:45 -0 --display=full --overlay-video --locator-rod-sz=30,15,50,/1280 $@
    )
elif [[ "$A" == "0p1" ]]; then
    LATEST_JSON=$( ls -1 --sort=time output/*.json | head -n 1 )
    ( set -x
    ./_run.sh -l "$LATEST_JSON" -1 --display=full --overlay-video --locator-rod-sz=30,15,50,/1280 $@
    )
else
    echo "@@ Missing test argument."
fi
