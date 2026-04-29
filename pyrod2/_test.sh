#!/usr/bin/bash

A="$1" ; shift

LATEST_JSON=$( ls -1 --sort=time output/*.json | head -n 1 )
ARGS="--display=full --overlay-video --locator-rod-sz=30,15,50,/1280"
ARGS="--display=full --overlay-video"

if [[ "$A" == "0p0" ]]; then
    mkdir -p output
    ( set -x
    time ./_run.sh -i 0 --start 0:20 --end 0:45 $ARGS $@ # -0
    )
elif [[ "$A" == "0p1" ]]; then
    ( set -x
    time ./_run.sh -l "$LATEST_JSON" -1 $ARGS $@
    )
elif [[ "$A" == "0p2" ]]; then
    ( set -x
    time ./_run.sh -l "$LATEST_JSON" $ARGS $@
    )
else
    echo "@@ Missing test argument."
fi
