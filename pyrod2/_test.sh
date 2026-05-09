#!/usr/bin/bash

A="$1" ; shift

LATEST_JSON=$( ls -1 --sort=time output/*.json | head -n 1 )
ARGS="--display=full --overlay-video --locator-rod-sz=30,15,50,/1280"
ARGS="--display=full --overlay-video"

mkdir -p output

if [[ "$A" == "0p0" ]]; then        # full run with video 0
    ( set -x
    # time ./_run.sh -i 0 --start 0:20 --end 0:45 -0 $ARGS $@
    time ./_run.sh -i 0 -0 $ARGS $@
    )
elif [[ "$A" == "0t0" ]]; then      # first tunnel in video 0
    # full run with video 0
    ( set -x
    time ./_run.sh -i 0 --start 1:30 --end 2:00 -0 $ARGS $@
    )
elif [[ "$A" == "1p0" ]]; then
    ( set -x
    # time ./_run.sh -i 0 --start 0:20 --end 0:45 -0 $ARGS $@
    time ./_run.sh -i 1 $ARGS $@
    )
elif [[ "$A" == "0p1" ]]; then
    ( set -x
    time ./_run.sh $ARGS $@
    )
elif [[ "$A" == "0p2" ]]; then
    ( set -x
    time ./_run.sh -l "$LATEST_JSON" $ARGS $@
    )
else
    echo "@@ Missing test argument."
fi
