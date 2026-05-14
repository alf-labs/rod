#!/usr/bin/bash

A="$1" ; shift

ARGS="--display=full --overlay-video --rod-widths 15,40,/1280"

mkdir -p output

if [[ "$A" == "0a" ]]; then        # full run with video 0
    ( set -x
    time ./_run.sh -i 0 -0 -o output/a0_TIME.mp4 $ARGS $@
    )
elif [[ "$A" == "0at" ]]; then      # first tunnel in video 0
    # full run with video 0
    ( set -x
    time ./_run.sh -i 0 --start 1:30 --end 2:00 -0 $ARGS $@
    )
elif [[ "$A" == "0b" ]]; then
    LATEST_JSON=$( ls -1 --sort=time output/a0_*.json | head -n 1 )
    ( set -x
    time ./_run.sh -1 --load-json "$LATEST_JSON" -o output/b0_TIME.mp4 $ARGS $@
    )
elif [[ "$A" == "0c" ]]; then
    LATEST_JSON=$( ls -1 --sort=time output/b0_*.json | head -n 1 )
    ( set -x
    time ./_run.sh --load-json "$LATEST_JSON" --no-json -o output/c0_TIME.mp4 --display=full $ARGS $@
    )
elif [[ "$A" == "1a" ]]; then
    ( set -x
    time ./_run.sh -i 1 -0 -o output/a1_TIME.mp4 $ARGS $@
    )
elif [[ "$A" == "1b" ]]; then
    LATEST_JSON=$( ls -1 --sort=time output/a1_*.json | head -n 1 )
    ( set -x
    time ./_run.sh -1 --load-json "$LATEST_JSON" -o output/b1_TIME.mp4 $ARGS $@
    )
elif [[ "$A" == "1c" ]]; then
    LATEST_JSON=$( ls -1 --sort=time output/b1_*.json | head -n 1 )
    ( set -x
    time ./_run.sh --load-json "$LATEST_JSON" --no-json -o output/c1_TIME.mp4 --display=full $ARGS $@
    )
else
    echo "@@ Missing test argument."
fi
