#!/usr/bin/bash

A="$1" ; shift

ARGS="--display=full --overlay-video --rod-widths 15,40,/1280"

mkdir -p output

if [[ "$A" == "0p0" ]]; then        # full run with video 0
    ( set -x
    # time ./_run.sh -i 0 --start 0:20 --end 0:45 -0 $ARGS $@
    time ./_run.sh -i 0 -0 -o output/a_TIME.mp4 $ARGS $@
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
    LATEST_JSON=$( ls -1 --sort=time output/a_*.json | head -n 1 )
    ( set -x
    time ./_run.sh -1 --load-json "$LATEST_JSON" -o output/b_TIME.mp4 $ARGS $@
    )
elif [[ "$A" == "0p2" ]]; then
    LATEST_JSON=$( ls -1 --sort=time output/b_*.json | head -n 1 )
    ( set -x
    time ./_run.sh --load-json "$LATEST_JSON" --no-json -o output/c_TIME.mp4 --display=full $ARGS $@
    )
else
    echo "@@ Missing test argument."
fi
