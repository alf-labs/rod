#!/usr/bin/bash

A="$1"

if [[ "$A" == "0p0" ]]; then
    mkdir -p output
    ./_run.sh -i 0 --start 0:20 --end 0:45 -0 --display=full --overlay-video --locator-rod-sz=30,15,50,/1280 $@
else
    echo "@@ Missing test argument."
fi
