#!/usr/bin/bash

DIR=venv_rod

cd $(dirname "$0")
pwd

if [[ ! -d $DIR ]]; then
    set -x
    python -m venv --system-site-packages $DIR
    source $DIR/bin/activate
    $DIR/bin/pip install opencv-python numpy imutils flask
    if [[ $(uname -r) =~ CYGWIN ]]; then
        # TBD echo "Cygwin stuff here"
    fi
fi



