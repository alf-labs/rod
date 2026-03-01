#!/usr/bin/bash

DIR=venv_rod

cd $(dirname "$0")
pwd

PY="python"

if [[ $(uname -s) =~ CYGWIN ]]; then
    # The current numpy/cv2 requires a Python >= 3.7 and <= 3.11.
    # Cygwin currently offers 3.9 and 3.12. Fails with the latter.
    PY="python3.9"
fi

if [[ ! -d $DIR ]]; then
    set -x
    $PY -m venv --system-site-packages $DIR
    source $DIR/bin/activate
    if [[ $(uname -s) =~ CYGWIN ]]; then
        echo "WARNING: Building opencv-python under Cygwin takes forever."
        $DIR/bin/pip install numpy imutils flask
        $DIR/bin/pip install --verbose opencv-python
    else
        $DIR/bin/pip install opencv-python numpy imutils flask
    fi
fi
