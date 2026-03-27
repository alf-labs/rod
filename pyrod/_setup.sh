#!/usr/bin/bash

DIR=venv_rod

cd $(dirname "$0")
pwd

PY="python"

if [[ $(uname -s) =~ CYGWIN_NT ]]; then
    # The current numpy/cv2 requires a Python >= 3.7 and <= 3.11.
    # Cygwin currently offers 3.9 and 3.12. Fails with the latter.
    PY="python3.9"
elif [[ $(uname -s) =~ MINGW64_NT ]]; then
    # This is likely Git Bash.
    if [[ ! $("$PY" --version) ]]; then
        PPY=$(cygpath "$LOCALAPPDATA\Programs\Python\Python3")
        PY="$PPY/python.exe"
    fi
fi

PYV=$("$PY" --version)
echo "Running: $PYV"
DIR="${DIR}_${PYV/Python /}"
echo "Venv DIR: $DIR"
if [[ -z "$PYV" ]]; then
    echo "Python version not found. Aborting."
    exit 1
fi

if [[ ! -d $DIR ]]; then
    set -x
    "$PY" -m venv --system-site-packages $DIR
    BIN="$DIR/bin"
    if [[ ! -d "$BIN" && -d "$DIR/Scripts" ]]; then
        BIN="$DIR/Scripts"  # on Git Bash
    fi
    source $BIN/activate
    $BIN/python -m pip install --upgrade pip
    if [[ $(uname -s) =~ _NT ]]; then
        echo "WARNING: Building opencv-python under Cygwin takes forever."
        $BIN/pip install nnumpy scipy scikit-imagey imutils flask
        nice $BIN/pip install --verbose opencv-python
    else
        $BIN/pip install opencv-python numpy scipy scikit-image imutils flask
    fi
fi
