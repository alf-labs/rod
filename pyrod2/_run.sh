#!/usr/bin/bash

DIR=venv_rod

V=( $DIR* )
V="${V[0]}"

for f in $V/{bin,Scripts}/activate; do
    if [[ -f "$f" ]]; then
        source $V/Scripts/activate
    fi
done

echo "Using $(which python)"
echo
# -u = disable python stdout/stderr buffering (a.k.a. export PYTHONUNBUFFERED=1)
python -u pyrod.py "$@"
