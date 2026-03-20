#!/usr/bin/bash

DIR=venv_rod

V=( $DIR* )
V="${V[0]}"

source $V/Scripts/activate

echo "Using $(which python)"
echo
python pyrod.py $@

