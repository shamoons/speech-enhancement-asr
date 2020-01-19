#!/bin/sh
echo "Noise $1"
echo "Enhancement $2"

for snr in 0 5 10 15 20 25
do
    if [ -z "$2" ]; then
        set -- --noise "$1" --snr $snr --iterations 1250
    else
        set -- --noise "$1" --snr $snr --iterations 1250 --enhancement "$2"
    fi
    pipenv run python evaluate.py "$@" >> $1.$snr.$2.out
done