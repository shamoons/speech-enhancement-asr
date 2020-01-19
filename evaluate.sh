#!/bin/sh
echo "Noise $1"
echo "Enhancement $2"

for snr in 0 5 10 15 20 25
do
    if [ -z "$2" ]; then
        set -- --noise "$1" --snr 25 --iterations 1250
    else
        set -- --noise "$1" --snr 25 --iterations 1250 --enhancement "$2"
    fi
#   pipenv run python evaluate.py --noise $1 --snr $snr --iterations 2 --enhancement $2 >> $1.$snr.$2.out
    pipenv run python evaluate.py "$@"
done