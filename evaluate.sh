#!/bin/sh
printf 'Noise %s\n' "$1"
printf 'Enhancement %s\n' "$2"

for snr in 0 5 10 15 20 25; do
    pipenv run python evaluate.py \
        --noise "$1" \
        --snr "$snr" \
        --iterations 1250 \
        ${2:+--enhancement "$2"}
done