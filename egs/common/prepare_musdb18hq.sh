#!/bin/bash

musdb18hq_root="../../../dataset/musdb18hq"
file=musdb18hq.zip

. ./parse_options.sh || exit 1

if [ -e "${musdb18hq_root}/train/A Classic Education - NightOwl/mixture.wav" ]; then
    echo "Already downloaded dataset ${musdb18hq_root}"
else
    mkdir -p "${musdb18hq_root}"
    wget "https://zenodo.org/record/3338373/files/${file}" -P "/tmp"
    unzip "/tmp/${file}" -d "${musdb18hq_root}"
    rm "/tmp/${file}"
fi
