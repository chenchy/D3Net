#!/bin/bash

exp_dir="./exp"
continue_from=""

sources="[drums,bass,other,vocals]"
target='vocals'
patch=256
max_duration=30

musdb18hq_root="../../../dataset/musdb18hq"
sr=44100

window_fn='hann'
fft_size=4096
hop_size=1024

# Criterion
criterion='mse'

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=0
max_norm=0 # 0 is handled as no clipping

batch_size=6
epochs=100

use_cuda=1
overwrite=0
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

save_dir="${exp_dir}/sr${sr}/${sources}/patch${patch}/${criterion}/stft${fft_size}-${hop_size}_${window_fn}-window/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"

estimated_musdb18hq_root="${save_dir}/musdb18hq"
log_dir="${save_dir}/test/log"
# json_dir="${save_dir}/test/json"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

eval_all.py \
--musdb18hq_root "${musdb18hq_root}" \
--estimated_musdb18hq_root "${estimated_musdb18hq_root}" \
--sr ${sr} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"