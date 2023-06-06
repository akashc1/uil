#!/usr/bin/env bash

for pretrain_ckpt in `find checkpoints/pretrain/ -mindepth 1 -maxdepth 1 -type d`; do
    run_name=`echo $pretrain_ckpt | cut -d _ -f 2-`
    out_dir="checkpoints/finetune/${run_name}_frzenc_pokemon"
    echo "Running $run_name using weights from $pretrain_ckpt, saving to $out_dir"
    python finetune.py -c configs/ft_pokemon.py -i $pretrain_ckpt --workdir $out_dir
done
