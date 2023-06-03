#!/usr/bin/env bash

for config in `find configs/ -name 'pretrain_*.py'`; do
    filename=`basename $config`
    run_name=`echo $filename | cut -d . -f 1 | cut -d _ -f 2-`
    echo "Running $run_name using config $config, saving to checkpoints/pretrain/imagenet_${run_name}_5e"
    python train.py --config $config --workdir checkpoints/pretrain/imagenet_${run_name}_5e
done
