#!/usr/bin/bash
python train_wgan.py -g 1 \
    -m "WGANGP" \
    -e 1000 \
    -o "model/wgangp"
