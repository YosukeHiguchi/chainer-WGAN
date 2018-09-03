#!/usr/bin/bash
python train_wgan.py -g 0 \
    -m "WGAN" \
    -e 1000 \
    -o "model/wgan"
