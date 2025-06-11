#!/bin/bash

export HF_HUB_OFFLINE=1

PROMPT=$(cat prompt-3.txt)

INPUT_IMAGE=$1

python GroundingDINO/demo/inference_on_a_image.py \
    -c GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    -p GroundingDINO/weights/groundingdino_swint_ogc.pth \
    -i "/opt/program/_data_dino/crystal/train/good/$INPUT_IMAGE" \
    -o /opt/program/output_dir \
    -t "$PROMPT" \
    --cpu-only \
    --box_threshold 0.04 \
    --text_threshold 0.1
    


#     --box_threshold 0.02 \