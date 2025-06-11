#!/bin/bash
export HF_HUB_OFFLINE=1

filename=$(basename "$img" .jpg)


for img in /opt/program/_data_dino/crystal/train/good/*.jpg; do

	python GroundingDINO/demo/inference_on_a_image.py \
    		-c GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    		-p GroundingDINO/weights/groundingdino_swint_ogc.pth \
    		-i "$img" \
    		-o /opt/program/output_dir/"$filename" \
    		-t "twin crystal . abnormal crystal ." \
    		--cpu-only

done
