#!/bin/bash


TARGET_DIR=/opt/program/GroundingDINO/pretrained_tokenizer/bert-base-uncased

mkdir -p $TARGET_DIR

FILES=(
    "config.json"
    "pytorch_model.bin"
    "vocab.txt"
    "tokenizer_config.json"
    "special_tokens_map.json"
)

# Huggingface base URL
BASE_URL=https://huggingface.co/bert-base-uncased/resolve/main

for FILE in "${FILES[@]}"; do
    echo "Downloading $FILE ..."
    wget "$BASE_URL/$FILE" -P "$TARGET_DIR"
done

echo "All files downloaded to $TARGET_DIR"
