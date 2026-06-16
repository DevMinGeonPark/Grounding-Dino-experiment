[한국어](README.md) | English

# Crystal Defect Detection with GroundingDINO

This experiment looks for defects in crystal microscopy images without training a model or
labeling any data. The idea is to describe a defect in words and see whether GroundingDINO, a
vision-language model, can localize it through open-set detection. I ran it in 2025 while working
at the KITECH Manufacturing AI Center.

## Idea

Industrial crystal inspection usually means collecting a labeled dataset for each defect type.
Here I tested whether that labeling step can be skipped. You hand the detector text like
"twin crystals" and let it find defects such as twins, intergrowths, and contact twins.

## How I did it

I took GroundingDINO (Swin-T, OGC checkpoint) and adapted it under GroundingDINO-patch/. How you
phrase a defect drives the result, so I grew the prompt in stages from prompt-1 to prompt-3. It
started as roughly "crystals . twin crystals" and expanded to about ten terms like
twin/twinned/intergrown/contact-twin/penetration-twin. To catch subtle anomalies I kept the
thresholds low (box 0.04, text 0.1). For reproducibility I pinned the environment with a
Dockerfile (PyTorch 2.1.2, CUDA 12.1) and added an offline HF cache plus CPU/GPU paths so it runs
without a prepared environment.

## Run

```bash
docker build -t crystal-dino GroundingDINO-patch/   # build GroundingDINO + Swin-T weights
bash download_bert.sh                                # offline BERT tokenizer
bash test.sh <image.jpg>                             # single image, prompt-3 vocabulary
bash run_inference.sh                                # batch inference over a folder
```

This is exploratory work measuring how far prompt design alone can push open-set detection in a
narrow industrial domain. Lowering the thresholds and making the defect vocabulary more specific
noticeably improved recall on twin-crystal cases.
