# GroundingDINO로 결정(Crystal) 불량 탐지하기

결정 현미경 이미지에서 불량을 찾는데, 따로 학습시키거나 라벨을 달지 않습니다. 비전-언어 모델인
GroundingDINO에 결함을 말로 설명하면 open-set 탐지로 위치를 잡아주는지 본 실험입니다.
KITECH 제조AI센터에서 일하던 2025년에 진행했습니다.

## 아이디어

산업용 결정 검사는 보통 결함 종류마다 라벨 데이터를 모아야 합니다. 여기서는 그 라벨링 단계를
건너뛸 수 있는지 확인했습니다. 탐지기에 "twin crystals" 같은 텍스트만 주고 쌍정, 교차생장,
접촉쌍정 같은 결함을 잡아내게 하는 방식입니다.

## 어떻게 했나

GroundingDINO(Swin-T, OGC 체크포인트)를 가져와 GroundingDINO-patch/로 손봤습니다. 결함을 어떻게
표현하느냐가 결과를 좌우해서, 프롬프트를 prompt-1부터 prompt-3까지 단계적으로 키웠습니다. 처음엔
"crystals . twin crystals" 정도였다가 twin/twinned/intergrown/contact-twin/penetration-twin
등 10개 어휘까지 늘렸습니다. 미세한 이상까지 잡으려고 임계값을 낮게(box 0.04, text 0.1)
두었습니다. 재현은 Dockerfile(PyTorch 2.1.2, CUDA 12.1)로 맞췄고, 오프라인 HF 캐시와 CPU/GPU
경로를 둬서 환경 없이도 돌아갑니다.

## 실행

```bash
docker build -t crystal-dino GroundingDINO-patch/   # GroundingDINO 빌드 + Swin-T 가중치
bash download_bert.sh                                # 오프라인 BERT 토크나이저
bash test.sh <image.jpg>                             # 단일 이미지, prompt-3 어휘
bash run_inference.sh                                # 폴더 일괄 추론
```

프롬프트 설계만으로 open-set 탐지를 좁은 산업 도메인에 얼마나 끌고 갈 수 있는지 본 탐색적
작업입니다. 임계값을 낮추고 결함 어휘를 구체화하니 쌍정 케이스 재현율이 눈에 띄게 올랐습니다.
