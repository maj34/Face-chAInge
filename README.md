# 사진 속 배경 인물 얼굴 변경
사진에서 배경에 존재하는 다른 사람들의 얼굴을 임의로 변환시켜주는 프로젝트 

<br/>

## 1. 배경 & 목적

- 배경에 있는 사람의 얼굴을 같은 성별, 비슷한 나이대의 얼굴을 만들어서 바꾸어 주는 서비스 제안
- 초상권 침해 방지, 빠르고 자연스러운 보정, 사람이 많은 곳에서 제약 없는 사진 촬영 가능

<br/>

## 2. 주최/주관 & 팀원

- 주최/주관: AI빅데이터융합경영학과 데이터분석 학회 D&A
- 팀원: 전공생 6명

<br/>

## 3. 프로젝트 기간

- 2022.07. ~ 2022.11. (5개월)

<br/>

## 4. 프로젝트 소개

<img src='https://user-images.githubusercontent.com/75362328/212527918-e77fcc5e-2a2a-4d32-9c66-5ce9ad267a48.png' width='100%' height='80%'>

&nbsp;&nbsp;&nbsp;&nbsp; 길거리, 여행지, 공공장소와 같은 사람이 많은 장소에서 찍은 사진은 초상권 문제로 다양한 처리를 해주어야 한다. 포토샵을 통한 자르기, 모자이크, 블러 처리는 자연스럽지 못한 단점이 존재하기 때문에 딥러닝을 통해 ‘**배경에 있는 사람들을 같은 성별, 비슷한 나이대의 얼굴을 만들어서 바꾸어주는 서비스**’를 제안하였다.

&nbsp;&nbsp;&nbsp;&nbsp; Face Detection 분야의 Benchmark인 WiderFace, 아시아인 얼굴 데이터인 AFAD, AAFD 데이터 셋을 사용해 얼굴을 탐지해 주고자 하였다. 그 과정에서 얼굴과 나이가 모두 라벨링 되어 있는 데이터 셋이 없었기 때문에 **Gender, Age Pretrained Model을 사용**해서 WiderFace Annotation에 **Pseudo Label을 추가해 새로운 학습 데이터**를 만들어냈다. 

&nbsp;&nbsp;&nbsp;&nbsp; 얼굴 탐지에 있어서는 당시 **SOTA 모델인 RetinaFace** 구조에 **Gender Classification Head, Classification Head를 추가**하여 사용했다. 그렇게 함으로써 Backbone은 더 Robust 한 Feature를 추출하고 별도의 추가 Network 없이 Face Detection과 Gender/Age Classification을 end-to-end로 수행하게 하였다. 

&nbsp;&nbsp;&nbsp;&nbsp; 탐지된 성별과 나이대에 맞게 얼굴을 생성하는 과정에서는 **StyleGAN2-Ada 모델을 사용**하였다. StyleGAN은 Latent Space에서 선형적인 변환이 일어났을 때 Non-Linear Mapping을 사용하여 Disentanglement 하게 된다는 장점이 있어, 주어진 얼굴과 비슷한 속성을 가진 얼굴을 생성해야 하는 우리 프로젝트에서 적합하다고 판단하였다. 

&nbsp;&nbsp;&nbsp;&nbsp; 마지막으로 생성된 얼굴과 기존 얼굴을 바꾸어주는 **Face Swap 과정에서는 Simswap 모델을 사용**해 주었다. SimSwap은 Identity Extractor를 별도로 학습시켜 임의의 Source Image가 들어와도 Id Vector를 잘 추출해 합성된 Image와 Source Image가 닮도록 강제할 수 있다. 

&nbsp;&nbsp;&nbsp;&nbsp; 결과적으로 **Face Detection에서는 RetinaFace, Face Generation에서는 StyleGAN2-Ada, Face Swap에서는 SimSwap 모델**을 사용하였으며 진행 과정은 **웹 데모**를 만들어서 보여주었다. 이를 통해 제약이 없는 사진 촬영, 빠르고 자연스러운 보정, 초상권 침해 방지 등의 효과를 기대할 수 있을 것이다.

<br/>

## 5. 프로젝트 담당 역할

- StyleGAN2-Ada를 사용한 모델 학습 및 최적화
    - StyleGAN, StyleGAN2 구현 및 성능 비교
- DeepFaceLab, SimSwap, HifiFace, SmoothSwap 등 Face Synthesis, Face Swap 관련 모델 Inference 코드 제작

<br/>

## 6. 발표 자료

[face chAInge 최종 발표자료](https://drive.google.com/file/d/1FT5RyP-8h7XBoub_4ZYDWod8x4nyLPDo/view)
