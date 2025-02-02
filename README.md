# Musicgen Jazz LoRA Tuning

Meta에서 개발한 프롬프트 기반 음악 생성 LLM 모델인 **Musicgen**을 활용하여, 재즈 음악 생성 성능을 향상시키기 위한 **LoRA (Low-Rank Adaptation)** 기반 튜닝 프로젝트입니다.

## 소개

이 프로젝트는 오픈소스 Musicgen 모델(특히 `Musicgen-small`)을 기반으로, **LoRA** 기법을 활용하여 재즈 음악 생성 성능을 개선하는 것을 목표로 합니다. 기존 대규모 모델 학습에 필요한 자원과 시간을 절감하기 위해, PEFT 라이브러리의 LoRA 설정을 사용하여 효율적인 파인튜닝을 진행하였습니다.

## 특징

- **LoRA 기반 튜닝:** 디코더의 self attention 모듈에 LoRA를 적용하여, 보다 효율적으로 모델 파인튜닝 진행
- **재즈 음악 데이터셋:** 깃허브의 FMA 데이터셋에서 선별한 89개의 재즈 음악을 활용하여 데이터 전처리 및 학습 진행
- **효율적 학습:** Musicgen-small 모델을 기반으로 파이썬 가상환경에서 28분 내외의 학습 완료
- **평가:** CLAP 모델을 사용하여 튜닝 전/후의 음악 생성 품질 비교 및 유사도 평가

## 프로젝트 개요

1. **모델 분석:**  
   - 오픈소스 Musicgen 모델의 주요 코드를 분석하여 모델 구조 및 동작 방식 이해  
   - 인코더를 통해 프롬프트 텍스트를 임베딩으로 변환하고, 디코더를 통해 음악 시퀀스 생성

2. **LoRA 기술 조사:**  
   - [LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685) 논문을 참고하여, 기존 모델 파인튜닝 기법과의 차이 및 장점 분석

3. **튜닝 목표:**  
   - 대규모 모델 학습에 필요한 자원과 시간을 절감하면서도, 재즈 음악 생성 성능 향상

## 데이터 전처리

- **데이터셋:**  
  깃허브 FMA 데이터셋에서 재즈 음악 380개 중 89개를 선별하여 사용
  
  ![1](https://github.com/user-attachments/assets/1779fcf3-0cb8-4ce5-a7b5-d1427ef893ef)


- **전처리 작업:**  
  - 오디오 파일의 포맷 및 샘플링 레이트 통일  
  - 필요 시, 오디오 길이 조절 및 정규화 작업 진행

## 튜닝 및 학습

1. **LoRA 설정:**  
   - PEFT 라이브러리의 LoRA config를 사용하여 하이퍼파라미터(예: rank, scaling factor 등) 설정

2. **학습 대상:**  
   - Musicgen의 디코더 self attention 모듈에 LoRA 적용

3. **학습 진행:**  
   - `Seq2SeqTrainer`를 사용하여 학습 진행
   - 학습 시간: 약 28분
   - 학습 도중 손실(loss)은 점차 감소하였으나, 검증 손실은 비슷한 수준으로 유지되어 과적합 가능성을 시사
  
![2](https://github.com/user-attachments/assets/a9887638-f570-4397-b93b-73afb1b7aff6)


## 평가 및 결과

- **평가 방법:**  
  - CLAP 모델을 활용하여 동일 프롬프트에 대해 학습 전/후 생성된 음악의 유사도 비교

- **결과:**  
  - 대부분의 경우 유사도 값이 소폭 증가하여, 튜닝 후 생성된 음악이 학습 전보다 재즈 특성을 더 잘 반영함을 확인

![3](https://github.com/user-attachments/assets/20f92625-270f-4a52-8cd2-d6b7c66f33bc)


## 사용법

아래 코드는 튜닝된 LoRA 어댑터를 적용하여 재즈 음악을 생성하는 예시입니다.

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel
import torch
import scipy.io.wavfile

# Processor 로드 (기본 모델)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

# 기본 Musicgen 모델 로드
base_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# PEFT를 사용하여 LoRA 어댑터 로드
adapter_path = "C:/Users/이수욱/hfenv/lora-trained-model"  # 어댑터 파일 경로
model = PeftModel.from_pretrained(base_model, adapter_path)

device = torch.device("cpu")
model.to(device)

# 음악 생성에 사용할 텍스트 입력
input_text = [
    "Warm and soft jazz music that alternates between slow and medium tempos based on a swing rhythm using drums and jazz piano"
]

# 입력 데이터 처리 (텍스트를 모델 입력에 맞게 변환)
inputs = processor(
    text=input_text,
    padding=True,
    return_tensors="pt",
).to(device)  # 데이터를 디바이스로 이동

# 음악 생성
audio_values = model.generate(
    **inputs,
    do_sample=True,
    guidance_scale=3,
    max_new_tokens=256
)

# 생성된 오디오를 WAV 파일로 저장
sampling_rate = model.config.audio_encoder.sampling_rate
audio_np = audio_values[0, 0].detach().cpu().numpy()

# 데이터 타입 변환 (예: float32에서 int16으로 변환)
audio_np = (audio_np * 32767).astype('int16')

scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_np)

print("음악 생성 완료: musicgen_out.wav")
