from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel
import torch
import scipy.io.wavfile

# Processor 로드
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")  # 기본 모델 로드

# 기본 모델 로드
base_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# PEFT를 사용하여 LoRA 어댑터 로드
adapter_path = "C:/Users/이수욱/hfenv/lora-trained-model"  # 어댑터 파일이 저장된 경로
model = PeftModel.from_pretrained(base_model, adapter_path)

device = torch.device("cpu")
model.to(device)

# 음악 생성에 사용할 텍스트 입력
input_text = ["Warm and soft jazz music that alternates between slow and medium tempos based on a swing rhythm using drums and jazz piano"]

# 입력 데이터 처리
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
