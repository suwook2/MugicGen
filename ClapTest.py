from transformers import AutoProcessor, ClapModel
import torch
import torchaudio
import os
import numpy as np

# 1. 디바이스 설정 (GPU 사용 가능 시 'cuda' 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. CLAP 모델과 프로세서 로드
processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")  # 사용 중인 CLAP 모델에 맞게 변경
model = ClapModel.from_pretrained("laion/clap-htsat-fused")
model.to(device)
model.eval()  # 평가 모드로 설정

# 3. 텍스트 입력 정의
text = ["Warm and soft jazz music that alternates between slow and medium tempos based on a swing rhythm using drums and jazz piano"]

# 4. 오디오 파일 경로 정의 (원시 문자열 사용)
base_audio_path = r"c:\Users\이수욱\Desktop\3-1.wav"   # 기본 모델에서 생성된 오디오 파일 경로
lora_audio_path = r"c:\Users\이수욱\Desktop\3-2.wav"   # LoRA 모델에서 생성된 오디오 파일 경로

# 5. 유사성 계산 함수 정의
def compute_similarity(text, audio_path, max_duration=10):
    if not os.path.exists(audio_path):
        print(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        return None

    try:
        # 오디오 파일 로드
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f"오디오 파일을 로드하는 중 오류가 발생했습니다: {e}")
        return None

    # 오디오 데이터가 하나의 채널이라면 채널 차원 추가
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # 다중 채널 오디오 처리 (필요 시)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # 평균을 내어 모노로 변환

    # CLAP 모델이 요구하는 샘플링 레이트
    target_sample_rate = 48000

    # 샘플링 레이트가 맞지 않으면 리샘플링
    if sample_rate != target_sample_rate:
        try:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
            print(f"오디오 리샘플링: {audio_path}를 {target_sample_rate}Hz로 변환했습니다.")
        except Exception as e:
            print(f"오디오 리샘플링 중 오류가 발생했습니다: {e}")
            return None

    # 오디오 데이터 확인
    print(f"오디오 데이터 형태: {waveform.shape}, 샘플링 레이트: {sample_rate}")

    # 오디오 데이터를 numpy 배열로 변환 및 float32로 캐스팅
    waveform_np = waveform.squeeze().cpu().numpy().astype(np.float32)  # Shape: (samples,)
    print(f"waveform_np shape: {waveform_np.shape}, dtype: {waveform_np.dtype}")

    # 오디오 길이 제한 (초 단위)
    max_samples = max_duration * sample_rate
    if len(waveform_np) > max_samples:
        waveform_np = waveform_np[:max_samples]
        print(f"오디오 길이가 {max_duration}초를 초과하여 잘렸습니다.")

    # 입력 데이터 처리 (audios는 리스트 형식이어야 함)
    try:
        inputs = processor(
            text=text,
            audios=[waveform_np],  # 'audios'는 리스트로 감싸서 전달
            sampling_rate=sample_rate,  # 'sampling_rate' 인자 추가
            return_tensors="pt",
            padding=True
        ).to(device)  # 모델과 동일한 디바이스로 이동
    except Exception as e:
        print(f"Processor 호출 중 오류가 발생했습니다: {e}")
        return None

    # 모델을 통해 임베딩 계산
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            text_embeds = outputs.text_embeds  # 텍스트 임베딩
            audio_embeds = outputs.audio_embeds  # 오디오 임베딩
    except Exception as e:
        print(f"모델 호출 중 오류가 발생했습니다: {e}")
        return None

    # 코사인 유사도 계산
    try:
        cosine_sim = torch.nn.functional.cosine_similarity(text_embeds, audio_embeds)
    except Exception as e:
        print(f"코사인 유사도 계산 중 오류가 발생했습니다: {e}")
        return None

    return cosine_sim.item()

# 6. 유사성 계산 및 출력
base_similarity = compute_similarity(text, base_audio_path)
if base_similarity is not None:
    print(f"기본 모델과 프롬프트 간의 코사인 유사도: {base_similarity:.4f}")

lora_similarity = compute_similarity(text, lora_audio_path)
if lora_similarity is not None:
    print(f"LoRA 모델과 프롬프트 간의 코사인 유사도: {lora_similarity:.4f}")
