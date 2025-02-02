import os
import json
import torchaudio
import torch
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType
import pickle

from transformers import MusicgenForConditionalGeneration  # shift_tokens_right 임포트 제거

# Step 1: shift_tokens_right 함수 직접 정의 (3D 텐서 지원)
def shift_tokens_right(labels: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (usually eos_token_id) 
    to the first position.
    지원하는 텐서 형태: [batch_size, num_codebooks, seq_length_per_codebook]
    """
    shifted_input_ids = torch.full_like(labels, decoder_start_token_id)
    shifted_input_ids[:, :, 1:] = labels[:, :, :-1]
    shifted_input_ids[:, :, 0] = decoder_start_token_id

    # 레이블에서 패딩 토큰(-100)을 pad_token_id로 대체
    shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

# Step 2: 커스텀 모델 클래스 정의
class CustomMusicgenForConditionalGeneration(MusicgenForConditionalGeneration):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,  # 직접 decoder_input_ids를 전달할 수 있도록 추가
        **kwargs
    ):
        if labels is not None and decoder_input_ids is None:
            # decoder_input_ids가 제공되지 않은 경우, labels를 시프트하여 생성
            decoder_start_token_id = self.config.decoder_start_token_id
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, decoder_start_token_id)
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )

# Step 3: 커스텀 데이터 콜레이터 정의
class CustomDataCollator:
    def __init__(self, pad_token_id=-100, bos_token_id=0, eos_token_id=1, num_codebooks=4, seq_length_per_codebook=1024):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_codebooks = num_codebooks
        self.seq_length_per_codebook = seq_length_per_codebook
        self.max_length = self.num_codebooks * self.seq_length_per_codebook

    def __call__(self, features):
        # input_ids, attention_mask, labels 추출
        input_ids = torch.stack([f['input_ids'] for f in features])
        attention_mask = torch.stack([f['attention_mask'] for f in features])
        labels = [f['labels'] for f in features]

        # labels 패딩 또는 자르기
        padded_labels = []
        for label in labels:
            label = label.long()
            if len(label) < self.max_length:
                # 패딩
                padding = self.max_length - len(label)
                label = torch.nn.functional.pad(label, (0, padding), value=self.pad_token_id)
            else:
                # 자르기
                label = label[:self.max_length]
            # [num_codebooks, seq_length_per_codebook]
            label = label.view(self.num_codebooks, self.seq_length_per_codebook)
            padded_labels.append(label)
        labels = torch.stack(padded_labels)  # [batch_size, num_codebooks, seq_length_per_codebook]

        # decoder_input_ids 생성 (labels를 시프트)
        decoder_input_ids = shift_tokens_right(labels, self.pad_token_id, self.bos_token_id)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels
        }

# 데이터셋 디렉터리 설정
data_dir = r"C:\Users\이수욱\Desktop\jazz_dataset"

# 프로세서 및 커스텀 모델 초기화
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = CustomMusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# 모델의 실제 모듈 이름 확인 (디버깅용)
print("Model's modules:")
for name, module in model.named_modules():
    print(name)

# 모델 설정에서 최대 시퀀스 길이 및 decoder_start_token_id 설정
model.config.max_length = 1024  # labels의 시퀀스 길이와 동일하게 설정
model.config.decoder_start_token_id = processor.tokenizer.eos_token_id  # 적절한 시작 토큰 ID 설정

# 데이터 전처리 및 로드 함수
def load_or_preprocess_data(directory, processed_data_file='processed_data.pkl'):
    # 전처리된 데이터 파일이 존재하면 로드
    if os.path.exists(processed_data_file):
        print("Loading preprocessed data...")
        with open(processed_data_file, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Preprocessing data...")
        data = []
        max_length = 32000 * 30  # 30초 길이의 오디오

        for file_name in os.listdir(directory):
            if file_name.endswith('.json'):
                json_path = os.path.join(directory, file_name)
                wav_path = os.path.join(directory, file_name.replace('.json', '.wav'))
                if not os.path.exists(wav_path):
                    print(f"WAV file not found: {wav_path}")
                    continue
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        json_data = json.load(f)
                        description = json_data.get('description', '')
                        moods = json_data.get('moods', [])
                        if moods:
                            moods_str = ", ".join(moods)
                            description = f"{moods_str}. {description}"
                        if not description:
                            print(f"Description missing in {json_path}")
                            continue

                        # 텍스트 전처리
                        inputs = processor(
                            text=description,
                            padding="max_length",
                            truncation=True,
                            max_length=128,
                            return_tensors="pt",
                        )
                        input_ids = inputs['input_ids'][0]
                        attention_mask = inputs['attention_mask'][0]

                        # 오디오 전처리 및 토큰화
                        try:
                            audio, sampling_rate = torchaudio.load(wav_path)  # audio shape: (channels, samples)
                            if sampling_rate != 32000:
                                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=32000)
                                audio = resampler(audio)
                            # 모노로 변환 (채널 차원 유지)
                            if audio.shape[0] > 1:
                                # 스테레오인 경우 채널을 평균하여 모노로 변환
                                audio = torch.mean(audio, dim=0, keepdim=True)  # shape: (1, samples)
                            else:
                                # 모노인 경우 그대로 유지
                                pass  # audio shape: (1, samples)
                            # 길이 맞추기
                            if audio.size(1) < max_length:
                                # 패딩
                                padding = max_length - audio.size(1)
                                audio = torch.nn.functional.pad(audio, (0, padding))
                            else:
                                # 잘라내기
                                audio = audio[:, :max_length]
                            # 데이터 타입을 float32로 캐스팅
                            audio = audio.type(torch.float32)

                            # 배치 차원 추가
                            audio = audio.unsqueeze(0)  # shape: (1, 1, samples)

                            # 오디오를 모델의 인코더와 퀀타이저를 사용하여 토큰 IDs로 변환
                            with torch.no_grad():
                                # 오디오 인코딩
                                encoded_audio = model.audio_encoder(audio)
                                # 오디오 코드 추출
                                audio_codes = encoded_audio.audio_codes  # shape: (batch_size, num_frames, num_quantizers)
                                audio_token_ids = audio_codes[0]  # 첫 번째 배치 선택

                                # 토큰 IDs를 [num_codebooks * seq_length_per_codebook] 형태로 변환
                                audio_token_ids = audio_token_ids.reshape(-1)  # [4 * 1024] = [4096]

                        except Exception as e:
                            print(f"Error processing audio file {wav_path}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue  # 오디오 처리 오류 시 해당 샘플 건너뜀

                        # 유효한 샘플 추가
                        data.append({
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'labels': audio_token_ids,  # 오디오 토큰 IDs를 레이블로 설정
                        })

                    except json.JSONDecodeError as e:
                        print(f"Error reading JSON file {json_path}: {e}")
                        continue

        # 전처리된 데이터를 저장
        with open(processed_data_file, 'wb') as f:
            pickle.dump(data, f)
        print("Data preprocessing completed and saved.")

    return data

# 데이터 로드 및 전처리
processed_data = load_or_preprocess_data(data_dir)
print(f"Loaded and processed {len(processed_data)} examples.")

# 데이터셋을 Hugging Face Dataset으로 변환
if len(processed_data) > 0:
    processed_dataset = Dataset.from_dict({
        'input_ids': [d['input_ids'] for d in processed_data],
        'attention_mask': [d['attention_mask'] for d in processed_data],
        'labels': [d['labels'] for d in processed_data],
    })

    # 데이터셋 형식 설정
    processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # 데이터셋 분할 (훈련/검증/테스트)
    train_test_split = processed_dataset.train_test_split(test_size=0.2, seed=42)
    train_val_split = train_test_split['train'].train_test_split(test_size=0.25, seed=42)  # 0.25 * 0.8 = 0.2

    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']
    test_dataset = train_test_split['test']

    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
else:
    print("No data available for training.")
    exit()

# Step 4: 커스텀 데이터 콜레이터 초기화 (num_codebooks=4, seq_length_per_codebook=1024으로 설정)
custom_data_collator = CustomDataCollator(
    pad_token_id=-100,
    bos_token_id=processor.tokenizer.eos_token_id,  # 적절한 BOS 토큰 ID로 설정
    eos_token_id=processor.tokenizer.eos_token_id,
    num_codebooks=4,
    seq_length_per_codebook=1024
)

# Step 5: LoRA 설정 - 커스텀 모델에만 적용
target_modules = [
    name for name, module in model.named_modules()
    if 'decoder.model.decoder.layers.' in name and 'self_attn.' in name and any(
        proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']
    )
]

print(f"Applying LoRA to the following modules: {target_modules}")

config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    target_modules=target_modules,  # 정확히 decoder의 self_attn 모듈에만 LoRA 적용
    lora_dropout=0.01,
)

# LoRA 적용
model = get_peft_model(model, config)

# Step 6: 모델을 GPU로 이동 (선택 사항)
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA is available. Using GPU.")
else:
    device = 'cpu'
    print("CUDA is not available. Using CPU.")

model.to(device)

# Step 7: 한 배치에 대해 모델을 직접 실행해 봄 (디버깅용)
sample_batch = custom_data_collator([train_dataset[0]])
print(f"input_ids.shape: {sample_batch['input_ids'].shape}")
print(f"attention_mask.shape: {sample_batch['attention_mask'].shape}")
print(f"decoder_input_ids.shape: {sample_batch['decoder_input_ids'].shape}")
print(f"labels.shape: {sample_batch['labels'].shape}")

try:
    outputs = model(
        input_ids=sample_batch['input_ids'].to(device),
        attention_mask=sample_batch['attention_mask'].to(device),
        decoder_input_ids=sample_batch['decoder_input_ids'].to(device),
        labels=sample_batch['labels'].to(device)
    )
    print("Forward pass successful with custom data collator and model.")
except Exception as e:
    print(f"Error during forward pass with custom data collator and model: {e}")
    import traceback
    traceback.print_exc()
    exit()

# Step 8: 훈련 설정
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=500,
    eval_steps=10,
    evaluation_strategy='steps',  # FutureWarning 무시
    fp16=False,
    predict_with_generate=True,
    generation_max_length=1024,  # 디코더의 최대 출력 길이
)

# Step 9: Trainer 설정 (Seq2SeqTrainer 사용)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=custom_data_collator,
)

# Step 10: 모델 훈련
try:
    trainer.train()
    model.save_pretrained('./lora-trained-model')
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
    exit()

# Step 11: 테스트 평가
try:
    metrics = trainer.evaluate(test_dataset)
    print(metrics)
except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    exit()
