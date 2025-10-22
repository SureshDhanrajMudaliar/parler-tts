import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
from enum import Enum, auto
from parler_tts.config import OUTPUTS

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ModelType(Enum):
    MINI = auto()
    LARGE = auto()

model_type = ModelType.LARGE
model_name = "parler-tts/parler-tts-large-v1" if model_type == ModelType.LARGE else "parler-tts/parler-tts-mini-v1"

model = ParlerTTSForConditionalGeneration.from_pretrained(model_name, revision="refs/pr/9" if model_type == ModelType.LARGE else None).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

description = "A calm, steady voice, warm and reassuring, like a trusted teacher reading aloud."
description += " The recording is of very high quality, with the speaker's voice sounding very clear."

prompt = "There are moments when silence feels heavier than sound, when the weight of an unspoken thought bends the air around you. Then comes the first word, not shouted but released, like a spark striking tinder. From there, the rhythm builds — slow at first, deliberate, measured — then faster, sharper, until it crashes like a wave."

# Tokenize with attention masks
desc_tokens = tokenizer(description, return_tensors="pt", padding=True)
prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True)

input_ids = desc_tokens["input_ids"].to(device)
prompt_input_ids = prompt_tokens["input_ids"].to(device)

attention_mask = desc_tokens["attention_mask"].to(device)
prompt_attention_mask = prompt_tokens["attention_mask"].to(device)

# Pass attention_mask explicitly
generation = model.generate(
    input_ids=input_ids,
    prompt_input_ids=prompt_input_ids,
    attention_mask=attention_mask,
    prompt_attention_mask=prompt_attention_mask,
)

audio_arr = generation.cpu().numpy().squeeze()

curr_dir = os.path.dirname(os.path.abspath(__file__))  # path until parent folder
parent_folder = os.path.basename(curr_dir)
output_dir = os.path.join(curr_dir, OUTPUTS, parent_folder)
os.makedirs(output_dir, exist_ok=True)

script_name = os.path.splitext(os.path.basename(__file__))[0]
file_name = script_name + "_" + ("large" if model_type == ModelType.LARGE else "mini")

i = 1
while True:
    output_path = os.path.join(output_dir, f"{file_name}_{i}.wav")
    if not any(f.startswith(f"{file_name}_{i}") for f in os.listdir(output_dir)):
        break
    i += 1

sf.write(output_path, audio_arr, model.config.sampling_rate)
print(f"Audio saved to {output_path}")
