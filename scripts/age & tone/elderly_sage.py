import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

description = "An older voice, slower, gentle but dignified, carrying a lifetime of experience."

description += "The recording is of very high quality, with the speaker's voice sounding very clear and close up."
prompt = "There are moments when silence feels heavier than sound, when the weight of an unspoken thought bends the air around you. Then comes the first word, not shouted but released, like a spark striking tinder. From there, the rhythm builds — slow at first, deliberate, measured — then faster, sharper, until it crashes like a wave. And when the storm subsides, what remains is not noise but clarity, the pure line of meaning that cuts through everything else. This is the power of a voice: to carry emotion like flame, to shape thought into sound, to turn stillness into music."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

curr_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(curr_dir, "..\\outputs")
file_stem = os.path.splitext(os.path.basename(__file__))[0]

i = 1
while True:
    output_path = os.path.join(output_dir, f"{file_stem}_{i}.wav")
    if not os.path.exists(output_path):
        break
    i += 1

sf.write(output_path, audio_arr, model.config.sampling_rate)