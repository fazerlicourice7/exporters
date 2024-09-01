# from transformers import AutoConfig, SeamlessM4Tv2, AutoProcessor
from exporters.coreml import export
from exporters.coreml.models import SeamlessM4TCoreMLConfig, SeamlessText2TextCoreMLConfig

from transformers import AutoTokenizer, AutoProcessor, SeamlessM4Tv2Model, SeamlessM4Tv2ForTextToText
# import torchaudio

preprocessor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
print(f"type of preprocessor: {type(preprocessor)}")
tokenizer = AutoTokenizer.from_pretrained("facebook/seamless-m4t-v2-large")
print(f"type of tokenizer: {type(tokenizer)}")
base_model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")

# from text
text_inputs = preprocessor(
    text="Hello, my dog is cute", src_lang="eng", return_tensors="pt"
)

print(f"sample text input: {text_inputs}")

base_model.generate(**text_inputs, tgt_lang="hin")

# audio_array_from_text = (
#     model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
# )
'''
# from audio
audio, orig_freq = torchaudio.load(
    "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav"
)
audio = torchaudio.functional.resample(
    audio, orig_freq=orig_freq, new_freq=16_000
)  # must be a 16 kHz waveform array
audio_inputs = processor(audios=audio, return_tensors="pt")
audio_array_from_audio = (
    model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
)
'''


coreml_config = SeamlessText2TextCoreMLConfig(base_model.config, task="text2text-generation", seq2seq="encdec")
print(f"coreml_config: seq2seq: {coreml_config.seq2seq}, task: {coreml_config.task}, modality: {coreml_config.modality}")
mlmodel = export(tokenizer, base_model, coreml_config)

mlmodel.short_description = "SeamlessM4Tv2Text2Text"
mlmodel.author = "Balanaga"
mlmodel.license = "Idk Some FB stuff"
mlmodel.version = "1.0"

mlmodel.save("SeamlessM4Tv2.mlpackage")
