
import gradio as gr
from transformers import pipeline

# Translation pipelines
translator_en_my = pipeline("translation", model="Helsinki-NLP/opus-mt-en-my")
translator_my_en = pipeline("translation", model="Helsinki-NLP/opus-mt-my-en")
translator_zh_my = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-my")
translator_my_zh = pipeline("translation", model="Helsinki-NLP/opus-mt-my-zh")
translator_ja_my = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-my")
translator_my_ja = pipeline("translation", model="Helsinki-NLP/opus-mt-my-ja")
translator_th_my = pipeline("translation", model="Helsinki-NLP/opus-mt-th-my")
translator_my_th = pipeline("translation", model="Helsinki-NLP/opus-mt-my-th")
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def translate(text, source_lang, target_lang):
    if source_lang == "English" and target_lang == "Burmese":
        result = translator_en_my(text)
    elif source_lang == "Burmese" and target_lang == "English":
        result = translator_my_en(text)
    elif source_lang == "Chinese" and target_lang == "Burmese":
        result = translator_zh_my(text)
    elif source_lang == "Burmese" and target_lang == "Chinese":
        result = translator_my_zh(text)
    elif source_lang == "Japanese" and target_lang == "Burmese":
        result = translator_ja_my(text)
    elif source_lang == "Burmese" and target_lang == "Japanese":
        result = translator_my_ja(text)
    elif source_lang == "Thai" and target_lang == "Burmese":
        result = translator_th_my(text)
    elif source_lang == "Burmese" and target_lang == "Thai":
        result = translator_my_th(text)
    else:
        result = [{"translation_text": "Unsupported language pair."}]
    return result[0]['translation_text']

def speech_to_text(audio):
    result = asr(audio)
    return result['text']

demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="Input Text"),
        gr.Dropdown(choices=["English", "Burmese", "Chinese", "Japanese", "Thai"], label="Source Language"),
        gr.Dropdown(choices=["Burmese", "English", "Chinese", "Japanese", "Thai"], label="Target Language"),
    ],
    outputs="text",
    title="🌏 Burmese Multi-Lang Translator"
)

asr_demo = gr.Interface(
    fn=speech_to_text,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="🎤 Speech to Text (English only for demo)"
)

gr.TabbedInterface([demo, asr_demo], ["Translate Text", "Speech to Text"]).launch()
