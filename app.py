import spaces
import torch
import gradio as gr
import time
import numpy as np
from transformers import pipeline
from omegaconf import OmegaConf
import soundfile as sf
import tempfile
from pydub import AudioSegment

from google import genai
import logging
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
default_model_id = "whisper-large-v3"

logger.warning(device)


def load_pipe(model_id: str):
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=8,
        dtype=torch_dtype,
        device=device,
    )


OmegaConf.register_new_resolver("load_pipe", load_pipe)
models_config = OmegaConf.to_object(OmegaConf.load("models.yaml"))
model = models_config[default_model_id]["model"]

orig_whisper_pipe = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-base",
    chunk_length_s=30,
    device=device,
)


@spaces.GPU
def stream_transcribe(stream, new_chunk, dialect_id, api_key, volume_threshold, is_zh):
    start_time = time.time()
    try:
        sr, y = new_chunk

        volume = np.sqrt(np.mean(y**2))
        should_asr = volume >= int(volume_threshold)

        # Convert to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)

        y = y.astype(np.float32)
        y /= np.max(np.abs(y))

        if stream["audio_buffer"] is not None:
            stream["audio_buffer"] = np.concatenate([stream["audio_buffer"], y])
        else:
            stream["audio_buffer"] = y

        generate_kwargs = {
            "task": "transcribe",
            "language": "Chinese",
            "num_beams": 1,
            "prompt_ids": torch.from_numpy(model.tokenizer.get_prompt_ids(dialect_id)).to(
                    device
            )
        }

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, stream["audio_buffer"], sr)
        audio = AudioSegment.from_file(tmp.name)
        audio = audio.set_frame_rate(16000)
        temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(temp_wav_file.name, format="wav")
        audio_file = temp_wav_file.name

        # transcription = pipe({"sampling_rate": sr, "raw": stream})["text"]
        if is_zh:
            res = orig_whisper_pipe(audio_file, batch_size=8, generate_kwargs={"task": "transcribe", "language": "chinese"}, return_timestamps=False)
        else:
            res = model(audio_file, generate_kwargs=generate_kwargs, return_timestamps=False)
        # logger.warning(res)
        transcription = res["text"]
        end_time = time.time()
        latency = end_time - start_time

        text = ""
        if len(stream["text_buffer"]) > 0:
            text = "\n".join(stream["text_buffer"][-6:])

        if transcription:
            if len(transcription) >= 25 or stream["last_text"] == transcription:
                stream["audio_buffer"] = None
                stream["text_buffer"].append("<p style='font-size:xxx-large;'>"+transcription+"</p>")
                try:
                    if (api_key):
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model="gemini-2.5-flash", contents=f"只回答答案，不要復述題目，將客語「{transcription}」 翻譯成繁體中文"
                        )
                        stream["text_buffer"].append("<p style='color:blue; font-size:xxx-large; font-weight:bold;'>"+response.text+"</p>")
                    else:
                        if is_zh:
                            ha_text = requests.post(
                                "https://api.gohakka.org/v2/render/txt/zh-to-sixian",
                                headers={
                                    "Content-Type": "application/json",
                                    "Authorization": "Bearer go_4h1254wRzXZ3HHJGGtOChmXHvFFvgu"
                                },
                                json={
                                    "text": transcription
                                }
                            ).json().get("hakka")
                            stream["text_buffer"].append("<p style='font-size:xxx-large; font-weight:bold;'>"+ha_text+"</p>")
                        else:
                            zh_text = requests.post(
                                "https://api.gohakka.org/v2/render/txt/sixian-to-zh",
                                headers={
                                    "Content-Type": "application/json",
                                    "Authorization": "Bearer go_4h1254wRzXZ3HHJGGtOChmXHvFFvgu"
                                },
                                json={
                                    "text": transcription
                                }
                            ).json().get("chinese")
                            stream["text_buffer"].append("<p style='font-size:xxx-large; font-weight:bold;'>"+zh_text+"</p>")
                        # stream["text_buffer"].append("API Key 未提供")
                except Exception as e:
                    stream["text_buffer"].append(f"{e}")
                text = "\n".join(stream["text_buffer"][-6:])
            else:
                if should_asr:
                    stream["last_text"] = transcription
                    text += "<p style='font-size:xxx-large;'>"+transcription+"</p>"
                else:
                    text += "<p style='font-size:xxx-large;'>&emsp;</p>"

        return stream, text, f"{latency:.2f}", f"{volume:.2f}"
    except Exception as e:
        print(f"Error during Transcription: {e}")
        return stream, e, "-", "-"


def clear():
    return ""


def clear_state():
    return {"audio_buffer": None, "text_buffer": [], "last_text": ""}


with gr.Blocks() as microphone:
    with gr.Column():
        gr.Markdown(
            f"# Realtime Hakka ASR: \nNote: add 'http://z.ouob.net:2515' to 'chrome://flags/#unsafely-treat-insecure-origin-as-secure' access the microphone.")
        with gr.Row():
            input_audio_microphone = gr.Audio(streaming=True)
            with gr.Column():
                dialect_drop_down = gr.Dropdown(
                    choices=[
                        (k, v)
                        for k, v in models_config[default_model_id]["dialect_mapping"].items()
                    ],
                    value=list(models_config[default_model_id]["dialect_mapping"].values())[0],
                    label="腔調",
                )
                api_key = gr.Textbox(label="Gemini API Key", type="password")
            with gr.Column(scale=0):
                is_zh = gr.Checkbox(label="啟用華文輸入", value=False)
                volume_threshold = gr.Textbox(label="Volume Threshold", value="10")
            with gr.Column(scale=0):
                latency_textbox = gr.Textbox(label="Latency (seconds)", value="0.0", scale=0)
                volume = gr.Textbox(label="Volume", value="0.0", scale=0)
        with gr.Row():
            # output = gr.Textbox(label="Transcription", value="")
            output = gr.HTML()
        #     with gr.Column(scale=0):
        #         latency_textbox = gr.Textbox(label="Latency (seconds)", value="0.0", scale=0)
        #         volume = gr.Textbox(label="Volume", value="0.0", scale=0)
        # with gr.Row():
            # clear_button = gr.Button("Clear Output")
        state = gr.State(clear_state())
        input_audio_microphone.stream(stream_transcribe, [state, input_audio_microphone, dialect_drop_down, api_key, volume_threshold, is_zh], [
                                      state, output, latency_textbox, volume], time_limit=30, stream_every=1, concurrency_limit=None)
        # clear_button.click(clear_state, outputs=[state]).then(clear, outputs=[output])


with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.TabbedInterface([microphone], ["Microphone"])

demo.launch()
