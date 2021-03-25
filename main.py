import configparser
import json
from pathlib import Path

from transformers import AutoTokenizer, Wav2Vec2ForCTC
import sounddevice as sd
import soundfile as sf
import torch

def record_from_mic(config):
    """Record audio from a microphone.

    Args:
        config (ConfigParser): Config params.
    Returns:
        audio (ndarray): Recorded audio.

    """
    sample_rate = config.getint('config', 'sample_rate')
    duration_secs = config.getint('microphone', 'duration_secs')
    channels = config.getint('microphone', 'channels')
    print("Start recording . . . ")
    audio = sd.rec(int(duration_secs*sample_rate), sample_rate, channels)
    sd.wait()  # Wait until recording is finished
    print("Finish recording")

    return audio

def wav2vec2_inference(audio, tokenizer, model):
    """Transcript audio with the Wav2Vec2 model.

    Args:
        audio (ndarray): Audio of interest.
        tokenizer (Wav2Vec2Tokenizer): Wav2Vec2 associated tokenizer.
        model (Wav2Vec2ForCTC): Wav2Vec2 to perform the transcription.
    Returns:
        transcriptions (str): Audio transcript.

    """
    input_values = tokenizer(audio.ravel(), return_tensors='pt').input_values
    logits = model(input_values).logits
    # Store predicted id's
    predicted_ids = torch.argmax(logits, dim =-1)
    # Decode the audio to generate text
    transcriptions = tokenizer.decode(predicted_ids[0])

    return transcriptions

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Initialize tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

    if config.getboolean('config', 'from_microphone'):
        # Record from microphone and transcript
        audio = record_from_mic(config)
        transcriptions = wav2vec2_inference(audio, tokenizer, model)
        print(f"Transcribed audio: {transcriptions}")
        if config.getboolean('config', 'save_transcriptions'):
            with open('mic_transcription.txt', 'w') as file:
                file.write(transcriptions)
            print(f"Transcribed audio stored in mic_transcription.txt")
    else:
        # Transcript files in configuration file
        audio_files = json.loads(config.get('config', 'audio_files'))
        for audio_file in audio_files:
            audio, _ = sf.read(audio_file, dtype='float32')
            transcriptions = wav2vec2_inference(audio, tokenizer, model)
            print(f"Transcribed audio: {transcriptions}")
            if config.getboolean('config', 'save_transcriptions'):
                with open(f'{Path(audio_file).stem}.txt', 'w') as file:
                    file.write(transcriptions)
                print(f"Transcribed audio stored in {Path(audio_file).stem}.txt")


if __name__ == '__main__':
    main()