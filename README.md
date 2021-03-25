# Text to Speech Wav2Vec2

This program uses the Facebook's [Wav2Vec2 model](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) ([HuggingFaace implementation](https://huggingface.co/transformers/model_doc/wav2vec2.html)) to transcript audio from local files or from an on-air recording through an available microphone.


## Dependencies

Running the application can be done following the instructions above:

1. To create a Python Virtual Environment (virtualenv) to run the code, type:

    ```python -m venv my-env```

2. Activate the new environment:
    * Windows: ```my-env\Scripts\activate.bat```
    * macOS and Linux: ```source my-env/bin/activate``` 

3. Install all the dependencies from *requirements.txt*:

    ```pip install -r requirements.txt```

## Use

The program can be executed with:

```python main.py```

There's available a *config.ini* file that allows the user to change the application's behaviour. The options are:

### [config]
* **from_microphone:** Boolean to indicate if the transcription is going to be made from an audio file (False) or from a live recording (True). Default True.
* **save_transcriptions:** Boolean to indicate if the transcriptions should be saved as *.txt* files or not. Default False.
* **sample_rate:** Integer that indicates the sample rate of the recording. Default 16000.
* **audio_files:** List of audio files to transcript.

### [microphone]
* **from_microphone:** Integer indicating the number of seconds that the microphone'll be open for the recording. Default 5.
* **channels:** Integer indicating the number of channels of the recording. Default 1.

## Acknowledgments

* [Wav2vec 2.0: Learning the structure of speech from raw audio](https://arxiv.org/abs/2006.11477)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details