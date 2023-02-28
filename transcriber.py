import config
import whisper
import numpy as np
import ffmpeg
import io
import pyaudio
import time

model = whisper.load_model("base")

def exact_div(x, y):
    assert x % y == 0
    return x // y

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
CAPTURE_DURATION = 5  # seconds


def load_audio_stream(stream, sr: int = SAMPLE_RATE):
    """
    Read an audio stream as mono waveform, resampling as necessary
    Parameters
    ----------
    stream: io.BytesIO
        The audio stream to read
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input("pipe:0", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True, input=stream.read())
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def get_audio_stream():

    # Create an instance of PyAudio
    pa = pyaudio.PyAudio()

    # Open a microphone stream
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    # Create a bytes buffer to hold the audio data
    audio_buffer = io.BytesIO()

    # Capture audio data from the microphone and write it to the buffer
    capture_start_time = time.monotonic()
    while time.monotonic() - capture_start_time < CAPTURE_DURATION:
        data = stream.read(CHUNK_SIZE)
        audio_buffer.write(data)

    # Stop capturing and clean up
    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Reset the buffer position to the beginning
    audio_buffer.seek(0)
    return audio_buffer


def main():
    while True:
        stream = get_audio_stream()
        audio = load_audio_stream(stream)
        print(audio)
        print(audio.shape)
        print(type(audio))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(language= 'en', fp16=False)

        result = whisper.decode(model, mel, options)
        print(result.text)

        if result.no_speech_prob < 0.5:
            print(result.text)

            # append text to transcript file
            with open(config.TRANSCRIPT_FILE, 'a') as f:
                f.write(result.text)


if __name__ == "__main__":
    main()
