import config
import whisper
import numpy as np
import ffmpeg
import io
import pyaudio
import time
import webrtcvad

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
CHUNK_SIZE = 480
# CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
CAPTURE_DURATION = 5  # seconds
# VAD_THRESHOLD = 0.01
VAD_THRESHOLD = 200
VAD_MODE = 3
VAD_SILENCE_LENGTH = 1000  # milliseconds

# Create a VAD object with the desired mode
vad = webrtcvad.Vad(VAD_MODE)


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


def start_audio_stream():
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
    # speaking = False
    silence_start_time = None
    # while time.monotonic() - capture_start_time < CAPTURE_DURATION:
    while True:
        data = stream.read(CHUNK_SIZE)
        # Check if there is speech activity in the current chunk
        is_speech = vad.is_speech(data, SAMPLE_RATE)
        if is_speech:
            # speaking = True
            silence_start_time = None
            # Write the audio data to the current audio buffer
            audio_buffer.write(data)
        # If there is no speech activity and we are not already in a silence period, start a new one
        elif silence_start_time is None:
            silence_start_time = time.monotonic()
        # If there is no speech activity and we are in a silence period, check if the period is long enough
        elif time.monotonic() - silence_start_time > VAD_SILENCE_LENGTH / 1000:
            # If the silence period is long enough, assume the current audio buffer is complete
            if audio_buffer.tell() > 0:
                print("End of speech detected")
                # Stop capturing and clean up
                break

    stream.stop_stream()
    stream.close()
    pa.terminate()
    # Reset the buffer position to the beginning
    audio_buffer.seek(0)
    return audio_buffer


def main():
    while True:
        new_stream = start_audio_stream()
        audio = load_audio_stream(new_stream)
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
