"""
Simple FastRTC Audio Echo App
This app echoes back the user's audio input.
"""

import numpy as np
from fastrtc import Stream, ReplyOnPause, get_stt_model

# Initialize the speech-to-text model
stt_model = get_stt_model(model="moonshine/base")


def echo(audio: tuple[int, np.ndarray]):
    """
    Echo handler that returns the received audio.
    Also transcribes the audio to text using speech-to-text.
    
    Args:
        audio: Tuple of (sample_rate, numpy array of audio)
    
    Yields:
        Audio chunks as tuples of (sample_rate, numpy audio array)
    """
    sample_rate, audio_array = audio

    # Transcribe the audio to text
    text = stt_model.stt(audio)
    print(f"Transcribed text: {text}")

    # Simply yield back the audio
    yield (sample_rate, audio_array)


# Create the stream with ReplyOnPause handler
stream = Stream(
    handler=ReplyOnPause(echo),
    modality="audio",
    mode="send-receive",
    ui_args={"title": "future biz assist"}
)


if __name__ == "__main__":
    # Launch the Gradio interface
    stream.ui.launch()



