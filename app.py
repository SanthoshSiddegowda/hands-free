"""
Simple FastRTC Audio Echo App with API Streaming Support
This app transcribes audio, sends transcription via API, and responds with TTS audio.
"""

import numpy as np
from fastrtc import Stream, ReplyOnPause, get_stt_model, KokoroTTSOptions, get_tts_model, AdditionalOutputs

# Initialize the speech-to-text model
stt_model = get_stt_model(model="moonshine/base")

tts_model = get_tts_model(model="kokoro")


def echo(audio: tuple[int, np.ndarray], webrtc_id: str = None):
    """
    Echo handler that transcribes audio, sends transcription via API, and responds with TTS.
    
    Args:
        audio: Tuple of (sample_rate, numpy array of audio)
        webrtc_id: The WebRTC connection ID (provided by fastrtc for API streaming)
    
    Yields:
        AdditionalOutputs for transcription, then audio chunks as tuples of (sample_rate, numpy audio array)
    """
    sample_rate, audio_array = audio

    # Transcribe the audio to text
    text = stt_model.stt(audio)
    print(f"Transcribed text: {text}")

    # Send transcription back to client via AdditionalOutputs (API streaming)
    # Yield AdditionalOutputs first - this will be received as "fetch_output" message type in the client
    # The client will receive: {"type": "fetch_output", "data": "<text>"}
    # AdditionalOutputs takes positional arguments, not keyword arguments
    yield AdditionalOutputs(text if text else "")

    # Skip TTS if text is empty or only whitespace
    if not text or not text.strip():
        # Yield silence if no text
        silence_duration = 0.5  # seconds
        silence_samples = int(sample_rate * silence_duration)
        silence = np.zeros(silence_samples, dtype=np.float32)
        yield (sample_rate, silence)
        return

    # Generate TTS audio from transcribed text
    tts_options = KokoroTTSOptions(
        voice="af_heart",
        speed=1.0,
        lang="en-us"
    )
    
    try:
        tts_audio = tts_model.tts(text, options=tts_options)
        tts_sample_rate, tts_audio_array = tts_audio
        # Yield the TTS audio response
        yield (tts_sample_rate, tts_audio_array)
    except Exception as e:
        print(f"Error in TTS: {e}")
        # Fallback: yield silence on TTS error
        silence_duration = 0.5  # seconds
        silence_samples = int(sample_rate * silence_duration)
        silence = np.zeros(silence_samples, dtype=np.float32)
        yield (sample_rate, silence)


# Create the stream with ReplyOnPause handler
stream = Stream(
    handler=ReplyOnPause(echo),
    modality="audio",
    mode="send-receive",
    ui_args={"title": "future biz assist"}
)


if __name__ == "__main__":
    # Launch the Gradio interface with network access for API clients
    # This enables API streaming from external clients (e.g., Android app)
    stream.ui.launch(
        server_name="0.0.0.0",  # Accept connections from network (required for API)
        server_port=7860,       # Default Gradio port
        share=True             # Set to True if you want a public URL
    )



