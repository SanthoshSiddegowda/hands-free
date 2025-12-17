"""
Simple FastRTC Audio Live Streaming App
This app receives audio from WebRTC connections and streams it back in real-time (echo).
"""

from fastapi import FastAPI
import numpy as np
from fastrtc import ReplyOnPause, Stream, get_current_context, get_stt_model
from fastrtc.utils import create_message


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

    # Transcribe the audio to text
    text = stt_model.stt(audio)
    print(f"Transcribed text: {text}")

    # Push transcription over the WebRTC data channel.
    if text:
        try:
            ctx = get_current_context()
            handler = stream.handlers.get(ctx.webrtc_id)  # type: ignore[attr-defined]
            if handler is not None and hasattr(handler, "send_message_sync"):
                handler.send_message_sync(create_message("fetch_output", text))
        except Exception as e:
            # Never crash the audio pipeline if transcription messaging fails.
            print(f"Failed to send transcription over data channel: {e}")

    # Echo audio back over the media track.
    yield audio

    

# Create the stream with live streaming handler
stream = Stream(
    handler=ReplyOnPause(echo),
    modality="audio",
    mode="send-receive",
)

app = FastAPI()
stream.mount(app)


@app.get("/")
async def index():
    """Return server status."""
    return {"status": "running", "endpoint": "/webrtc/offer"}


if __name__ == "__main__":
    from pyngrok import ngrok
     
    port = 7860
    public_url = ngrok.connect(port)
    print(f"Public URL: {public_url}")
    
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        ngrok.kill()
