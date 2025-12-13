"""
Simple FastRTC Audio Live Streaming App
This app receives audio from WebRTC connections and streams it back in real-time (echo).
"""

from fastapi import FastAPI
import numpy as np
from fastrtc import Stream, StreamHandler, get_cloudflare_turn_credentials_async, get_stt_model
from gradio.utils import get_space
from collections import deque
import time


class AudioLiveStreamHandler(StreamHandler):
    """
    Handler that receives audio and streams it back in real-time (echo).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_rate = 48000  # Default, will be updated from first frame
        self.frame_count = 0
        self.last_log_time = time.time()
        # Queue for audio frames to be streamed back
        self.output_queue = deque(maxlen=10)  # Keep last 10 frames to prevent overflow
        # Initialize speech-to-text model
        self.stt_model = get_stt_model(model="moonshine/base")
        # Buffer to accumulate audio for transcription
        self.audio_buffer = []
        self.buffer_duration = 2.0  # Transcribe every 2 seconds of audio
        self.buffer_samples = 0
        
    def receive(self, frame: tuple[int, np.ndarray]):
        """
        Process incoming audio frames and queue them for streaming back.
        Also transcribes the audio using speech-to-text.
        
        Args:
            frame: Tuple of (sample_rate, numpy array of audio)
        """
        sample_rate, audio_array = frame
        self.sample_rate = sample_rate
        self.frame_count += 1
        
        # Log first few frames for debugging
        if self.frame_count <= 3:
            print(f"Received audio frame #{self.frame_count}: sample_rate={sample_rate}, samples={len(audio_array)}, dtype={audio_array.dtype}")
        
        # Accumulate audio for transcription
        audio_copy = audio_array.copy() if hasattr(audio_array, 'copy') else audio_array
        self.audio_buffer.append(audio_copy)
        self.buffer_samples += audio_array.shape[-1] if len(audio_array.shape) > 1 else len(audio_array)
        
        # Transcribe when we have enough audio (every ~2 seconds)
        samples_per_second = sample_rate
        if self.buffer_samples >= samples_per_second * self.buffer_duration:
            # Concatenate all buffered audio
            if len(self.audio_buffer) > 0:
                # Handle different array shapes
                if len(self.audio_buffer[0].shape) == 1:
                    accumulated_audio = np.concatenate(self.audio_buffer)
                else:
                    accumulated_audio = np.concatenate(self.audio_buffer, axis=-1)
                
                # Ensure correct shape: (1, num_samples)
                if len(accumulated_audio.shape) == 1:
                    accumulated_audio = accumulated_audio.reshape(1, -1)
                
                # Transcribe the accumulated audio
                try:
                    transcription = self.stt_model.stt((sample_rate, accumulated_audio))
                    if transcription and transcription.strip():
                        print(f"[Transcription]: {transcription}")
                except Exception as e:
                    print(f"Transcription error: {e}")
                
                # Clear buffer
                self.audio_buffer = []
                self.buffer_samples = 0
        
        # Log periodically (every 5 seconds)
        current_time = time.time()
        if current_time - self.last_log_time >= 5.0:
            print(f"Live streaming: {self.frame_count} frames received, {len(self.output_queue)} in queue")
            self.last_log_time = current_time
        
        # Add frame to output queue for immediate echo back
        self.output_queue.append((sample_rate, audio_copy))
    
    def emit(self):
        """
        Emit audio frames for live streaming (echo back).
        
        Returns:
            Audio frame tuple (sample_rate, numpy array) or None
        """
        # Echo back frames from the queue
        if self.output_queue:
            return self.output_queue.popleft()
        return None
    
    def shutdown(self):
        """Transcribe any remaining buffered audio when stream ends."""
        if self.audio_buffer and self.buffer_samples > 0:
            try:
                # Concatenate remaining audio
                if len(self.audio_buffer[0].shape) == 1:
                    accumulated_audio = np.concatenate(self.audio_buffer)
                else:
                    accumulated_audio = np.concatenate(self.audio_buffer, axis=-1)
                
                # Ensure correct shape: (1, num_samples)
                if len(accumulated_audio.shape) == 1:
                    accumulated_audio = accumulated_audio.reshape(1, -1)
                
                # Transcribe remaining audio
                transcription = self.stt_model.stt((self.sample_rate, accumulated_audio))
                if transcription and transcription.strip():
                    print(f"[Final Transcription]: {transcription}")
            except Exception as e:
                print(f"Final transcription error: {e}")
    
    def copy(self):
        """Create a copy of this handler for a new connection."""
        # Create a new instance with default parameters
        # Each connection gets its own handler instance
        return AudioLiveStreamHandler()


# Create the stream with live streaming handler
stream = Stream(
    handler=AudioLiveStreamHandler(),
    modality="audio",
    mode="send-receive",
    rtc_configuration=get_cloudflare_turn_credentials_async if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
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
