"""
Simple FastRTC Audio Live Streaming App
This app receives audio from WebRTC connections and streams it back in real-time (echo).
Uses Google Gemma 3n for multilingual speech-to-text (supports 100+ languages).
"""

from fastapi import FastAPI
import numpy as np
import torch
import tempfile
import os
import wave
from transformers import AutoProcessor, AutoModelForImageTextToText
from fastrtc import ReplyOnPause, Stream, get_current_context
from fastrtc.utils import create_message
from dotenv import load_dotenv


load_dotenv()


class GemmaAudioSTT:
    """
    Custom STT wrapper for Google Gemma 3n model with audio processing.
    Supports multilingual speech recognition (100+ languages) including Hindi/Hinglish/English.
    Reference: https://ai.google.dev/gemma/docs/capabilities/audio
    """
    
    def __init__(self, model_id="google/gemma-3n-E4B-it"):
        """Initialize the Gemma 3n model and processor."""
        print(f"Loading Gemma 3n model: {model_id}")
        
        # Authenticate with Hugging Face if token is provided
        # This is required for gated models like Gemma 3n
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import login
                print("Authenticating with Hugging Face...")
                login(token=hf_token)
                print("Authentication successful!")
            except Exception as e:
                print(f"Warning: Hugging Face authentication failed: {e}")
                print("You may need to set HF_TOKEN environment variable or run: huggingface-cli login")
        else:
            print("Note: HF_TOKEN not set. If you encounter authentication errors,")
            print("set HF_TOKEN environment variable or run: huggingface-cli login")
            print("Get your token from: https://huggingface.co/settings/tokens")
        
        # Determine device and dtype
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Using device: {self.device}, dtype: {self.torch_dtype}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
        
        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto"
        )
        
        print("Gemma 3n model loaded successfully!")
    
    def _prepare_audio(self, sample_rate: int, audio_array: np.ndarray) -> np.ndarray:
        """
        Prepare audio for Gemma processing.
        Gemma expects: mono-channel, 16 kHz, float32 waveforms in range [-1, 1]
        Reference: https://ai.google.dev/gemma/docs/capabilities/audio
        """
        # Check for empty audio
        if audio_array is None or len(audio_array) == 0:
            return None
        
        # need duration of audio_array
        duration = len(audio_array) / sample_rate
        print("duration of audio_array 1", duration);
        
        # Handle multi-channel audio (convert to mono by averaging)
        if len(audio_array.shape) > 1:
            # FastRTC provides audio in shape (channels, samples)
            if audio_array.shape[0] == 1:
                # Already mono (1, samples), just flatten to (samples,)
                audio_array = audio_array.flatten()
            else:
                # Multi-channel, average across channels (axis=0)
                audio_array = np.mean(audio_array, axis=0)
        
        # need duration of audio_array
        duration = len(audio_array) / sample_rate
        print("duration of audio_array 2", duration);

        # Ensure audio is float32 and normalized to [-1, 1]
        if audio_array.dtype != np.float32:
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            else:
                audio_array = audio_array.astype(np.float32)

        # need duration of audio_array
        duration = len(audio_array) / sample_rate
        print("duration of audio_array 3", duration);
        
        # Normalize if needed (ensure range [-1, 1])
        max_val = np.abs(audio_array).max()
        if max_val > 1.0:
            audio_array = audio_array / max_val

        # need duration of audio_array
        duration = len(audio_array) / sample_rate
        print("duration of audio_array 4", duration);
        
        # Resample to 16 kHz if needed (Gemma expects 16kHz)
        # Using scipy.signal.resample as recommended in documentation
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio_array) * 16000 / sample_rate)
            audio_array = signal.resample(audio_array, num_samples)

        # need duration of audio_array
        duration = len(audio_array) / 16000
        print("duration of audio_array 5", duration);
        
        return audio_array
    
    def stt(self, audio: tuple[int, np.ndarray]) -> str:
        """
        Transcribe audio to text using Gemma 3n.
        
        Args:
            audio: Tuple of (sample_rate, numpy array of audio)
        
        Returns:
            Transcribed text string
        """
        sample_rate, audio_array = audio
        
        # Log original audio info
        original_duration = len(audio_array) / sample_rate
        print(f"Original audio: {original_duration:.2f}s at {sample_rate}Hz, {len(audio_array)} samples")
        
        # Prepare audio
        audio_array = self._prepare_audio(sample_rate, audio_array)
        if audio_array is None:
            return ""
        
        # Check minimum audio duration (Gemma needs at least ~1 second for conv layers)
        # After resampling to 16kHz, we have 16000 samples per second
        min_samples = 16000  # 1 second minimum
        actual_duration = len(audio_array) / 16000
        
        if len(audio_array) < min_samples:
            print(f"Audio too short ({actual_duration:.2f}s), padding to minimum duration")
            # Pad with silence to meet minimum duration
            padding = np.zeros(min_samples - len(audio_array), dtype=np.float32)
            audio_array = np.concatenate([audio_array, padding])
        else:
            print(f"Audio duration: {actual_duration:.2f}s")
        
        try:
            # Stream audio directly to Gemma processor (no temp file needed)
            # Pass audio as numpy array (processor expects raw numpy array at 16kHz)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": audio_array  # Pass numpy array directly
                        },
                        {"type": "text", "text": "Transcribe this audio."},
                    ]
                }
            ]
            
            # Apply chat template and tokenize
            input_ids = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
            )
            # Move tensors to device, but preserve dtypes for input_ids and masks
            # input_ids must be Long (int64) for embeddings
            # masks must be bool/int for bitwise operations
            # Only actual feature tensors should use model's float16 dtype
            processed_inputs = {}
            for k, v in input_ids.items():
                if isinstance(v, torch.Tensor):
                    # Keep input_ids and all masks in their original dtype
                    if k == "input_ids" or "mask" in k.lower():
                        # Preserve original dtype (int64 for input_ids, bool/int for masks)
                        processed_inputs[k] = v.to(self.model.device)
                    else:
                        # Feature tensors can use model dtype (float16)
                        processed_inputs[k] = v.to(self.model.device, dtype=self.model.dtype)
                else:
                    processed_inputs[k] = v
            input_ids = processed_inputs
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    **input_ids,
                    max_new_tokens=256,  # Adjust based on expected transcription length
                    do_sample=False,
                )
            
            # Decode the output
            full_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            # Extract only the model's response (after "model" marker)
            # The format is: "user\n...\nTranscribe this audio.\nmodel\n<actual transcription>"
            if "model" in full_text:
                # Split by "model" and take everything after it
                parts = full_text.split("model", 1)
                if len(parts) > 1:
                    text = parts[1].strip()
                else:
                    text = full_text.strip()
            else:
                text = full_text.strip()
            
            print(f"Extracted transcription: {text}")
            return text
                
        except Exception as e:
            print(f"Error during transcription: {e}")
            import traceback
            traceback.print_exc()
            return ""


# Initialize the custom STT model
stt_model = GemmaAudioSTT()


def echo(audio: tuple[int, np.ndarray]):
    """
    Echo handler that returns the received audio.
    Also transcribes the audio to text using speech-to-text.
    
    Args:
        audio: Tuple of (sample_rate, numpy array of audio)
    
    Yields:
        Audio chunks as tuples of (sample_rate, numpy audio array) """

    # Debug: Check audio format
    sample_rate, audio_array = audio
    print(f"DEBUG echo: type={type(audio_array)}, shape={audio_array.shape if hasattr(audio_array, 'shape') else 'N/A'}, dtype={audio_array.dtype if hasattr(audio_array, 'dtype') else 'N/A'}")
    
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
# ReplyOnPause accumulates audio and triggers on silence detection
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
