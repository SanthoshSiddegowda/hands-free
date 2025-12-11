"""
Simple FastRTC Audio Echo App with API Streaming Support
This app uses Shuka-1 for audio-to-text processing in Indian languages and responds with TTS audio.
"""

import os
import numpy as np
from fastrtc import Stream, ReplyOnPause, KokoroTTSOptions, get_tts_model, AdditionalOutputs
from dotenv import load_dotenv
import transformers
import torch

# Load environment variables from .env file
load_dotenv()

# Initialize Shuka-1 model (Audio-to-Text model for Indian languages)
# Shuka-1 supports: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, 
# Marathi, Oriya, Punjabi, Tamil, and Telugu
# Works offline after first model download
print("Loading Shuka-1 model... This may take a few minutes on first run.")
try:
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    shuka_pipe = transformers.pipeline(
        model='sarvamai/shuka-1',
        trust_remote_code=True,
        device=device,
        torch_dtype=dtype
    )
    print(f"Shuka-1 model loaded successfully on {'GPU' if device >= 0 else 'CPU'}")
except Exception as e:
    print(f"Error loading Shuka-1 model: {e}")
    print("Falling back to basic functionality")
    shuka_pipe = None

# Initialize the text-to-speech model
tts_model = get_tts_model(model="kokoro")


def echo(audio: tuple[int, np.ndarray], webrtc_id: str = None):
    """
    Echo handler that processes audio with Shuka-1, sends response via API, and responds with TTS.
    
    Args:
        audio: Tuple of (sample_rate, numpy array of audio)
        webrtc_id: The WebRTC connection ID (provided by fastrtc for API streaming)
    
    Yields:
        AdditionalOutputs for response text, then audio chunks as tuples of (sample_rate, numpy audio array)
    """
    sample_rate, audio_array = audio

    # Process audio with Shuka-1 (audio-to-text model for Indian languages)
    if shuka_pipe:
        try:
            # Ensure audio is in the correct format (float32, mono)
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize audio if needed
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Shuka-1 system prompt for Bizom voice assistant
            turns = [
                {
                    'role': 'system', 
                    'content': 'You are Bizom voice assistant. Bizom Salesman use you to get quick answers. Always respond in short and sweet answers. Be concise and helpful. IMPORTANT: Respond in the SAME LANGUAGE as the user\'s query. Support all Indian languages: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Malayalam, Kannada, Urdu, Punjabi, and English.'
                },
                {
                    'role': 'user', 
                    'content': '<|audio|>'
                }
            ]
            
            # Process audio with Shuka-1
            result = shuka_pipe(
                {
                    'audio': audio_array,
                    'turns': turns,
                    'sampling_rate': sample_rate
                },
                max_new_tokens=512
            )
            
            # Extract response text from result
            if isinstance(result, dict):
                response_text = result.get('generated_text', '') or result.get('text', '')
            elif isinstance(result, str):
                response_text = result
            elif isinstance(result, list) and len(result) > 0:
                response_text = result[0].get('generated_text', '') if isinstance(result[0], dict) else str(result[0])
            else:
                response_text = str(result)
            
            # Clean up response text
            response_text = response_text.strip()
            print(f"Shuka-1 response: {response_text}")
            
            # Send response back to client via AdditionalOutputs (API streaming)
            # The client will receive: {"type": "fetch_output", "data": "<response_text>"}
            yield AdditionalOutputs(response_text if response_text else "")
            
            # Skip TTS if response is empty
            if not response_text:
                silence_duration = 0.5  # seconds
                silence_samples = int(sample_rate * silence_duration)
                silence = np.zeros(silence_samples, dtype=np.float32)
                yield (sample_rate, silence)
                return
            
            # Generate TTS audio from response text
            try:
                # Detect language from response (simple heuristic)
                # For Indian languages, try Hindi TTS, fallback to English
                has_indian_script = any([
                    '\u0900' <= char <= '\u097F' for char in response_text  # Devanagari
                ]) or any([
                    '\u0B80' <= char <= '\u0BFF' for char in response_text  # Tamil
                ]) or any([
                    '\u0C00' <= char <= '\u0C7F' for char in response_text  # Telugu/Kannada
                ])
                
                tts_lang = 'hi' if has_indian_script else 'en-us'
                
                tts_options = KokoroTTSOptions(
                    voice="af_bella",
                    speed=1.0,
                    lang=tts_lang
                )
                
                tts_audio = tts_model.tts(response_text, options=tts_options)
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
                
        except Exception as e:
            print(f"Error processing audio with Shuka-1: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: yield silence
            yield AdditionalOutputs("")
            silence_duration = 0.5  # seconds
            silence_samples = int(sample_rate * silence_duration)
            silence = np.zeros(silence_samples, dtype=np.float32)
            yield (sample_rate, silence)
    else:
        # Fallback if Shuka-1 is not loaded
        print("Shuka-1 model not available")
        yield AdditionalOutputs("")
        silence_duration = 0.5  # seconds
        silence_samples = int(sample_rate * silence_duration)
        silence = np.zeros(silence_samples, dtype=np.float32)
        yield (sample_rate, silence)


# Create the stream with ReplyOnPause handler
stream = Stream(
    handler=ReplyOnPause(echo),
    modality="audio",
    mode="send-receive",
    ui_args={"title": "Bizom Assist"}
)


if __name__ == "__main__":
    # Launch the Gradio interface with network access for API clients
    # This enables API streaming from external clients (e.g., Android app)
    stream.ui.launch(
        server_name="0.0.0.0",  # Accept connections from network (required for API)
        server_port=7860,       # Default Gradio port
        share=False             # Set to True if you want a public URL
    )



