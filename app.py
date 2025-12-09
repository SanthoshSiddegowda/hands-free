"""
Simple FastRTC Audio Echo App with API Streaming Support
This app transcribes audio, sends transcription via API, and responds with TTS audio.
"""

import os
import numpy as np
from fastrtc import Stream, ReplyOnPause, get_stt_model, KokoroTTSOptions, get_tts_model, AdditionalOutputs
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Google Generative AI client (only if API key is available)
gemini_api_key = os.getenv('GEMINI_API_KEY')
if gemini_api_key:
    client = genai.Client(api_key=gemini_api_key)
else:
    client = None
    print("Warning: GEMINI_API_KEY not set. Gemini integration will be disabled.")

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

    # Generate response using Google Gemini (if available)
    if client:
        try:
            # System prompt for Bizom voice assistant
            system_prompt = """
            You are Bizom voice assistant.
            Bizom salespeople use you to get quick answers.
            Always respond in short and sweet answers. Be concise and helpful.
            """
            
            # Combine system prompt with user query
            prompt = f"{system_prompt}\n\nUser: {text}\nAssistant:"
            
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    top_p=0.95,
                    top_k=20,
                ),
            )

            response_text = response.text
            print(f"Gemini response: {response_text}")
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Fallback to using transcribed text directly
            response_text = text
    else:
        # If Gemini is not available, use transcribed text directly
        response_text = text
        print(f"Using transcribed text directly (Gemini not available): {response_text}")

    # Generate TTS audio from response text
    try:
        tts_options = KokoroTTSOptions(
            voice="af_heart",
            speed=1.0,
            lang="en-us"
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
        share=False             # Set to True if you want a public URL
    )



