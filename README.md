# FastRTC Audio Streaming with Transcription & TTS

A real-time audio streaming application built with FastRTC that provides:
- **Speech-to-Text (STT)**: Transcribes incoming audio in real-time
- **Text-to-Speech (TTS)**: Converts transcribed text back to audio
- **API Streaming Support**: Connect from external clients (Android/KMM apps) via WebRTC
- **Bidirectional Communication**: Send and receive audio with transcription feedback

## Features

- 🎤 **Real-time Audio Streaming**: Low-latency audio streaming using WebRTC
- 📝 **Automatic Transcription**: Speech-to-text using Moonshine STT model
- 🔊 **Voice Response**: Text-to-speech using Kokoro TTS model
- 📡 **API Support**: Connect from external applications via WebRTC API
- 🌐 **Network Access**: Accepts connections from network (not just localhost)
- ⏸️ **Pause Detection**: Uses `ReplyOnPause` handler to process complete utterances

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd fastrtc
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `fastrtc[stt]` - FastRTC with Speech-to-Text support
   - All required dependencies (numpy, gradio, etc.)

## Usage

### Running the Server

1. **Activate your virtual environment** (if not already activated):
   ```bash
   source venv/bin/activate
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Access the web interface:**
   - Open your browser and navigate to: `http://localhost:7860`
   - The Gradio interface will be available for testing

### Server Configuration

The server is configured to:
- Listen on `0.0.0.0:7860` (accepts connections from network)
- Use `ReplyOnPause` handler (processes audio when user pauses speaking)
- Support bidirectional audio (`send-receive` mode)

To modify settings, edit `app.py`:

```python
stream.ui.launch(
    server_name="0.0.0.0",  # Change to "127.0.0.1" for localhost only
    server_port=7860,       # Change port if needed
    share=False             # Set to True for public Gradio URL
)
```

## API Streaming for External Clients

This application supports connecting from external clients (e.g., Android/KMM apps) via the FastRTC WebRTC API.

### Connection Endpoint

- **URL**: `http://YOUR_SERVER_IP:7860/webrtc/offer`
- **Method**: POST
- **Content-Type**: application/json

### Request Format

```json
{
    "sdp": "<webrtc_offer_sdp>",
    "type": "offer"
}
```

### Response Format

```json
{
    "sdp": "<webrtc_answer_sdp>",
    "type": "answer",
    "webrtc_id": "<unique_connection_id>"
}
```

### Message Types

The server sends messages via Data Channel with the following format:

```json
{
    "type": "fetch_output" | "log" | "error" | "warning",
    "data": "<message_content>"
}
```

#### Transcription Messages

When audio is transcribed, clients receive:

```json
{
    "type": "fetch_output",
    "data": "<transcribed_text>"
}
```

#### Log Messages

The server sends log messages for debugging:

```json
{
    "type": "log",
    "data": "pause_detected" | "response_starting" | "started_talking"
}
```

### Connecting from Android/KMM App

1. **Establish WebRTC Connection:**
   - Create a `PeerConnection` with ICE servers
   - Create an audio track from microphone
   - Create a data channel for text messages

2. **Send WebRTC Offer:**
   - Create an offer
   - POST to `/webrtc/offer` endpoint
   - Receive answer and set remote description

3. **Handle Messages:**
   - Listen for `fetch_output` messages on data channel
   - Display transcription text
   - Play received audio (TTS response)

4. **Receive Audio:**
   - Audio track receives TTS audio response
   - Play through device speakers/headphones

For detailed Android/KMM implementation, see the [FastRTC API Documentation](https://fastrtc.org/userguide/api/).

## Architecture

### Components

1. **STT Model** (`moonshine/base`):
   - Converts speech audio to text
   - Processes complete utterances (on pause)

2. **TTS Model** (`kokoro`):
   - Converts transcribed text to speech audio
   - Uses voice: `af_heart`
   - Language: `en-us`

3. **ReplyOnPause Handler**:
   - Buffers audio chunks
   - Detects when user stops speaking
   - Processes complete utterances

4. **Stream Handler**:
   - Receives audio from client
   - Transcribes using STT
   - Sends transcription via `AdditionalOutputs`
   - Generates TTS audio
   - Returns audio to client

### Flow Diagram

```
Client (Android/Web) 
    ↓ [Audio Stream]
WebRTC Connection
    ↓
ReplyOnPause Handler (buffers audio)
    ↓ [On Pause]
Echo Handler
    ↓
STT Model → Transcription
    ↓
AdditionalOutputs → Client (via Data Channel)
    ↓
TTS Model → Audio Response
    ↓ [Audio Stream]
WebRTC Connection
    ↓
Client (plays audio)
```

## Configuration

### TTS Options

Modify TTS settings in `app.py`:

```python
tts_options = KokoroTTSOptions(
    voice="af_heart",    # Change voice
    speed=1.0,          # Adjust speed (0.5 - 2.0)
    lang="en-us"        # Change language
)
```

### STT Model

Change STT model in `app.py`:

```python
stt_model = get_stt_model(model="moonshine/base")  # Change model
```

## Error Handling

The application includes error handling for:
- Empty transcriptions (yields silence)
- TTS generation errors (yields silence fallback)
- Connection errors (handled by FastRTC)

## Troubleshooting

### Server Not Accessible from Network

- Ensure `server_name="0.0.0.0"` in `app.py`
- Check firewall settings
- Verify server IP address

### No Transcription Received

- Check that audio is being sent from client
- Verify STT model is loaded correctly
- Check console logs for errors

### TTS Errors

- Ensure text is not empty before calling TTS
- Check TTS model is loaded correctly
- Verify TTS options are valid

## Development

### Project Structure

```
fastrtc/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── venv/              # Virtual environment (gitignored)
```

### Dependencies

- `fastrtc[stt]` - FastRTC with STT support
- `numpy` - Audio processing
- `gradio` - Web interface

## Resources

- [FastRTC Documentation](https://fastrtc.org/)
- [FastRTC API Guide](https://fastrtc.org/userguide/api/)
- [FastRTC Audio Streaming](https://fastrtc.org/userguide/audio/)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
