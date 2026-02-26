# tts.py
from fall_detection import run_detection
from google.cloud import texttospeech, speech
import numpy as np
import sounddevice as sd

# ====== Audio I/O: Default Output (Laptop Speakers) ======
TTS_SAMPLE_RATE = 48000 
# ====== STT (Microphone) Configuration ======
STT_SAMPLE_RATE = 16000  
STT_CHANNELS = 1
RESPONSE_TIMEOUT = 10  # seconds

sd.default.device = None  # Uses system default input/output devices
sd.default.samplerate = TTS_SAMPLE_RATE

# ====== Google Cloud Clients ======
# Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set.
tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()

def _mono_pcm16_to_stereo_int16(raw_bytes: bytes) -> np.ndarray:
    """Converts mono PCM16 bytes to stereo int16 numpy array."""
    mono = np.frombuffer(raw_bytes, dtype=np.int16)
    stereo = np.stack([mono, mono], axis=1)
    return stereo

def text_to_speech(text: str) -> None:
    """Synthesizes the given text into speech and plays it via speakers."""
    if not text:
        return
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",  # Keep language as Korean for the user
        name="ko-KR-Standard-A",
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=TTS_SAMPLE_RATE,
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    stereo_i16 = _mono_pcm16_to_stereo_int16(response.audio_content)
    sd.play(stereo_i16, TTS_SAMPLE_RATE, blocking=True)

def _record_for_stt(seconds=RESPONSE_TIMEOUT):
    """Records audio from the microphone for STT processing."""
    # STT recommends 16kHz mono.
    audio = sd.rec(int(seconds * STT_SAMPLE_RATE),
                   samplerate=STT_SAMPLE_RATE,
                   channels=STT_CHANNELS,
                   dtype="int16")
    sd.wait()
    return audio

def _speech_to_text(audio_np: np.ndarray) -> str:
    """Sends recorded audio to Google STT and returns the transcript."""
    audio_bytes = audio_np.tobytes()
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=STT_SAMPLE_RATE,
        language_code="ko-KR",
    )
    resp = stt_client.recognize(config=config, audio=audio)
    if not resp.results:
        return ""
    return resp.results[0].alternatives[0].transcript

# ====== Callback for Fall Detection ======
def handle_fall_event():
    """
    Triggered when a fall is detected.
    - Plays a voice prompt: "Are you okay?" (in Korean)
    - Listens for a response for 10 seconds.
    - Returns "OK" if the user is safe, otherwise returns "ALERT".
    """
    try:
        # Prompting the user in Korean
        text_to_speech("Are you okay?")
    except Exception as e:
        print(f"[TTS Error] {e}")

    try:
        print("Recording user response...")
        audio_np = _record_for_stt(RESPONSE_TIMEOUT)
        transcript = _speech_to_text(audio_np)
        print(f"User Response: {transcript}")

        # Patterns to recognize positive responses in Korean
        ok_patterns = ["Yes", "Sure", "No problem", "I'm fine", "It's okay"]
        if any(pat in transcript for pat in ok_patterns):
            return "OK"
        
        # If no response or unknown response, trigger ALERT
        return "ALERT"
    except Exception as e:
        print(f"[STT Error] {e}")
        return "ALERT"

if __name__ == "__main__":
    # Start detection with the callback
    run_detection(on_fall=handle_fall_event)
