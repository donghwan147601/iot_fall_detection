# tts.py
from fall_detection import run_detection
from google.cloud import texttospeech, speech
import numpy as np
import sounddevice as sd

# ====== Audio I/O: 기본 출력(노트북 스피커) ======
TTS_SAMPLE_RATE = 48000
# ====== STT(마이크) 설정 ======
STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1
RESPONSE_TIMEOUT = 10  # 초

sd.default.device = None  # 시스템 기본 입/출력 장치 사용
sd.default.samplerate = TTS_SAMPLE_RATE

# ====== Google 클라이언트 ======
tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()

def _mono_pcm16_to_stereo_int16(raw_bytes: bytes) -> np.ndarray:
    mono = np.frombuffer(raw_bytes, dtype=np.int16)
    stereo = np.stack([mono, mono], axis=1)
    return stereo

def text_to_speech(text: str) -> None:
    if not text:
        return
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
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
    # STT는 16kHz mono를 권장하므로, 녹음 시 샘플레이트/채널을 맞춰줍니다.
    audio = sd.rec(int(seconds * STT_SAMPLE_RATE),
                   samplerate=STT_SAMPLE_RATE,
                   channels=STT_CHANNELS,
                   dtype="int16")
    sd.wait()
    return audio

def _speech_to_text(audio_np: np.ndarray) -> str:
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

# ====== 낙상 시 호출될 콜백 ======
def say_are_you_ok():
    """
    낙상 '처음' 감지되었을 때 호출됩니다.
    - "괜찮으세요?"를 재생하고
    - 10초 동안 마이크로 응답을 듣고
    - "OK" 또는 "ALERT"를 반환합니다.
    """
    try:
        text_to_speech("괜찮으세요?")
    except Exception as e:
        print(f"[TTS] 오류: {e}")

    try:
        audio_np = _record_for_stt(RESPONSE_TIMEOUT)
        transcript = _speech_to_text(audio_np)
        print("사용자 응답:", transcript)

        # 간단한 한국어 긍정 패턴(필요시 확장)
        ok_patterns = ["괜찮", "네", "문제없", "괜찮습니다", "괜찮아요", "예"]
        if any(pat in transcript for pat in ok_patterns):
            return "OK"
        # 무응답/다른 말은 ALERT
        return "ALERT"
    except Exception as e:
        print(f"[STT] 오류: {e}")
        return "ALERT"

if __name__ == "__main__":
    run_detection(on_fall=say_are_you_ok)
