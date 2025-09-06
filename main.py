# main.py
import threading
from fall_detection import run_detection, reset_detection_state
from send_alarm import say_are_you_ok  # "OK" 또는 "ALERT"를 반환해야 함

def _guard_flow():
    """백그라운드에서 질문→10초 대기→판단."""
    try:
        result = say_are_you_ok()  # "OK" or "ALERT"
        if result == "OK":
            # 정상 응답이면 감지 상태 리셋 (낙상 감지는 계속 순환 중)
            reset_detection_state()
        else:
            # 응답 없음/다른 말이면 아무 것도 안 함(기존 낙상 처리 계속)
            pass
    except Exception as e:
        print(f"[guard] 오류: {e}")

def on_fall_async():
    """
    fall_detection.run_detection(on_fall=...) 에 전달되는 콜백.
    여기서는 '스레드만' 띄우고 즉시 리턴 → 감지 루프는 멈추지 않음.
    """
    threading.Thread(target=_guard_flow, daemon=True).start()
    # 반환값 없음: 비동기이므로 즉시 복귀

if __name__ == "__main__":
    # 낙상 감지 루프 시작 (on_fall 비동기 콜백 연결)
    run_detection(on_fall=on_fall_async)
