# main.py
import threading
from fall_detection import run_detection, _reset_detection_state # Changed to match your fall_detection.py
from send_alarm import say_are_you_ok  # Expected to return "OK" or "ALERT"

def _guard_flow():
    """
    Handles the background verification flow: 
    Ask "Are you okay?" -> Wait for response/timeout -> Determine action.
    """
    try:
        # Blocks only within this thread while waiting for voice/STT response
        result = say_are_you_ok()  # Returns "OK" or "ALERT"
        
        if result == "OK":
            # If user responds "OK", reset the detection state to resume normal monitoring
            _reset_detection_state()
            print("[guard] User responded 'OK'. Resetting state.")
        else:
            # If no response or help is requested, take no action (continue fall alert process)
            print("[guard] Alert confirmed or no response. Proceeding with emergency protocol.")
            pass
    except Exception as e:
        print(f"[guard] Error in verification flow: {e}")

def on_fall_async():
    """
    Asynchronous callback passed to fall_detection.run_detection.
    Starts the verification thread and returns immediately to keep the detection loop running.
    """
    # Start the verification flow in a daemon thread so it doesn't freeze the video feed
    threading.Thread(target=_guard_flow, daemon=True).start()
    # No return value: Returns immediately as it is asynchronous

if __name__ == "__main__":
    # Start the main fall detection loop with the asynchronous callback
    print("[main] Starting fall detection system...")
    run_detection(on_fall=on_fall_async)
