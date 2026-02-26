# fall_detection.py
import cv2, time
from copy import deepcopy
import numpy as np
from ultralytics import YOLO

# ====== Basic Parameters ======
VIDEO_PATH = "/home/donghwan/Downloads/IMG_5052.mov"
MODEL_PATH = "yolo_models/yolov8n-pose.pt"
CONF_THRES = 0.3

MIN_ELAPSED_TIME_THRESHOLD = 5  # Time to maintain fallen state before alert (seconds)
VIDEO_FPS = 10
BEFORE_MAX = 100               # Buffer size for frames before fall
AFTER_MAX = 150                # Buffer size for frames after fall
VEL_MARGIN = 80.0              # Velocity margin threshold (pixels/sec)

# ====== Global State Variables ======
previous_y_values = None
first_vel_threshold = None     # Velocity threshold for keypoint 0 (Nose)
second_vel_threshold = None    # Velocity threshold for keypoint 5 (Shoulder)
falling_threshold = None       # Height range threshold to distinguish standing/laying

fallen_state = False
taking_video = False
fall_start_time = None
fall_alerted = False
elapsed_time_states = []

video_frames_before = []
frozen_video_frames_before = []
video_frames_after = []

minimum = 0.0
maximum = 0.0

# ====== Utility Functions ======
def _reset_detection_state():
    """Resets all detection states and buffers."""
    global previous_y_values, fallen_state, taking_video, fall_start_time, fall_alerted
    global elapsed_time_states, video_frames_before, frozen_video_frames_before, video_frames_after
    previous_y_values = None
    fallen_state = False
    taking_video = False
    fall_start_time = None
    fall_alerted = False
    elapsed_time_states.clear()
    video_frames_before.clear()
    frozen_video_frames_before.clear()
    video_frames_after.clear()
    print("[state] State reset complete.")

def frame_coordinates(result):
    """Extracts y-coordinates of the first detected person as a fixed-length array (K)."""
    if result.keypoints is None or result.keypoints.xy is None or result.keypoints.xy.shape[0] == 0:
        return None
    kp_xy = result.keypoints.xy[0].cpu().numpy()   # (K, 2)
    y = kp_xy[:, 1].astype(float)                  # (K,)
    
    # Treat keypoints with confidence <= 0 as NaN
    if getattr(result.keypoints, "conf", None) is not None:
        conf = result.keypoints.conf[0].cpu().numpy()  # (K,)
        y[conf <= 0] = np.nan
    
    # Also treat 0.0 coordinates as undetected (NaN)
    y[y == 0.0] = np.nan
    return y  # Length K (usually 17), contains floats and NaNs

def valid_count(arr):
    """Counts non-NaN elements in the array."""
    if arr is None:
        return 0
    return int(np.count_nonzero(~np.isnan(arr)))

def nan_range(arr):
    """Calculates the vertical range (max - min) ignoring NaNs."""
    if arr is None or valid_count(arr) == 0:
        return 0.0
    return float(np.nanmax(arr) - np.nanmin(arr))

def velocity_from_prev(prev_y, curr_y, fps):
    """Calculates velocity (pixels/sec) between previous and current y-coordinates."""
    return (curr_y - prev_y) * float(fps)

def init_thresholds_from_two_frames(res1, res2):
    """Bootstraps thresholds using two valid initial frames."""
    global falling_threshold, first_vel_threshold, second_vel_threshold

    y1 = frame_coordinates(res1)
    y2 = frame_coordinates(res2)
    if valid_count(y1) < 6 or valid_count(y2) < 6:
        return False

    # Set posture threshold (standing vs laying)
    falling_threshold = (nan_range(y1) * 2.0 / 3.0) + 20.0

    # Set velocity thresholds based on inter-frame movement
    v12 = velocity_from_prev(y1, y2, VIDEO_FPS)  # (K,)
    v0 = v12[0] if not np.isnan(v12[0]) else 0.0
    v5 = v12[5] if not np.isnan(v12[5]) else 0.0
    first_vel_threshold  = abs(v0) + VEL_MARGIN
    second_vel_threshold = abs(v5) + VEL_MARGIN

    print(f"[init] falling_thr={falling_threshold:.1f}, v0_thr={first_vel_threshold:.1f} px/s, v5_thr={second_vel_threshold:.1f} px/s")
    return True

def check_falling_time_common():
    """Checks if the fallen state duration exceeds the safety threshold."""
    global fall_start_time, elapsed_time_states, fall_alerted, taking_video, fallen_state
    if fall_start_time is None:
        return
    dur = time.time() - fall_start_time
    if dur >= MIN_ELAPSED_TIME_THRESHOLD:
        print("[fall] ALERT! duration >= threshold")
        fall_alerted = True
        taking_video = True
        fall_start_time = None
        elapsed_time_states.clear()
        fallen_state = False

def check_falling(y_values, on_fall=None, announce_flag=None):
    """Determines fall vs laying down based on velocity and vertical range."""
    global previous_y_values, fallen_state, minimum, maximum
    global fall_start_time, elapsed_time_states, taking_video, fall_alerted
    global first_vel_threshold, second_vel_threshold, falling_threshold

    if previous_y_values is not None and valid_count(y_values) >= 6 and valid_count(previous_y_values) >= 6:
        # Calculate current velocity
        v_curr = velocity_from_prev(previous_y_values, y_values, VIDEO_FPS)
        first_speed  = abs(v_curr[0]) if not np.isnan(v_curr[0]) else 0.0
        second_speed = abs(v_curr[5]) if not np.isnan(v_curr[5]) else 0.0

        # Current vertical span of the posture
        range_y = nan_range(y_values)

        if (falling_threshold is not None) and (range_y <= falling_threshold):
            # Case: Posture range indicates laying down
            if (first_vel_threshold is not None) and (second_vel_threshold is not None) and \
               (first_speed <= first_vel_threshold) and (second_speed <= second_vel_threshold):
                # Normal laying down (slow speed)
                if fallen_state:
                    elapsed_time_states.append("Laying down")
                    check_falling_time_common()
            else:
                # Fall event (fast speed)
                if not fallen_state:
                    fallen_state = True
                    taking_video = True
                    fall_start_time = time.time()
                    elapsed_time_states.append("Fallen")
                    print("[fall] start (velocity-based)")

                    # Trigger on_fall callback for voice/STT verification
                    if on_fall and announce_flag is not None and not announce_flag.get("done", False):
                        try:
                            result = on_fall()   # Expects "OK" or "ALERT"
                            if result == "OK":
                                # False alarm / User responded OK
                                _reset_detection_state()
                            else:
                                # No response or help requested
                                announce_flag["done"] = True
                        except Exception as e:
                            print(f"[on_fall] Error in callback: {e}")
                            announce_flag["done"] = True
                else:
                    elapsed_time_states.append("Fallen")
                    check_falling_time_common()
        else:
            # Case: Safe (Standing/Recovery)
            if fall_alerted:
                taking_video = True
            else:
                fallen_state = False
                taking_video = False
                frozen_video_frames_before.clear()
                video_frames_after.clear()
            
            # Reset flag for future fall events
            if announce_flag is not None:
                announce_flag["done"] = False
            fall_start_time = None
            elapsed_time_states.clear()

    previous_y_values = y_values

def run_detection(video_path=VIDEO_PATH, model_path=MODEL_PATH, on_fall=None):
    """Main loop for inference and fall detection logic."""
    global minimum, maximum, taking_video, fallen_state
    global frozen_video_frames_before, video_frames_before, video_frames_after

    model = YOLO(model_path)

    # === Bootstrapping: Process first two valid frames to set thresholds ===
    boot_stream = model(source=video_path, show=True, conf=CONF_THRES, stream=True, save=False)
    res_first = None; res_second = None
    for r in boot_stream:
        cv2.imshow("Video Feed", r.orig_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return
        yvals = frame_coordinates(r)
        if valid_count(yvals) >= 6:
            if res_first is None:
                res_first = r
            else:
                res_second = r
                if init_thresholds_from_two_frames(res_first, res_second):
                    break
                else:
                    res_first = res_second; res_second = None

    # === Main Stream Inference ===
    results = model(source=video_path, show=True, conf=CONF_THRES, stream=True, save=False)
    announce_flag = {"done": False} # Prevents redundant TTS/Alerts

    for r in results:
        # Maintain "Before" buffer
        if len(video_frames_before) >= BEFORE_MAX:
            video_frames_before.pop(0)
        video_frames_before.append(r.orig_img)

        # Collect "After" frames if fall detected
        if taking_video:
            if not frozen_video_frames_before:
                frozen_video_frames_before = deepcopy(video_frames_before)
            if len(video_frames_after) <= AFTER_MAX:
                video_frames_after.append(r.orig_img)

        # Decision based on y-coordinates
        y_values = frame_coordinates(r)
        if valid_count(y_values) >= 6:
            minimum = float(np.nanmin(y_values))
            maximum = float(np.nanmax(y_values))
            check_falling(y_values, on_fall=on_fall, announce_flag=announce_flag)
        else:
            # Handle cases where person is missing during a fall event
            if fallen_state:
                elapsed_time_states.append("No human detected")
                check_falling_time_common()

        # Visual Overlay
        if fallen_state:
            cv2.putText(r.orig_img, "FALL DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Video Feed", r.orig_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
