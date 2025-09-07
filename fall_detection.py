# fall_detection.py
import cv2, time
from copy import deepcopy
import numpy as np
from ultralytics import YOLO

# ====== 기본 파라미터 ======
VIDEO_PATH = "/home/donghwan/Downloads/IMG_5052.mov"
MODEL_PATH = "yolo_models/yolov8n-pose.pt"
CONF_THRES = 0.3

MIN_ELAPSED_TIME_THRESHOLD = 5
VIDEO_FPS = 10
BEFORE_MAX = 100
AFTER_MAX = 150
VEL_MARGIN = 80.0  # 속도 임계 여유(픽셀/초)

# ====== 상태 ======
previous_y_values = None
first_vel_threshold = None   # 키포인트 0 속도 임계
second_vel_threshold = None  # 키포인트 5 속도 임계
falling_threshold = None     # 자세 퍼짐(세로 범위) 임계

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

# ====== 유틸 ======
def _reset_detection_state():
    """낙상 감지 상태와 버퍼 초기화."""
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
    print("[state] 초기화 완료")

def frame_coordinates(result):
    """첫 번째 인물의 y좌표를 고정 길이(K)로 반환. 미검출/저신뢰=NaN."""
    if result.keypoints is None or result.keypoints.xy is None or result.keypoints.xy.shape[0] == 0:
        return None
    kp_xy = result.keypoints.xy[0].cpu().numpy()   # (K, 2)
    y = kp_xy[:, 1].astype(float)                  # (K,)
    # conf가 있으면 신뢰도<=0을 NaN 처리
    if getattr(result.keypoints, "conf", None) is not None:
        conf = result.keypoints.conf[0].cpu().numpy()  # (K,)
        y[conf <= 0] = np.nan
    # 0.0도 미검출로 간주
    y[y == 0.0] = np.nan
    return y  # 길이 K(보통 17), float+NaN

def valid_count(arr):
    if arr is None:
        return 0
    return int(np.count_nonzero(~np.isnan(arr)))

def nan_range(arr):
    if arr is None or valid_count(arr) == 0:
        return 0.0
    return float(np.nanmax(arr) - np.nanmin(arr))

def velocity_from_prev(prev_y, curr_y, fps):
    """이전/현재 y배열로 속도(픽셀/초) 계산. NaN은 그대로 전파."""
    return (curr_y - prev_y) * float(fps)

def init_thresholds_from_two_frames(res1, res2):
    """두 개의 유효 프레임으로 임계값 초기화."""
    global falling_threshold, first_vel_threshold, second_vel_threshold

    y1 = frame_coordinates(res1)
    y2 = frame_coordinates(res2)
    if valid_count(y1) < 6 or valid_count(y2) < 6:
        return False

    # 자세 퍼짐(눕기/서기 구분)
    falling_threshold = (nan_range(y1) * 2.0 / 3.0) + 20.0

    # 속도 임계치 (프레임 간 속도)
    v12 = velocity_from_prev(y1, y2, VIDEO_FPS)  # (K,)
    v0 = v12[0] if not np.isnan(v12[0]) else 0.0
    v5 = v12[5] if not np.isnan(v12[5]) else 0.0
    first_vel_threshold  = abs(v0) + VEL_MARGIN
    second_vel_threshold = abs(v5) + VEL_MARGIN

    print(f"[init] falling={falling_threshold:.1f}, v0_thr={first_vel_threshold:.1f} px/s, v5_thr={second_vel_threshold:.1f} px/s")
    return True

def check_falling_time_common():
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
    """속도(픽셀/초) 기반 낙상/눕기 판정. 처음 낙상 감지 시 on_fall() 1회 호출."""
    global previous_y_values, fallen_state, minimum, maximum
    global fall_start_time, elapsed_time_states, taking_video, fall_alerted
    global first_vel_threshold, second_vel_threshold, falling_threshold

    if previous_y_values is not None and valid_count(y_values) >= 6 and valid_count(previous_y_values) >= 6:
        # 현재 프레임 속도
        v_curr = velocity_from_prev(previous_y_values, y_values, VIDEO_FPS)
        first_speed  = abs(v_curr[0]) if not np.isnan(v_curr[0]) else 0.0
        second_speed = abs(v_curr[5]) if not np.isnan(v_curr[5]) else 0.0

        # 자세 퍼짐(세로 범위)
        range_y = nan_range(y_values)

        if (falling_threshold is not None) and (range_y <= falling_threshold):
            # 눕는 자세 범위: 속도 느리면 눕기, 빠르면 낙상 후보
            if (first_vel_threshold is not None) and (second_vel_threshold is not None) and \
               (first_speed <= first_vel_threshold) and (second_speed <= second_vel_threshold):
                if fallen_state:
                    elapsed_time_states.append("Laying down")
                    check_falling_time_common()
            else:
                # 낙상 이벤트
                if not fallen_state:
                    fallen_state = True
                    taking_video = True
                    fall_start_time = time.time()
                    elapsed_time_states.append("Fallen")
                    print("[fall] start (velocity-based)")

                    # === on_fall 콜백: 질문+STT → "OK"/"ALERT" ===
                    if on_fall and announce_flag is not None and not announce_flag.get("done", False):
                        try:
                            result = on_fall()   # "OK" or "ALERT"
                            if result == "OK":
                                # 정상 응답 → 전체 상태 초기화(클립 저장/알림 없이 종료)
                                _reset_detection_state()
                            else:
                                # 응답 없음/다른 말 → 계속 관찰
                                announce_flag["done"] = True
                        except Exception as e:
                            print(f"[on_fall] 오류: {e}")
                            announce_flag["done"] = True
                else:
                    elapsed_time_states.append("Fallen")
                    check_falling_time_common()
        else:
            # 안전(복귀)
            if fall_alerted:
                taking_video = True
            else:
                fallen_state = False
                taking_video = False
                frozen_video_frames_before.clear()
                video_frames_after.clear()
            # 다음 낙상 알림을 위해 플래그 리셋
            if announce_flag is not None:
                announce_flag["done"] = False
            fall_start_time = None
            elapsed_time_states.clear()

    previous_y_values = y_values

def run_detection(video_path=VIDEO_PATH, model_path=MODEL_PATH, on_fall=None):
    """주 루프. on_fall()을 전달하면 최초 낙상 시 호출."""
    global minimum, maximum, taking_video, fallen_state
    global frozen_video_frames_before, video_frames_before, video_frames_after

    model = YOLO(model_path)

    # === 부트스트랩: 유효 프레임 2개로 임계 산출 ===
    boot_stream = model(source=video_path, show=True, conf=CONF_THRES, stream=True, save=False)
    res_first = None; res_second = None
    for r in boot_stream:
        # 보기
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

    # === 본 스트림 ===
    results = model(source=video_path, show=True, conf=CONF_THRES, stream=True, save=False)

    # 낙상 TTS 1회 방지용 플래그
    announce_flag = {"done": False}

    for r in results:
        # before 버퍼 유지
        if len(video_frames_before) >= BEFORE_MAX:
            video_frames_before.pop(0)
        video_frames_before.append(r.orig_img)

        # 낙상 후 프레임 수집
        if taking_video:
            if not frozen_video_frames_before:
                frozen_video_frames_before = deepcopy(video_frames_before)
            if len(video_frames_after) <= AFTER_MAX:
                video_frames_after.append(r.orig_img)

        # y값 기반 판정
        y_values = frame_coordinates(r)
        if valid_count(y_values) >= 6:
            minimum = float(np.nanmin(y_values))
            maximum = float(np.nanmax(y_values))
            check_falling(y_values, on_fall=on_fall, announce_flag=announce_flag)
        else:
            # 미검출 중 낙상 진행 시 타이머
            if fallen_state:
                elapsed_time_states.append("No human detected")
                check_falling_time_common()

        # 오버레이
        if fallen_state:
            cv2.putText(r.orig_img, "fall detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Video Feed", r.orig_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
