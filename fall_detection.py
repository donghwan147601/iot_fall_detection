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

# ====== 상태 ======
previous_y_values = None
first_point_threshold = None
second_point_threshold = None
falling_threshold = None

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
    if result.keypoints is None or result.keypoints.xy.shape[0] == 0:
        return []
    yvals = []
    for kp in result.keypoints.xy[0]:
        y = float(kp[1].cpu().numpy())
        if y != 0:
            yvals.append(y)
    return yvals

def init_thresholds_from_two_frames(res1, res2):
    global first_point_threshold, second_point_threshold, falling_threshold
    y1 = frame_coordinates(res1)
    y2 = frame_coordinates(res2)
    if len(y1) < 6 or len(y2) < 6:
        return False
    falling_threshold = ((y1[-1] - y1[0]) * 2.0 / 3.0) + 20.0
    first_point_threshold = abs(y1[0] - y2[0]) + 15.0
    second_point_threshold = abs(y1[5] - y2[5]) + 15.0
    print(f"[init] falling={falling_threshold:.1f}, p0={first_point_threshold:.1f}, p5={second_point_threshold:.1f}")
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
    """낙상 vs 눕기 판정. 처음 낙상 감지 시 on_fall() 1회 호출."""
    global previous_y_values, fallen_state, minimum, maximum
    global fall_start_time, elapsed_time_states, taking_video, fall_alerted

    if previous_y_values is not None and len(y_values) >= 6 and len(previous_y_values) >= 6:
        p0 = abs(previous_y_values[0] - y_values[0])
        p5 = abs(previous_y_values[5] - y_values[5])

        if falling_threshold is not None and (maximum - minimum) <= falling_threshold:
            # 눕기(완만 변화)
            if p0 <= first_point_threshold and p5 <= second_point_threshold:
                if fallen_state:
                    elapsed_time_states.append("Laying down")
                    check_falling_time_common()
                else:
                    # 안전
                    pass
            else:
                # 급격 변화 → 낙상 이벤트
                # 급격 변화 → 낙상 이벤트
                if not fallen_state:
                    fallen_state = True
                    taking_video = True
                    fall_start_time = time.time()
                    elapsed_time_states.append("Fallen")
                    print("[fall] start")

                    # === on_fall 콜백: 질문+STT → "OK"/"ALERT" ===
                    if on_fall and announce_flag is not None and not announce_flag["done"]:
                        try:
                            result = on_fall()   # "OK" or "ALERT"
                            if result == "OK":
                            # 정상 응답 → 전체 상태 초기화(클립 저장/알림 없이 종료)
                                _reset_detection_state()
                            else:
                             # 응답 없음/다른 말 → 위험상황 계속 관찰
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
    # 초기화: 유효 프레임 2개로 임계값 산출
    boot_stream = model(source=video_path, show=True, conf=CONF_THRES, stream=True, save=False)
    res_first = None; res_second = None
    for r in boot_stream:
        cv2.imshow("Video Feed", r.orig_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return
        yvals = frame_coordinates(r)
        if len(yvals) >= 6:
            if res_first is None:
                res_first = r; continue
            else:
                res_second = r
                if init_thresholds_from_two_frames(res_first, res_second):
                    break
                else:
                    res_first = res_second; res_second = None

    # 본 스트림 시작
    results = model(source=video_path, show=True, conf=CONF_THRES, stream=True, save=False)

    # 낙상 TTS 1회 방지용 플래그
    announce_flag = {"done": False}

    for r in results:
        # before 버퍼 유지
        if len(video_frames_before) >= BEFORE_MAX:
            video_frames_before.pop(0)
        video_frames_before.append(r.orig_img)

        # 낙상 후 프레임 수집(필요 시)
        if taking_video:
            if not frozen_video_frames_before:
                frozen_video_frames_before = deepcopy(video_frames_before)
            if len(video_frames_after) <= AFTER_MAX:
                video_frames_after.append(r.orig_img)

        # y값 기반 판정
        y_values = frame_coordinates(r)
        if len(y_values) >= 6:
            minimum = min(y_values); maximum = max(y_values)
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
