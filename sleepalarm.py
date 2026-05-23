import cv2
import mediapipe as mp
import numpy as np
import winsound
import time
import threading
import tkinter as tk
from datetime import datetime

EAR_THRESHOLD   = 0.26
ALARM_DELAY     = 2.5
YAWN_THRESHOLD  = 0.6
HEAD_DROP_ANGLE = 20
BEEP_FREQ       = 1000
BEEP_DURATION   = 500

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH_TOP = 13
MOUTH_BOT = 14
MOUTH_L   = 78
MOUTH_R   = 308
NOSE_TIP  = 1
CHIN      = 152

def get_ear(landmarks, indices):
    p    = lambda i: np.array([landmarks[i].x, landmarks[i].y])
    dist = lambda a, b: np.linalg.norm(p(a) - p(b))
    return (dist(indices[1], indices[5]) + dist(indices[2], indices[4])) \
           / (2.0 * dist(indices[0], indices[3]))

def get_mar(landmarks):
    p    = lambda i: np.array([landmarks[i].x, landmarks[i].y])
    dist = lambda a, b: np.linalg.norm(p(a) - p(b))
    return dist(MOUTH_TOP, MOUTH_BOT) / dist(MOUTH_L, MOUTH_R)

def get_head_angle(landmarks):
    nose  = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y])
    chin  = np.array([landmarks[CHIN].x,     landmarks[CHIN].y])
    diff  = chin - nose
    angle = np.degrees(np.arctan2(diff[0], diff[1]))
    return abs(angle)

alarm_on    = False
alert_shown = False

def alarm_loop():
    while alarm_on:
        winsound.Beep(BEEP_FREQ, BEEP_DURATION)
        time.sleep(0.1)

def start_alarm():
    global alarm_on
    if not alarm_on:
        alarm_on = True
        threading.Thread(target=alarm_loop, daemon=True).start()

def stop_alarm():
    global alarm_on
    alarm_on = False

def show_fullscreen_alert(reason):
    def _show():
        global alarm_on, alert_shown
        root = tk.Tk()
        root.attributes('-fullscreen', True)
        root.attributes('-topmost', True)
        root.configure(bg='red')
        tk.Label(root, text="⚠ WAKE UP! ⚠",
                 font=("Arial", 80, "bold"),
                 bg="red", fg="white").pack(expand=True)
        tk.Label(root, text=reason,
                 font=("Arial", 30),
                 bg="red", fg="white").pack()
        def dismiss():
            global alarm_on, alert_shown
            alarm_on    = False
            alert_shown = False
            root.destroy()
        tk.Button(root, text="I'm Awake",
                  font=("Arial", 20),
                  command=dismiss).pack(pady=40)
        root.protocol("WM_DELETE_WINDOW", dismiss)
        root.mainloop()
    threading.Thread(target=_show, daemon=True).start()

session_start    = datetime.now()
total_focus_time = 0
total_drowsy     = 0
yawn_count       = 0
distraction_count= 0
last_focus_start = time.time()
was_focused      = True

def get_focus_level(ear, mar, head_angle):
    if ear < EAR_THRESHOLD:   return "Drowsy"
    if mar > YAWN_THRESHOLD:  return "Yawning"
    if head_angle > HEAD_DROP_ANGLE: return "Head Dropping"
    return "Focused"

def draw_dashboard(frame, ear, mar, head_angle, focus,
                   elapsed_closed, study_mins, focus_pct):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def put(text, y, color=(255,255,255), size=0.55):
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    put("== SLEEP ALARM ==", 30, (100,255,100), 0.65)
    put(f"EAR     : {ear:.3f}", 70)
    put(f"MAR     : {mar:.3f}", 100)
    put(f"Head    : {head_angle:.1f} deg", 130)
    put(f"Eyes closed: {elapsed_closed:.1f}s", 165)

    colors = {
        "Focused":       (0, 220, 0),
        "Drowsy":        (0, 0, 255),
        "Yawning":       (0, 165, 255),
        "Head Dropping": (0, 0, 255),
    }
    put(f"Status  : {focus}", 205, colors.get(focus, (255,255,255)), 0.6)
    put("-- Productivity --", 250, (180,180,180))
    put(f"Study   : {study_mins:.0f} min", 285)
    put(f"Focus   : {focus_pct:.0f}%", 315)
    put(f"Yawns   : {yawn_count}", 345)
    put(f"Distractions: {distraction_count}", 375)
    put("F=fullscreen  Q=quit", h-15, (120,120,120), 0.45)


cap             = cv2.VideoCapture(0)
eye_closed_since= None
head_drop_since = None
yawn_active     = False
fullscreen      = False

print("Sleep Alarm running... Press Q to quit, F for fullscreen.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    elapsed_closed = 0.0
    focus          = "No Face"
    ear = mar = head_angle = 0.0

    if results.multi_face_landmarks:
        lms        = results.multi_face_landmarks[0].landmark
        ear        = (get_ear(lms, LEFT_EYE) + get_ear(lms, RIGHT_EYE)) / 2.0
        mar        = get_mar(lms)
        head_angle = get_head_angle(lms)
        focus      = get_focus_level(ear, mar, head_angle)
        
        if mar > YAWN_THRESHOLD and not yawn_active:
            yawn_count += 1
            yawn_active = True
        elif mar <= YAWN_THRESHOLD:
            yawn_active = False

    
        if ear < EAR_THRESHOLD:
            if eye_closed_since is None:
                eye_closed_since = time.time()
            elapsed_closed = time.time() - eye_closed_since
            if elapsed_closed >= ALARM_DELAY and not alert_shown:
                alert_shown = True
                total_drowsy += 1
                distraction_count += 1
                start_alarm()
                show_fullscreen_alert("Eyes closed while studying!")
        else:
            eye_closed_since = None
            if not alert_shown:
                stop_alarm()

        
        if head_angle > HEAD_DROP_ANGLE:
            if head_drop_since is None:
                head_drop_since = time.time()
            elif time.time() - head_drop_since > 2.0 and not alert_shown:
                alert_shown = True
                distraction_count += 1
                start_alarm()
                show_fullscreen_alert("Head dropping — you're falling asleep!")
        else:
            head_drop_since = None

        
        if focus == "Focused":
            if not was_focused:
                last_focus_start = time.time()
                was_focused      = True
        else:
            if was_focused:
                total_focus_time += time.time() - last_focus_start
                was_focused       = False

    study_secs = (datetime.now() - session_start).total_seconds()
    study_mins = study_secs / 60
    focus_pct  = (total_focus_time / study_secs * 100) if study_secs > 0 else 100

    draw_dashboard(frame, ear, mar, head_angle, focus,
                   elapsed_closed, study_mins, focus_pct)

    cv2.imshow("Sleep Alarm", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('f'):
        if fullscreen:
            cv2.setWindowProperty("Sleep Alarm", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            fullscreen = False
        else:
            cv2.namedWindow("Sleep Alarm", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Sleep Alarm", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            fullscreen = True


stop_alarm()
cap.release()
cv2.destroyAllWindows()

if was_focused:
    total_focus_time += time.time() - last_focus_start

study_secs = (datetime.now() - session_start).total_seconds()
focus_pct  = (total_focus_time / study_secs * 100) if study_secs > 0 else 100

print("\n======= STUDY SESSION REPORT =======")
print(f"Total study time : {study_secs/60:.1f} minutes")
print(f"Focus time       : {total_focus_time/60:.1f} minutes")
print(f"Focus level      : {focus_pct:.0f}%")
print(f"Yawn count       : {yawn_count}")
print(f"Drowsy alerts    : {total_drowsy}")
print(f"Distractions     : {distraction_count}")
print("=====================================")
