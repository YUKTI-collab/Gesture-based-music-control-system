import cv2
import mediapipe as mp
import time
import numpy as np
import pygame
import os
import subprocess

# === Hand Detector Class ===
class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.75, trackCon=0.75):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks and handNo < len(self.results.multi_hand_landmarks):
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList

    def fingersUp(self, lmList):
        if not lmList:
            return []
        fingers = []
        for tipId in self.tipIds:
            knuckle_id = tipId - 2
            if lmList[tipId][2] < lmList[knuckle_id][2] - 30:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

# === macOS Volume Control ===
def set_volume(change):
    current_vol = int(subprocess.check_output(
        "osascript -e 'output volume of (get volume settings)'", shell=True).decode().strip())
    new_vol = max(0, min(100, current_vol + change))
    subprocess.call(f"osascript -e 'set volume output volume {new_vol}'", shell=True)
    return f"Volume: {new_vol}%"

def volume_up():
    return set_volume(10)

def volume_down():
    return set_volume(-10)

# === Music Controls ===
MUSIC_FOLDER = os.getcwd()
os.makedirs(MUSIC_FOLDER, exist_ok=True)

playlist = [os.path.join(MUSIC_FOLDER, f) for f in os.listdir(MUSIC_FOLDER)
            if f.lower().endswith((".mp3", ".wav", ".m4a", ".aac", ".ogg"))]
playlist.sort()

print("=== Music Files Found ===")
if not playlist:
    print("No supported music files found in:", MUSIC_FOLDER)
else:
    for song in playlist:
        print(song)

pygame.init()
pygame.mixer.init()
current_track = 0
is_playing = False

if playlist:
    try:
        pygame.mixer.music.load(playlist[current_track])
        pygame.mixer.music.play()
        is_playing = True
    except Exception as e:
        print(f"Error loading track: {playlist[current_track]}")
        print("Exception:", e)

def pause_music():
    global is_playing
    if is_playing:
        pygame.mixer.music.pause()
        is_playing = False
        return "Paused"
    return "Already paused"

def play_music():
    global is_playing
    if not is_playing:
        pygame.mixer.music.unpause()
        is_playing = True
        return "Playing"
    return "Already playing"

def next_track():
    global current_track, is_playing
    if playlist:
        current_track = (current_track + 1) % len(playlist)
        try:
            pygame.mixer.music.load(playlist[current_track])
            pygame.mixer.music.play()
            is_playing = True
            return "Next Track"
        except Exception as e:
            return f"Error: {e}"
    return "No playlist"

def play_beep():
    arr = (np.sin(2 * np.pi * 440 * np.arange(4410) / 44100) * 32767).astype(np.int16)
    beep_sound = pygame.mixer.Sound(buffer=arr.tobytes())
    beep_sound.play()

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Main Program ===
detector = HandDetector(detectionCon=0.8)
pTime = 0
last_action_time = 0
cooldown = 0.7

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from webcam.")
        continue

    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    current_time = time.time()

    h, w = img.shape[:2]
    roi_ratio = 0.6
    roi_x_min = int(w * (1 - roi_ratio) / 2)
    roi_x_max = int(w * (1 + roi_ratio) / 2)
    roi_y_min = int(h * (1 - roi_ratio) / 2)
    roi_y_max = int(h * (1 + roi_ratio) / 2)
    cv2.rectangle(img, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (255, 255, 0), 2)

    feedback_text = ""
    if lmList:
        palm_x, palm_y = lmList[0][1], lmList[0][2]
        fingers = detector.fingersUp(lmList)
        totalFingers = fingers.count(1)

        for i, isUp in enumerate(fingers):
            if isUp:
                tipId = detector.tipIds[i]
                cx, cy = lmList[tipId][1], lmList[tipId][2]
                cv2.rectangle(img, (cx - 20, cy - 20), (cx + 20, cy + 20), (0, 255, 0), 2)

        if roi_x_min <= palm_x <= roi_x_max and roi_y_min <= palm_y <= roi_y_max:
            if current_time - last_action_time > cooldown:
                if totalFingers == 4:
                    feedback_text = pause_music()
                    play_beep()
                    last_action_time = current_time
                elif totalFingers == 1:
                    feedback_text = volume_down()
                    play_beep()
                    last_action_time = current_time
                elif totalFingers == 2:
                    feedback_text = volume_up()
                    play_beep()
                    last_action_time = current_time
                elif totalFingers == 3:
                    feedback_text = next_track()
                    play_beep()
                    last_action_time = current_time
        else:
            feedback_text = "Place hand in ROI"

        cv2.putText(img, f"Fingers: {totalFingers}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        if feedback_text:
            cv2.putText(img, feedback_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    else:
        feedback_text = "No hand detected"
        cv2.putText(img, feedback_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Gesture-Controlled Music Player", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
