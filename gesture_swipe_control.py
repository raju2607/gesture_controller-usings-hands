import cv2
import mediapipe as mp
import pyautogui

# Find available webcam index using DirectShow backend
def find_camera(max_idx=3):
    for i in range(max_idx):
        cap_t = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap_t.isOpened():
            cap_t.release()
            return i
    return None

camera_index = find_camera()
if camera_index is None:
    exit("⚠️ No working webcam found.")
print("➡️ Using camera index:", camera_index)

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
print("Camera opened:", cap.isOpened())
if not cap.isOpened():
    exit("⚠️ Could not open camera.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Variables for swipe detection
last_x, last_y = None, None
SWIPE_THRESH = 0.15  # adjust sensitivity

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame read failed. Exiting.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Gesture detection
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark[8]
        curr_x, curr_y = lm.x, lm.y

        if last_x is not None:
            dx, dy = curr_x - last_x, curr_y - last_y
            if abs(dx) > abs(dy) and abs(dx) > SWIPE_THRESH:
                direction = 'left' if dx < 0 else 'right'
                pyautogui.press(direction)
                print(f"Swipe {direction.title()}")
            elif abs(dy) > SWIPE_THRESH:
                direction = 'up' if dy < 0 else 'down'
                pyautogui.press(direction)
                print(f"Swipe {direction.title()}")

        last_x, last_y = curr_x, curr_y
    else:
        last_x, last_y = None, None

    # Draw hand landmarks and show frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Swipe Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break  # Exit on ESC

cap.release()
cv2.destroyAllWindows()
