import cv2
import mediapipe as mp

# 1. Detect available camera indices
def find_available_camera(max_index=2):
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
        cap.release()
    return available

print("Available camera indices:", find_available_camera())

# 2. Select the first available camera
idx = find_available_camera()
if not idx:
    print("⚠️ No webcam found. Exiting.")
    exit()
camera_index = idx[0]
print(f"➡️ Using camera index: {camera_index}")

# 3. Initialize capture (try different backends if needed)
cap = cv2.VideoCapture(camera_index)
# On Windows, if it doesn't work, try:
# cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

print("Camera opened:", cap.isOpened())
if not cap.isOpened():
    exit("⚠️ Error: Could not open camera")

# 4. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 5. Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read a frame. Exiting.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Multi-Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
