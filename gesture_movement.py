import cv2
import numpy as np
import math
import tensorflow as tf

model = tf.lite.Interpreter(model_path="movenet_thunder.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

reactions = {
    "idle": cv2.imread("react_idle.png"),
    "finger": cv2.imread("react1.png"),
    "mount": cv2.imread("react2.png"),
}

MIN_CONFIDENCE_SCORE = 0.4
STABILITY_THRESHOLD = 5

def preprocess_frame(frame):
    img = cv2.resize(frame, (256, 256))
    img = np.expand_dims(img.astype(np.float32), axis=0)
    return img

def get_keypoints(frame):
    img = preprocess_frame(frame)
    model.set_tensor(input_details[0]["index"], img)
    model.invoke()
    return model.get_tensor(output_details[0]["index"])[0, 0, :, :]

def safe_point(pt, w, h, min_conf=MIN_CONFIDENCE_SCORE):
    y, x, c = pt
    if np.isnan(x) or np.isnan(y) or c < min_conf:
        return None
    return (int(x * w), int(y * h), c)

def distance(p1, p2):
    if p1 is None or p2 is None:
        return np.inf
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

cap = cv2.VideoCapture(0)

previous_gesture = "idle"
stability_counter = 0

MOUTH_DISTANCE_THRESHOLD = 130
FINGER_Y_OFFSET = 60
MAX_HORIZONTAL_OFFSET = 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    keypoints = get_keypoints(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    kp = [safe_point(pt, w, h) for pt in keypoints]

    nose = kp[0]
    left_wrist = kp[9]
    right_wrist = kp[10]
    left_shoulder = kp[5]
    right_shoulder = kp[6]

    gesture = "idle"

    if nose:
        left_close = left_wrist and distance(left_wrist, nose) < MOUTH_DISTANCE_THRESHOLD
        right_close = right_wrist and distance(right_wrist, nose) < MOUTH_DISTANCE_THRESHOLD
        if left_close or right_close:
            gesture = "mount"

        else:
            head_y_ref = nose[1] - FINGER_Y_OFFSET
            if left_shoulder and left_shoulder[1] < head_y_ref:
                head_y_ref = left_shoulder[1]
            if right_shoulder and right_shoulder[1] < head_y_ref:
                head_y_ref = right_shoulder[1]

            wrist_above = (
                (left_wrist and left_wrist[1] < head_y_ref)
                or (right_wrist and right_wrist[1] < head_y_ref)
            )

            horizontal_ok = True
            if left_wrist and abs(left_wrist[0] - nose[0]) > MAX_HORIZONTAL_OFFSET:
                horizontal_ok = False
            if right_wrist and abs(right_wrist[0] - nose[0]) > MAX_HORIZONTAL_OFFSET:
                horizontal_ok = False

            if wrist_above and horizontal_ok:
                gesture = "finger"
            else:
                gesture = "idle"

    if gesture == previous_gesture:
        stability_counter += 1
    else:
        stability_counter = 1
        previous_gesture = gesture

    if stability_counter < STABILITY_THRESHOLD:
        gesture = previous_gesture

    reaction_img = reactions.get(gesture, reactions["idle"])
    if reaction_img is None:
        reaction_img = reactions["idle"]

    reaction_img = cv2.resize(reaction_img, (w, h))
    combined = cv2.hconcat([frame, reaction_img])

    cv2.imshow("Gesture Reaction (MoveNet Thunder - Mount Fixed)", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()
