# import cv2
# import torch
# import mediapipe as mp
# import numpy as np
# from ultralytics import YOLO

# # Load trained YOLO model
# try:
#     model = YOLO("best.pt")
#     print("YOLO model loaded successfully!")
# except Exception as e:
#     print(f"Error loading YOLO model: {e}")
#     exit()

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils

# # Webcam
# cap = cv2.VideoCapture(0)
# exit_flag = [False]  # Using list to make it mutable in callback

# # Callback to check mouse click on "Close" button
# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if 10 <= x <= 110 and 10 <= y <= 50:
#             print("Closing camera...")
#             exit_flag[0] = True

# cv2.namedWindow("Sign Language Detection")
# cv2.setMouseCallback("Sign Language Detection", click_event)

# with mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret or exit_flag[0]:
#             break

#         frame = cv2.flip(frame, 1)
#         frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         results = hands.process(rgb_frame)
#         h, w, _ = frame.shape

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 x_min, y_min = w, h
#                 x_max, y_max = 0, 0

#                 for landmark in hand_landmarks.landmark:
#                     x, y = int(landmark.x * w), int(landmark.y * h)
#                     x_min, y_min = min(x_min, x), min(y_min, y)
#                     x_max, y_max = max(x_max, x), max(y_max, y)

#                 margin = 40
#                 x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
#                 x_max, y_max = min(x_max + margin, w), min(y_max + margin, h)

#                 hand_input = frame[y_min:y_max, x_min:x_max]

#                 if hand_input.size > 0:
#                     results_yolo = model(hand_input, conf=0.15)

#                     for result in results_yolo:
#                         boxes = result.boxes
#                         for i in range(len(boxes.cls)):
#                             class_id = int(boxes.cls[i])
#                             conf = float(boxes.conf[i])
#                             sign_class = model.names[class_id]

#                             y_offset = y_min - 10 + (i * 35)
#                             y_text = max(30, y_offset)

#                             cv2.putText(frame, f"{sign_class} ({conf:.2f})",
#                                         (x_min, y_text),
#                                         cv2.FONT_HERSHEY_SIMPLEX,
#                                         0.8, (0, 255, 0), 2)
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#         # Draw "Close" button
#         cv2.rectangle(frame, (10, 10), (110, 50), (0, 0, 255), -1)
#         cv2.putText(frame, "Close", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         cv2.imshow("Sign Language Detection", frame)

#         if cv2.waitKey(1) & 0xFF == 27:  # ESC as backup
#             break

# cap.release()
# cv2.destroyAllWindows()


