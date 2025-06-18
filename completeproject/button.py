# Real-Time Alphabet Gesture Recognition and Sentence Construction

# Install dependencies if not already installed:
# !pip install ultralytics opencv-python transformers torch torchvision

from ultralytics import YOLO
import cv2
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the YOLOv8 model (trained on alphabet gestures) on CPU
yolo_model_path = "D:/FinalProject(GestureBridge)/Gesture Bridge/best.pt"  # <-- Replace with your .pt weights file path
try:
    model = YOLO(yolo_model_path)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)
model.overrides['device'] = 'cpu'  # ensure the model runs on CPU

# Load a lightweight GPT-2 model (DistilGPT-2) for sentence generation on CPU
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
language_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
language_model.to('cpu')

# Open webcam video stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Reduce resolution for speed (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Parameters for smoothing and segmentation
STABLE_LETTER_FRAMES = 5
WORD_BREAK_FRAMES = 20
LETTER_CONF_THRESHOLD = 0.5

# State variables
current_letter = None
letter_frame_count = 0
current_word_letters = []
recognized_words = []
no_detection_frames = 0
sentence_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=LETTER_CONF_THRESHOLD)
    if isinstance(results, (list, tuple)):
        results = results[0] if len(results) > 0 else None

    letter_detected = None
    bbox = None
    if results is not None and len(results.boxes) > 0:
        best_conf = 0.0
        best_box = None
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf >= LETTER_CONF_THRESHOLD and conf > best_conf:
                best_conf = conf
                best_box = box
        if best_box is not None:
            class_id = int(best_box.cls.item())
            label = model.names.get(class_id, str(class_id))
            letter_detected = label
            x1, y1, x2, y2 = best_box.xyxy.tolist()[0]
            bbox = (int(x1), int(y1), int(x2), int(y2))

    if letter_detected:
        no_detection_frames = 0
        if current_letter is None:
            current_letter = letter_detected
            letter_frame_count = 1
        elif letter_detected == current_letter:
            letter_frame_count += 1
        else:
            if letter_frame_count >= STABLE_LETTER_FRAMES:
                current_word_letters.append(current_letter)
            current_letter = letter_detected
            letter_frame_count = 1
    else:
        if current_letter is not None:
            no_detection_frames += 1
            if no_detection_frames >= WORD_BREAK_FRAMES:
                if current_letter and letter_frame_count >= STABLE_LETTER_FRAMES:
                    current_word_letters.append(current_letter)
                if current_word_letters:
                    word = "".join(current_word_letters)
                    if word != "I":
                        word = word.lower()
                    if len(recognized_words) == 0:
                        word = word.capitalize()
                    recognized_words.append(word)

                    raw_sentence = " ".join(recognized_words)
                    input_ids = tokenizer.encode(raw_sentence, return_tensors='pt')
                    output_ids = language_model.generate(input_ids, max_new_tokens=1, do_sample=False)
                    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                    sentence_text = raw_sentence
                    added_text = ""
                    if generated_text.startswith(raw_sentence):
                        added_text = generated_text[len(raw_sentence):].strip()
                    else:
                        added_text = generated_text.strip()

                    if added_text.lower() in ("is", "are", "to"):
                        if added_text.lower() == "is" and "name " in raw_sentence and "name is" not in raw_sentence:
                            sentence_text = raw_sentence.replace("name ", "name is ", 1)
                        elif added_text.lower() == "are" and "how you" in raw_sentence:
                            sentence_text = raw_sentence.replace("how you", "how are you", 1)
                        elif added_text.lower() == "to" and "nice meet you" in raw_sentence:
                            sentence_text = raw_sentence.replace("nice meet you", "nice to meet you", 1)
                        else:
                            sentence_text = raw_sentence + " " + added_text
                    elif added_text in (".", "?", "!"):
                        sentence_text = raw_sentence.strip() + added_text
                    else:
                        sentence_text = raw_sentence

                    if sentence_text:
                        sentence_text = sentence_text.strip()
                        sentence_text = sentence_text[0].upper() + sentence_text[1:]

                current_letter = None
                current_word_letters = []
                letter_frame_count = 0
                no_detection_frames = 0

    if current_word_letters:
        temp_sentence = " ".join(recognized_words + ["".join(current_word_letters)])
        sentence_text = temp_sentence
        if sentence_text:
            sentence_text = sentence_text.strip()
            sentence_text = sentence_text[0].upper() + sentence_text[1:]
    else:
        if recognized_words:
            sentence_text = " ".join(recognized_words)
            sentence_text = sentence_text.strip()
            sentence_text = sentence_text[0].upper() + sentence_text[1:]

    if bbox and letter_detected:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{letter_detected}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if sentence_text:
        cv2.putText(frame, sentence_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Alphabet Gesture Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    # Handle key press
    if key == ord('q'):
        break
    if key == ord('c'):
        current_letter = None
        letter_frame_count = 0
        current_word_letters = []
        recognized_words = []
        no_detection_frames = 0
        sentence_text = ""
        print("Sentence cleared!")
    if key == 8:  # Backspace key
        if current_word_letters:
            removed_letter = current_word_letters.pop()
            print(f"Removed last letter: {removed_letter}")
        elif recognized_words:
            last_word = recognized_words[-1]
            if len(last_word) > 1:
                recognized_words[-1] = last_word[:-1]
            else:
                recognized_words.pop()
            print("Removed last letter from recognized words.")
        # After removing, rebuild the sentence_text
        temp_sentence = " ".join(recognized_words + ["".join(current_word_letters)])
        sentence_text = temp_sentence
        if sentence_text:
            sentence_text = sentence_text.strip()
            sentence_text = sentence_text[0].upper() + sentence_text[1:]

# Cleanup
cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import mediapipe as mp
# import pyautogui
# import keyboard

# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands

# # Initialize MediaPipe Hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # Define keyboard layout
# keys = [
#     ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
#     ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
#     ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
# ]

# # Initialize sentence and cursor
# sentence_text = []
# cursor_pos = 0

# def draw_keyboard(image):
#     key_width = 60
#     key_height = 60
#     gap = 10
#     start_x = 50
#     start_y = 50
#     for i, row in enumerate(keys):
#         for j, key in enumerate(row):
#             x = start_x + j * (key_width + gap) + (i * (key_width // 2))
#             y = start_y + i * (key_height + gap)
#             cv2.rectangle(image, (x, y), (x + key_width, y + key_height), (255, 0, 0), 2)
#             cv2.putText(image, key, (x + 15, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     return image

# def detect_key(x, y):
#     key_width = 60
#     key_height = 60
#     gap = 10
#     start_x = 50
#     start_y = 50
#     for i, row in enumerate(keys):
#         for j, key in enumerate(row):
#             key_x = start_x + j * (key_width + gap) + (i * (key_width // 2))
#             key_y = start_y + i * (key_height + gap)
#             if key_x < x < key_x + key_width and key_y < y < key_y + key_height:
#                 return key
#     return None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     frame = cv2.flip(frame, 1)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             h, w, c = frame.shape
#             index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#             x = int(index_finger_tip.x * w)
#             y = int(index_finger_tip.y * h)

#             detected_key = detect_key(x, y)
#             if detected_key:
#                 print("Detected key:", detected_key)
#                 sentence_text.insert(cursor_pos, detected_key)
#                 cursor_pos += 1

#     # Handle keyboard inputs
#     if keyboard.is_pressed('q'):
#         break
#     if keyboard.is_pressed('c'):
#         sentence_text = []
#         cursor_pos = 0
#     if keyboard.is_pressed('backspace'):
#         if cursor_pos > 0:
#             sentence_text.pop(cursor_pos - 1)
#             cursor_pos -= 1
#         cv2.waitKey(200)  # prevent multiple deletions
#     if keyboard.is_pressed('left'):
#         if cursor_pos > 0:
#             cursor_pos -= 1
#         cv2.waitKey(200)  # prevent multiple jumps
#     if keyboard.is_pressed('right'):
#         if cursor_pos < len(sentence_text):
#             cursor_pos += 1
#         cv2.waitKey(200)
#     if keyboard.is_pressed('caps lock'):
#         if 0 <= cursor_pos < len(sentence_text):
#             sentence_text[cursor_pos] = sentence_text[cursor_pos].upper()
#         cv2.waitKey(200)

#     frame = draw_keyboard(frame)

#     # Draw the sentence with cursor
#     sentence_display = ''.join(sentence_text)
#     cursor_display = sentence_display[:cursor_pos] + '|' + sentence_display[cursor_pos:]
#     cv2.putText(frame, cursor_display, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow('Virtual Keyboard', frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()























