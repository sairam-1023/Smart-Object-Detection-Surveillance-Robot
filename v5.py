import cv2
import requests
import speech_recognition as sr
import time
import json
from gtts import gTTS
import pygame
import os
from ultralytics import YOLO

# ---------------- TELEGRAM ----------------
TOKEN = "TOKEN"
CHAT_ID = "CHAT_ID"

# ---------------- YOLO MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------------- SPEAK ----------------
def speak(text):
    print("Robot:", text)
    tts = gTTS(text=text, lang='en')
    tts.save("speech.mp3")

    pygame.mixer.init()
    pygame.mixer.music.load("speech.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.quit()
    os.remove("speech.mp3")

# ---------------- LISTEN ----------------
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("User:", text)
        return text
    except:
        speak("Sorry, I didn't understand. Please repeat.")
        return listen()

# ---------------- TELEGRAM SEND ----------------
def send_to_telegram(image_path, name, purpose):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

    caption = f"👤 Name: {name}\n📝 Purpose: {purpose}\n\nChoose an action:"

    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Admit", "callback_data": "admit"},
            {"text": "⏳ Wait", "callback_data": "wait"},
            {"text": "❌ Busy", "callback_data": "busy"}
        ]]
    }

    with open(image_path, 'rb') as img:
        requests.post(
            url,
            files={'photo': img},
            data={
                'chat_id': CHAT_ID,
                'caption': caption,
                'reply_markup': json.dumps(keyboard)
            }
        )

# ---------------- TELEGRAM RECEIVE ----------------
last_update_id = None

def answer_callback_query(callback_id):
    url = f"https://api.telegram.org/bot{TOKEN}/answerCallbackQuery"
    requests.post(url, data={"callback_query_id": callback_id})

def get_latest_message():
    global last_update_id

    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

    params = {}
    if last_update_id is not None:
        params["offset"] = last_update_id + 1  # only NEW messages

    response = requests.get(url, params=params).json()

    if not response.get("ok"):
        print("Telegram API Error:", response)
        return None

    updates = response.get("result", [])

    if not updates:
        return None

    for update in updates:
        last_update_id = update["update_id"]

        # ONLY process callback (button clicks)
        if "callback_query" in update:
            answer_callback_query(update["callback_query"]["id"])
            return update["callback_query"]["data"]

    return None



def wait_for_decision():
    global last_update_id

    speak("Waiting for owner response.")

    clear_old_updates()  # 🔥 flush old messages

    while True:
        msg = get_latest_message()
        print("Received:", msg)

        if msg == "admit":
            speak("You are allowed to enter.")
            return
        elif msg == "wait":
            speak("Please wait for some time.")
            return
        elif msg == "busy":
            speak("Sorry, the person is busy and cannot meet you.")
            return

        time.sleep(2)


def clear_old_updates():
    global last_update_id

    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    response = requests.get(url).json()

    if response.get("result"):
        last_update_id = response["result"][-1]["update_id"]

clear_old_updates()
# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

person_detected_frames = 0
REQUIRED_FRAMES = 8
triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    person_found = False
    face_found = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls == 0:  # PERSON
                person_found = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw PERSON box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(frame, "Person", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                # -------- FACE DETECTION INSIDE PERSON --------
                person_roi = frame[y1:y2, x1:x2]

                if person_roi.size == 0:
                    continue

                gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_roi, 1.3, 5)

                for (fx, fy, fw, fh) in faces:
                    face_found = True

                    cv2.rectangle(
                        frame,
                        (x1 + fx, y1 + fy),
                        (x1 + fx + fw, y1 + fy + fh),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Face",
                        (x1 + fx, y1 + fy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        2
                    )

    cv2.imshow("Camera", frame)

    # -------- STABILITY CHECK --------
    if person_found and face_found:
        person_detected_frames += 1
    else:
        person_detected_frames = 0
        triggered = False

    # -------- TRIGGER --------
    if person_detected_frames >= REQUIRED_FRAMES and not triggered:
        triggered = True

        image_path = "visitor.jpg"
        cv2.imwrite(image_path, frame)

        speak("Hello! What is your name?")
        name = listen()

        speak(f"Hi {name}, what is your purpose of visit?")
        purpose = listen()

        speak("Sending your details.")

        send_to_telegram(image_path, name, purpose)
        wait_for_decision()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()