"""
Smart surveillance System — Merged v2
==================================
- Known person  → greet by name (pyttsx3, offline TTS)
- Unknown person → ask name + purpose → notify owner via Telegram → log to SQLite
- Uses: face_recognition (known DB) + YOLO (person detection) + Haar (face crop)
"""

import cv2
import face_recognition
import pickle
import pyttsx3
import requests
import speech_recognition as sr
import time
import json
from gtts import gTTS
import pygame
import os
from ultralytics import YOLO
import spacy
from difflib import get_close_matches
import sqlite3

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════

# Telegram bot token
TOKEN = "8310932356:AAGfJ3SekJRxTI4jAXU_ppuKa9b7cvEb64Y"

# Owner chat IDs — add/remove as needed
USER_CHAT_IDS = {
    "tarun":  "6663176931",
    "sairam": "8171470054",
    "ganesh": "5693255882",
}

# How many consecutive frames with person+face before triggering
REQUIRED_FRAMES = 8

# Face recognition tolerance (lower = stricter)
FACE_TOLERANCE = 0.5

# Cooldown for unknown visitors — seconds (20 min = 1200)
UNKNOWN_COOLDOWN = 1200

# ════════════════════════════════════════════════════════════
#  LOAD KNOWN FACES DB
# ════════════════════════════════════════════════════════════

try:
    with open("faces_db.pkl", "rb") as f:
        known_encodings, known_names = pickle.load(f)
    print(f"✅ Loaded {len(known_names)} known faces: {known_names}")
except FileNotFoundError:
    known_encodings, known_names = [], []
    print("⚠️  faces_db.pkl not found — all visitors will be treated as unknown.")

# ════════════════════════════════════════════════════════════
#  MODELS
# ════════════════════════════════════════════════════════════

yolo_model   = YOLO("yolov8n.pt")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
nlp = spacy.load("en_core_web_sm")

# ════════════════════════════════════════════════════════════
#  SQLITE — VISITOR LOG
# ════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect("visitors.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    DEFAULT (datetime('now','localtime')),
            name       TEXT,
            purpose    TEXT,
            person     TEXT,
            decision   TEXT,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_log(name, purpose, person, decision, image_path):
    try:
        conn = sqlite3.connect("visitors.db")
        conn.execute(
            "INSERT INTO visitors (name, purpose, person, decision, image_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, purpose, person, decision, image_path)
        )
        conn.commit()
        conn.close()
        print("✅ Visitor log saved to DB")
    except Exception as e:
        print("DB Error:", e)

init_db()

# ════════════════════════════════════════════════════════════
#  SPEECH — TTS
#  Known-person greetings use pyttsx3 (fast, offline).
#  Unknown-visitor flow uses gTTS (better quality).
# ════════════════════════════════════════════════════════════

_tts_engine = pyttsx3.init()

def speak_offline(text):
    """Fast offline TTS — used for known-person greetings."""
    print("Robot:", text)
    _tts_engine.say(text)
    _tts_engine.runAndWait()

def speak(text):
    """Online TTS via gTTS — used during unknown-visitor flow."""
    print("Robot:", text)
    try:
        tts = gTTS(text=text, lang="en")
        tts.save("speech.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("speech.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.quit()
        os.remove("speech.mp3")
    except Exception:
        # Fallback to offline if gTTS fails (no internet, etc.)
        speak_offline(text)

# ════════════════════════════════════════════════════════════
#  SPEECH — STT
# ════════════════════════════════════════════════════════════

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
    except Exception:
        speak("Sorry, I didn't catch that. Please repeat.")
        return listen()

# ════════════════════════════════════════════════════════════
#  NLP HELPERS
# ════════════════════════════════════════════════════════════

def extract_person_nlp(text):
    """Extract PERSON entity from purpose text."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.lower()
    return None

def match_person(name):
    """Fuzzy-match a name to registered owners."""
    matches = get_close_matches(name, USER_CHAT_IDS.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

# ════════════════════════════════════════════════════════════
#  TELEGRAM
# ════════════════════════════════════════════════════════════

last_update_id = None

def clear_old_updates():
    global last_update_id
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    resp = requests.get(url).json()
    if resp.get("result"):
        last_update_id = resp["result"][-1]["update_id"]

def answer_callback_query(callback_id):
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/answerCallbackQuery",
        data={"callback_query_id": callback_id}
    )

def get_latest_message():
    global last_update_id
    params = {}
    if last_update_id is not None:
        params["offset"] = last_update_id + 1
    resp = requests.get(
        f"https://api.telegram.org/bot{TOKEN}/getUpdates",
        params=params
    ).json()
    if not resp.get("ok"):
        return None
    for update in resp.get("result", []):
        last_update_id = update["update_id"]
        if "callback_query" in update:
            answer_callback_query(update["callback_query"]["id"])
            return update["callback_query"]["data"]
    return None

def send_to_telegram(image_path, name, purpose, chat_id):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    caption = f"👤 Name: {name}\n📝 Purpose: {purpose}\n\nChoose an action:"
    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Admit", "callback_data": "admit"},
            {"text": "⏳ Wait",  "callback_data": "wait"},
            {"text": "❌ Busy",  "callback_data": "busy"},
        ]]
    }
    with open(image_path, "rb") as img:
        requests.post(
            url,
            files={"photo": img},
            data={
                "chat_id":      chat_id,
                "caption":      caption,
                "reply_markup": json.dumps(keyboard),
            }
        )

def wait_for_decision(timeout=60):
    speak("Waiting for owner response.")
    clear_old_updates()
    start = time.time()
    while True:
        if time.time() - start > timeout:
            speak("No response received. Please try again later.")
            return "timeout"
        msg = get_latest_message()
        print("Received:", msg)
        if msg == "admit":
            speak("You are allowed to enter.")
            return "admit"
        elif msg == "wait":
            speak("Please wait for some time.")
            return "wait"
        elif msg == "busy":
            speak("Sorry, the person is busy right now.")
            return "busy"
        time.sleep(2)

# ════════════════════════════════════════════════════════════
#  FACE RECOGNITION HELPER
# ════════════════════════════════════════════════════════════

def identify_face(rgb_frame, location):
    """
    Returns (name, is_known, encoding).
    name = matched name string, or "Unknown".
    encoding = raw face encoding (used for unknown visitor fingerprinting).
    """
    enc = face_recognition.face_encodings(rgb_frame, [location])
    if not enc:
        return "Unknown", False, None

    if known_encodings:
        matches = face_recognition.compare_faces(
            known_encodings, enc[0], tolerance=FACE_TOLERANCE
        )
        if True in matches:
            name = known_names[matches.index(True)]
            return name, True, enc[0]

    return "Unknown", False, enc[0]

def get_unknown_fingerprint(encoding):
    """
    Match a new unknown encoding against previously seen unknown visitors.
    Returns index into unknown_cooldowns list, or None if new face.
    """
    for i, (stored_enc, _) in enumerate(unknown_cooldowns):
        results = face_recognition.compare_faces(
            [stored_enc], encoding, tolerance=FACE_TOLERANCE
        )
        if results[0]:
            return i
    return None

# ════════════════════════════════════════════════════════════
#  WAKE WORD LISTENER
# ════════════════════════════════════════════════════════════

WAKE_WORDS = ["hi robot", "hey robot", "hi robort", "hey robort"]  # variants for mis-recognition

def listen_for_wake_word():
    """
    Lightweight mic listen — returns True if wake word detected.
    Short timeout so it loops quickly without blocking.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            # phrase_time_limit keeps it non-blocking (3 sec max per attempt)
            audio = recognizer.listen(source, phrase_time_limit=3)
            text  = recognizer.recognize_google(audio).lower()
            print(f"Heard: '{text}'")
            return any(w in text for w in WAKE_WORDS)
        except (sr.UnknownValueError, sr.WaitTimeoutError):
            return False
        except sr.RequestError:
            print("⚠️  Speech API unavailable.")
            return False

# ════════════════════════════════════════════════════════════
#  FACE SCAN — called only after wake word
# ════════════════════════════════════════════════════════════

def scan_face():
    """
    Opens camera, tries to detect + identify a face within SCAN_TIMEOUT seconds.
    Returns (detected_name, is_known, face_encoding, frame) or None if no face found.
    """
    SCAN_TIMEOUT = 10   # seconds to wait for a face after wake word
    cap = cv2.VideoCapture(0)
    start = time.time()

    print("📷 Camera active — scanning for face...")

    while time.time() - start < SCAN_TIMEOUT:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yolo_results = yolo_model(frame, verbose=False)

        for r in yolo_results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue

                gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                faces    = face_cascade.detectMultiScale(gray_roi, 1.3, 5)

                for (fx, fy, fw, fh) in faces:
                    abs_top    = y1 + fy
                    abs_right  = x1 + fx + fw
                    abs_bottom = y1 + fy + fh
                    abs_left   = x1 + fx
                    face_location = (abs_top, abs_right, abs_bottom, abs_left)

                    name, is_known, face_enc = identify_face(rgb_frame, face_location)

                    # Draw result on frame
                    color = (0, 200, 100) if is_known else (0, 0, 255)
                    cv2.rectangle(frame, (abs_left, abs_top), (abs_right, abs_bottom), color, 2)
                    cv2.putText(frame, name, (abs_left, abs_top - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.imshow("Smart surveillance system", frame)
                    cv2.waitKey(1)

                    cap.release()
                    return name, is_known, face_enc, frame

        # Show live feed while scanning
        cv2.putText(frame, "Scanning...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Smart surveillance system", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            return None

    cap.release()
    print("⚠️  No face detected within timeout.")
    return None

# ════════════════════════════════════════════════════════════
#  UNKNOWN VISITOR FLOW
# ════════════════════════════════════════════════════════════

def handle_unknown_visitor(face_enc, frame, unknown_cooldowns):
    """Full unknown visitor flow with 20-min cooldown check."""
    now = time.time()

    # Check cooldown
    cooldown_idx = get_unknown_fingerprint(face_enc) if face_enc is not None else None

    if cooldown_idx is not None:
        _, last_time = unknown_cooldowns[cooldown_idx]
        remaining = UNKNOWN_COOLDOWN - (now - last_time)
        if remaining > 0:
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            print(f"⏳ Cooldown active: {mins}m {secs}s remaining. Ignoring.")
            speak("You have already been registered. Please wait.")
            return unknown_cooldowns
        else:
            unknown_cooldowns[cooldown_idx] = (face_enc, now)
            print("✅ Cooldown expired. Re-processing.")
    else:
        if face_enc is not None:
            unknown_cooldowns.append((face_enc, now))
        print("🆕 New unknown visitor.")

    # Save snapshot
    image_path = f"visitor_{int(now)}.jpg"
    cv2.imwrite(image_path, frame)

    # Conversation flow
    speak("Hello! I don't recognise you. What is your name?")
    name = listen()

    speak(f"Hi {name}, what is your purpose of visit?")
    purpose = listen()

    speak("Let me find out whom you want to meet.")
    person = extract_person_nlp(purpose)

    if not person:
        speak("I couldn't catch the name. Who would you like to meet?")
        person = listen().lower()

    matched = match_person(person)

    if matched:
        chat_id = USER_CHAT_IDS[matched]
        speak(f"Sending your request to {matched}. Please wait.")
        send_to_telegram(image_path, name, purpose, chat_id)
        decision = wait_for_decision()
    else:
        speak("Sorry, I could not find that person in our system.")
        decision = "person_not_found"
        matched  = person

    save_log(
        name       = name,
        purpose    = purpose,
        person     = matched or "unknown",
        decision   = decision,
        image_path = image_path,
    )

    return unknown_cooldowns

# ════════════════════════════════════════════════════════════
#  MAIN — Wake-word driven
# ════════════════════════════════════════════════════════════

clear_old_updates()

greeted_known    = set()
unknown_cooldowns = []

print("🚀 Smart surveillance system ready. Say 'Hi Robot' to activate.")
print("   (Press Ctrl+C to quit)\n")

try:
    while True:

        # ── Step 1: Wait for wake word ─────────────────────
        print("👂 Listening for wake word...")
        woken = listen_for_wake_word()

        if not woken:
            continue   # keep listening

        print("✅ Wake word detected!")
        speak_offline("Hello! Let me see who you are.")

        # ── Step 2: Scan face ──────────────────────────────
        result = scan_face()

        if result is None:
            speak("I couldn't see a face. Please try again.")
            continue

        detected_name, is_known, face_enc, frame = result

        # ── Step 3: Known or Unknown? ──────────────────────
        if is_known:
            if detected_name not in greeted_known:
                speak(f"Hello {detected_name}, welcome!")
                greeted_known.add(detected_name)
            else:
                speak(f"Welcome back {detected_name}!")

        else:
            unknown_cooldowns = handle_unknown_visitor(face_enc, frame, unknown_cooldowns)

        # Small pause before listening for wake word again
        time.sleep(2)

except KeyboardInterrupt:
    print("\n👋 Shutting down Smart surveillance system.")
    cv2.destroyAllWindows()
