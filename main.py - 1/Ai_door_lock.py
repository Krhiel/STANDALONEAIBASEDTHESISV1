import os
import cv2
import pyttsx3
import serial
import time
from vosk import Model, KaldiRecognizer
import pyaudio
import json
import sys
from datetime import datetime
import serial.tools.list_ports  # Added for dynamic port detection

# Initialize text-to-speech engine
try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 180)
    print(f"{datetime.now()}: Text-to-speech engine initialized.")
    sys.stdout.flush()
except Exception as e:
    print(f"{datetime.now()}: Error initializing text-to-speech engine: {e}")
    sys.stdout.flush()
    exit()

CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "servo1_angle": 180,
        "servo2_angle": 180,
        "unlock_angle1": 45,
        "unlock_angle2": 45,
        "lock_delay": 7,
        "camera_index": 1,
        "com_port": "COM7"
    }

config = load_config()
print(f"{datetime.now()}: Configuration loaded: {config}")
sys.stdout.flush()

# Initialize camera with retry mechanism
def initialize_camera(camera_index):
    for attempt in range(3):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"{datetime.now()}: Camera opened successfully on index {camera_index}.")
            sys.stdout.flush()
            return cap
        print(f"{datetime.now()}: Failed to open camera on index {camera_index}, attempt {attempt + 1}/3.")
        sys.stdout.flush()
        cap.release()
        time.sleep(1)
    print(f"{datetime.now()}: Error: Could not open camera on index {camera_index} after 3 attempts.")
    sys.stdout.flush()
    exit()

cap = initialize_camera(config["camera_index"])

# Initialize serial port with dynamic detection
def detect_ports():
    """Detect available COM ports."""
    ports = serial.tools.list_ports.comports()
    available_ports = [port.device for port in ports]
    if not available_ports:
        print(f"{datetime.now()}: No COM ports detected. Defaulting to COM7.")
        sys.stdout.flush()
        return ["COM7"]
    return available_ports

def initialize_serial(config_port):
    """Initialize serial connection with dynamic port detection."""
    available_ports = detect_ports()
    print(f"{datetime.now()}: Available ports: {available_ports}")
    sys.stdout.flush()

    # First, try the port from config.json
    if config_port in available_ports:
        for attempt in range(3):
            try:
                ser = serial.Serial(config_port, 9600, timeout=1)
                time.sleep(2)
                print(f"{datetime.now()}: Serial port {config_port} opened successfully on attempt {attempt + 1}.")
                sys.stdout.flush()
                # Test communication with Arduino
                ser.write(b'arduino_ready\n')
                response = ser.readline().strip().decode(errors='replace')
                if response:
                    print(f"{datetime.now()}: Arduino responded: {response}")
                    sys.stdout.flush()
                    return ser
                ser.close()
            except Exception as e:
                print(f"{datetime.now()}: Failed to open {config_port}, attempt {attempt + 1}/3: {e}")
                sys.stdout.flush()
                time.sleep(1)
        print(f"{datetime.now()}: Could not connect to {config_port} after 3 attempts.")
        sys.stdout.flush()
    else:
        print(f"{datetime.now()}: Configured port {config_port} not available.")
        sys.stdout.flush()

    # If config port fails, try other available ports
    for port in available_ports:
        if port == config_port:
            continue  # Skip the already-tried port
        for attempt in range(3):
            try:
                ser = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)
                print(f"{datetime.now()}: Testing port {port}, attempt {attempt + 1}.")
                sys.stdout.flush()
                # Test communication with Arduino
                ser.write(b'arduino_ready\n')
                response = ser.readline().strip().decode(errors='replace')
                if response:
                    print(f"{datetime.now()}: Serial port {port} opened successfully. Arduino responded: {response}")
                    sys.stdout.flush()
                    # Update config with the working port
                    config["com_port"] = port
                    with open(CONFIG_FILE, 'w') as f:
                        json.dump(config, f)
                    print(f"{datetime.now()}: Updated config.json with new port: {port}")
                    sys.stdout.flush()
                    return ser
                ser.close()
            except Exception as e:
                print(f"{datetime.now()}: Failed to open {port}, attempt {attempt + 1}/3: {e}")
                sys.stdout.flush()
                time.sleep(1)

    print(f"{datetime.now()}: Error: Could not connect to any available port after trying {available_ports}.")
    sys.stdout.flush()
    cap.release()
    exit()

ser = initialize_serial(config["com_port"])

# Send configuration data to Arduino
try:
    ser.write(json.dumps(config).encode())
    time.sleep(3)
    print(f"{datetime.now()}: Sent configuration data to Arduino.")
    sys.stdout.flush()
except Exception as e:
    print(f"{datetime.now()}: Error sending configuration to Arduino: {e}")
    sys.stdout.flush()
    cap.release()
    ser.close()
    exit()

# Face recognition variables
door_unlocked = False
# Load Vosk model
vosk_model_path = "vosk-model-small-en-us-0.15"
if not os.path.exists(vosk_model_path):
    print(f"{datetime.now()}: Vosk model not found at {vosk_model_path}")
    sys.stdout.flush()
    exit()

model = Model(vosk_model_path)
recognizer_vosk = KaldiRecognizer(model, 16000)
print(f"{datetime.now()}: Vosk model loaded.")
sys.stdout.flush()

# Load Haar cascade with fallback
cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    print(f"{datetime.now()}: Using fallback Haar cascade path: {cascade_path}")
    sys.stdout.flush()
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print(f"{datetime.now()}: Error: Could not load Haar cascade file.")
    sys.stdout.flush()
    cap.release()
    ser.close()
    exit()
dataset_folder = "PersonFaceDataset"
known_faces = {}

# Ensure dataset folder exists
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)
    print(f"{datetime.now()}: Created dataset folder: {dataset_folder}")
    sys.stdout.flush()

# Function to recognize speech with timeout
def recognize_speech(timeout=5):
    print(f"{datetime.now()}: Listening for speech using Vosk...")
    sys.stdout.flush()
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        stream.start_stream()
        start_time = time.time()
        text_result = ""

        while time.time() - start_time < timeout:
            data = stream.read(4096, exception_on_overflow=False)
            if recognizer_vosk.AcceptWaveform(data):
                result = recognizer_vosk.Result()
                result_dict = json.loads(result)
                text_result = result_dict.get("text", "")
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

        if text_result:
            print(f"{datetime.now()}: Recognized speech: {text_result}")
            sys.stdout.flush()
            return text_result.lower()
        else:
            print(f"{datetime.now()}: No recognizable speech detected.")
            sys.stdout.flush()
            engine.say("Sorry, I didn't hear anything. Please try again.")
            engine.runAndWait()
            return None
    except Exception as e:
        print(f"{datetime.now()}: Vosk speech recognition error: {e}")
        sys.stdout.flush()
        engine.say("Sorry, there was an issue with speech recognition.")
        engine.runAndWait()
        return None


# Function to extract the actual name from the raw speech input
def extract_name_from_response(raw_name):
    if raw_name:
        raw_name = raw_name.lower()
        if "he is" in raw_name:
            persn_name = raw_name.split("he is")[-1].strip()
        elif "she is" in raw_name:
            person_name = raw_name.split("she is")[-1].strip()
        elif "i am" in raw_name:
            person_name = raw_name.split("i am")[-1].strip()
        elif "my name is" in raw_name:
            person_name = raw_name.split("my name is")[-1].strip()
        else:
            person_name = raw_name.strip()
        person_name = person_name.replace(" ", "_")
        print(f"{datetime.now()}: Extracted name: {person_name}")
        sys.stdout.flush()
        return person_name
    return None

def save_image_with_name(image, person_name):
    if person_name:
        person_dir = os.path.join(dataset_folder, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            print(f"{datetime.now()}: Created directory for {person_name}: {person_dir}")
            sys.stdout.flush()

        img_path = os.path.join(person_dir, f"{person_name}.jpg")
        cv2.imwrite(img_path, image)
        print(f"{datetime.now()}: Saved image of {person_name} at {img_path}")
        sys.stdout.flush()
        known_faces[person_name] = cv2.imread(img_path, 0)
        engine.say(f"Thank you, {person_name}. I will remember you.")
        engine.runAndWait()
    else:
        print(f"{datetime.now()}: Failed to recognize the name.")
        sys.stdout.flush()
        engine.say("Failed to recognize the name. Please try again.")
        engine.runAndWait()

# Load known faces
for person_dir in os.listdir(dataset_folder):
    person_path = os.path.join(dataset_folder, person_dir)
    if os.path.isdir(person_path):
        for img_file in os.listdir(person_path):
            if img_file.endswith('.jpg'):
                name = os.path.splitext(img_file)[0]
                known_faces[name] = cv2.imread(os.path.join(person_path, img_file), 0)
                print(f"{datetime.now()}: Loaded known face: {name}")
                sys.stdout.flush()

def send_servo_command(command):
    try:
        ser.write((command + '\n').encode())
        print(f"{datetime.now()}: Sent command: {command}")
        sys.stdout.flush()
    except Exception as e:
        print(f"{datetime.now()}: Error sending servo command: {e}")
        sys.stdout.flush()
        raise

try:
    print(f"{datetime.now()}: Starting main loop...")
    sys.stdout.flush()
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"{datetime.now()}: Failed to capture frame from camera.")
            sys.stdout.flush()
            break

        # Only process frames if the door is not in the process of unlocking/locking
        if not door_unlocked:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Check for faces in the current frame
            if len(faces) > 0:
                print(f"{datetime.now()}: Face detected: {len(faces)} face(s).")
                sys.stdout.flush()
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    detected_person = None

                    for name, known_face in known_faces.items():
                        result = cv2.matchTemplate(face_roi, known_face, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(result)
                        print(f"{datetime.now()}: Matching with {name}, Confidence: {max_val}")
                        sys.stdout.flush()

                        if max_val > 0.5:
                            detected_person = name
                            print(f"{datetime.now()}: Detected Person: {detected_person}")
                            sys.stdout.flush()
                            break

                    if detected_person and not door_unlocked:
                        print(f"{datetime.now()}: Speaking: {detected_person} is outside the door. I am opening the door.")
                        sys.stdout.flush()
                        engine.say(f"{detected_person} is outside the door. I am opening the door.")
                        engine.runAndWait()
                        try:
                            send_servo_command('O')
                            ser.flush()
                            print(f"{datetime.now()}: Unlocking successful.")
                            sys.stdout.flush()
                            
                            door_unlocked = True
                            time.sleep(config["lock_delay"])
                        
                            print(f"{datetime.now()}: Speaking: The door is now locked again.")
                            sys.stdout.flush()
                            engine.say("The door is now locked again.")
                            engine.runAndWait()
                            
                            send_servo_command('L')  # Explicit lock command
                            ser.flush()
                            print(f"{datetime.now()}: Locking successful.")
                            sys.stdout.flush()
                            door_unlocked = False
                            ser.flushInput()
                            ser.flushOutput()
                        except Exception as e:
                            print(f"{datetime.now()}: Error in sending unlock command: {e}")
                            sys.stdout.flush()
                            ser.flush()

                    elif not detected_person and not door_unlocked:
                        print(f"{datetime.now()}: Speaking: Admin, Someone is outside the door. Should I open the door?")
                        sys.stdout.flush()
                        engine.say("Admin, Someone is outside the door. Should I open the door?")
                        engine.runAndWait()

                        response = recognize_speech()
                        if response:
                            print(f"{datetime.now()}: You said: {response}")
                            sys.stdout.flush()
                            if 'yes' in response or 'open' in response or 'unlock' in response:
                                print(f"{datetime.now()}: Unlocking the door...")
                                sys.stdout.flush()
                                try:
                                    send_servo_command('O')
                                    ser.flush()
                                    print(f"{datetime.now()}: Unlocking successful.")
                                    sys.stdout.flush()
                                    print(f"{datetime.now()}: Speaking: The door is unlocked.")
                                    sys.stdout.flush()
                                    engine.say("The door is unlocked.")
                                    engine.runAndWait()
                                    door_unlocked = True
                                    time.sleep(config["lock_delay"])
                                    
                                    print(f"{datetime.now()}: Speaking: The door is locked again.")
                                    sys.stdout.flush()
                                    engine.say("The door is locked again.")
                                    engine.runAndWait()
                                    send_servo_command('L')  # Explicit lock command
                                    ser.flush()
                                    print(f"{datetime.now()}: Locking successful.")
                                    sys.stdout.flush()
                                    door_unlocked = False
                                    ser.flushInput()
                                    ser.flushOutput()
                                except Exception as e:
                                    print(f"{datetime.now()}: Error in sending unlock command: {e}")
                                    sys.stdout.flush()
                                    ser.flush()
                                
                            elif 'no' in response or 'lock' in response or 'dont' in response:
                                print(f"{datetime.now()}: Speaking: The main door will remain locked.")
                                sys.stdout.flush()
                                engine.say("The main door will remain locked.")
                                engine.runAndWait()
                            elif 'snap' in response:
                                print(f"{datetime.now()}: Speaking: What is the name, please?.")
                                sys.stdout.flush()
                                engine.say("What is the name, please?.")
                                engine.runAndWait()

                                raw_name = recognize_speech()
                                person_name = extract_name_from_response(raw_name)

                                if person_name:
                                    save_image_with_name(face_roi, person_name)
                                else:
                                    print(f"{datetime.now()}: Speaking: Failed to recognize your name. Please try again.")
                                    sys.stdout.flush()
                                    engine.say("Failed to recognize your name. Please try again.")
                                    engine.runAndWait()
                            else:
                                print(f"{datetime.now()}: Speaking: I didn't understand your response. The door will remain locked.")
                                sys.stdout.flush()
                                engine.say("I didn't understand your response. The door will remain locked.")
                                engine.runAndWait()

        # Display the camera feed
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"{datetime.now()}: User pressed 'q' to quit.")
            sys.stdout.flush()
            break

finally:
    print(f"{datetime.now()}: Cleaning up resources...")
    sys.stdout.flush()
    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()
        print(f"{datetime.now()}: Serial port closed.")
        sys.stdout.flush()
    print(f"{datetime.now()}: Ai_room_lock.py terminated.")
    sys.stdout.flush()