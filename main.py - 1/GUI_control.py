import customtkinter as ctk
import cv2
import os
import pyttsx3
import serial
import time
import threading
from tkinter import filedialog, messagebox
from subprocess import Popen, PIPE
import json
import serial.tools.list_ports
from PIL import Image, ImageTk
import numpy as np
import sys

# Initialize text-to-speech engine
try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 180)
    print("Text-to-speech engine initialized")
except Exception as e:
    print(f"Error initializing text-to-speech engine: {e}")

CONFIG_FILE = "config.json"

def load_config():
    default_config = {
        "servo1_angle": 180,
        "servo2_angle": 180,
        "unlock_angle1": 45,
        "unlock_angle2": 45,
        "lock_delay": 7,
        "com_port": "COM7",
        "camera_index": 1
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            default_config.update(config)
    return default_config

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

class SmartDoorApp(ctk.CTk):
    def __init__(self):
        print("Starting GUI initialization...")
        try:
            super().__init__()
            self.title("MAIN DOOR CONTROL")
            self.geometry("1200x800")
            self.configure(fg_color="#16a34a")
            print("GUI window created")
        except Exception as e:
            print(f"Error creating GUI window: {e}")
            raise

        self.config = load_config()
        self.servo1_angle = self.config["servo1_angle"]
        self.servo2_angle = self.config["servo2_angle"]
        self.unlock_angle1 = self.config["unlock_angle1"]
        self.unlock_angle2 = self.config["unlock_angle2"]
        self.lock_delay = self.config["lock_delay"]
        self.com_port = self.config["com_port"]
        self.camera_index = self.config["camera_index"]

        # Configure grid layout with proper weights
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        # Modern theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Feedback Label
        self.feedback_label = ctk.CTkLabel(self, text="System ready", font=("Helvetica", 14, "bold"), text_color="#ffffff")
        self.feedback_label.grid(row=3, column=0, columnspan=2, pady=(10, 20), sticky="ew")

        # Initialize serial connection
        self.ser = None
        self.connect_serial()

        # Device Selection Frame
        self.device_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#14532d")
        self.device_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
        self.device_frame.grid_columnconfigure(0, weight=1)
        self.device_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.device_label = ctk.CTkLabel(self.device_frame, text="Device Settings", font=("Helvetica", 18, "bold"), text_color="#ffffff")
        self.device_label.pack(pady=5)

        self.camera_label = ctk.CTkLabel(self.device_frame, text="Select Camera", font=("Helvetica", 14))
        self.camera_label.pack(pady=2)
        self.camera_options = self.detect_cameras()
        self.camera_var = ctk.StringVar(value=str(self.camera_index))
        self.camera_menu = ctk.CTkOptionMenu(self.device_frame, values=self.camera_options, variable=self.camera_var, command=self.update_camera, fg_color="#166534", button_color="#15803d")
        self.camera_menu.pack(pady=2)

        self.port_label = ctk.CTkLabel(self.device_frame, text="Select COM Port", font=("Helvetica", 14))
        self.port_label.pack(pady=2)
        self.port_options = self.detect_ports()
        self.port_var = ctk.StringVar(value=self.com_port)
        self.port_menu = ctk.CTkOptionMenu(self.device_frame, values=self.port_options, variable=self.port_var, command=self.update_port, fg_color="#166534", button_color="#15803d")
        self.port_menu.pack(pady=2)

        # Servo Angle Adjustment Frame
        self.servo_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#14532d")
        self.servo_frame.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")
        self.servo_frame.grid_columnconfigure(0, weight=1)
        self.servo_frame.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

        self.servo_label = ctk.CTkLabel(self.servo_frame, text="Servo Tuning", font=("Helvetica", 18, "bold"), text_color="#ffffff")
        self.servo_label.pack(pady=5)

        self.servo_slider1 = ctk.CTkSlider(self.servo_frame, from_=0, to=180, command=self.update_servo1, fg_color="#166534")
        self.servo_slider1.set(self.servo1_angle)
        self.servo_slider1.pack(pady=2, padx=20)
        self.servo_value1_label = ctk.CTkLabel(self.servo_frame, text=f"Lock Angle 1: {self.servo1_angle}°", font=("Helvetica", 12))
        self.servo_value1_label.pack()

        self.servo_slider2 = ctk.CTkSlider(self.servo_frame, from_=0, to=180, command=self.update_servo2, fg_color="#166534")
        self.servo_slider2.set(self.servo2_angle)
        self.servo_slider2.pack(pady=2, padx=20)
        self.servo_value2_label = ctk.CTkLabel(self.servo_frame, text=f"Lock Angle 2: {self.servo2_angle}°", font=("Helvetica", 12))
        self.servo_value2_label.pack()

        self.unlock_slider1 = ctk.CTkSlider(self.servo_frame, from_=0, to=180, command=self.update_unlock_angle1, fg_color="#166534")
        self.unlock_slider1.set(self.unlock_angle1)
        self.unlock_slider1.pack(pady=2, padx=20)
        self.unlock_value1_label = ctk.CTkLabel(self.servo_frame, text=f"Unlock Angle 1: {self.unlock_angle1}°", font=("Helvetica", 12))
        self.unlock_value1_label.pack()

        self.unlock_slider2 = ctk.CTkSlider(self.servo_frame, from_=0, to=180, command=self.update_unlock_angle2, fg_color="#166534")
        self.unlock_slider2.set(self.unlock_angle2)
        self.unlock_slider2.pack(pady=2, padx=20)
        self.unlock_value2_label = ctk.CTkLabel(self.servo_frame, text=f"Unlock Angle 2: {self.unlock_angle2}°", font=("Helvetica", 12))
        self.unlock_value2_label.pack()

        self.test_servo_button = ctk.CTkButton(self.servo_frame, text="Test Servo", command=self.test_servo, fg_color="#166534", hover_color="#15803d")
        self.test_servo_button.pack(pady=5)

        # Delay and Control Frame
        self.control_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#14532d")
        self.control_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        self.control_frame.grid_columnconfigure((0, 1), weight=1)
        self.control_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.control_label = ctk.CTkLabel(self.control_frame, text="Door Control & Delays", font=("Helvetica", 18, "bold"), text_color="#ffffff")
        self.control_label.pack(pady=5)

        self.lock_delay_label = ctk.CTkLabel(self.control_frame, text="Lock Delay (s)", font=("Helvetica", 14))
        self.lock_delay_label.pack(pady=2)
        self.lock_delay_entry = ctk.CTkEntry(self.control_frame, fg_color="#3a3f44", border_color="#166534")
        self.lock_delay_entry.insert(0, str(self.lock_delay))
        self.lock_delay_entry.pack(pady=2, padx=20)

        self.update_delay_button = ctk.CTkButton(self.control_frame, text="Update Delay", command=self.update_delay, fg_color="#166534", hover_color="#15803d")
        self.update_delay_button.pack(pady=5)

        self.button_frame = ctk.CTkFrame(self.control_frame, fg_color="#14532d")
        self.button_frame.pack(pady=5)
        self.unlock_button = ctk.CTkButton(self.button_frame, text="Unlock Door", command=self.unlock_door, fg_color="#166534", hover_color="#15803d")
        self.unlock_button.pack(side="left", padx=10)
        self.lock_button = ctk.CTkButton(self.button_frame, text="Lock Door", command=self.lock_door, fg_color="#166534", hover_color="#15803d")
        self.lock_button.pack(side="left", padx=10)

        # Face Recognition Frame
        self.recognition_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#14532d")
        self.recognition_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.recognition_frame.grid_columnconfigure(0, weight=1)
        self.recognition_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.recognition_label = ctk.CTkLabel(self.recognition_frame, text="Add New User", font=("Helvetica", 18, "bold"), text_color="#ffffff")
        self.recognition_label.pack(pady=5)

        self.capture_button = ctk.CTkButton(self.recognition_frame, text="Capture Face", command=self.capture_face, fg_color="#166534", hover_color="#15803d")
        self.capture_button.pack(pady=5)
        self.name_entry = ctk.CTkEntry(self.recognition_frame, placeholder_text="Enter Name", fg_color="#3a3f44", border_color="#166534")
        self.name_entry.pack(pady=5)
        self.save_button = ctk.CTkButton(self.recognition_frame, text="Save to Dataset", command=self.save_to_dataset, fg_color="#166534", hover_color="#15803d")
        self.save_button.pack(pady=5)

        # Start System Frame
        self.start_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#14532d")
        self.start_frame.grid(row=2, column=1, padx=20, pady=10, sticky="nsew")
        self.start_frame.grid_columnconfigure(0, weight=1)
        self.start_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.start_label = ctk.CTkLabel(self.start_frame, text="System Control", font=("Helvetica", 18, "bold"), text_color="#ffffff")
        self.start_label.pack(pady=5)
        self.initialize_button = ctk.CTkButton(self.start_frame, text="Initialize System", command=self.initialize_system, fg_color="#166534", hover_color="#15803d")
        self.initialize_button.pack(pady=5)
        self.start_system_button = ctk.CTkButton(self.start_frame, text="Start System", command=self.start_system, fg_color="#166534", hover_color="#15803d")
        self.start_system_button.pack(pady=5)

        # Feedback Frame (Enhanced for Start System)
        self.feedback_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#14532d")
        self.feedback_frame.grid(row=0, column=0, columnspan=2, rowspan=3, padx=20, pady=20, sticky="nsew")
        self.feedback_frame.grid_columnconfigure((0, 1), weight=1)
        self.feedback_frame.grid_rowconfigure((0, 1), weight=1)
        self.feedback_frame.grid_remove()

        self.process_container = ctk.CTkFrame(self.feedback_frame, corner_radius=10, fg_color="#3a3f44")
        self.process_container.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
        self.process_label = ctk.CTkLabel(self.process_container, text="Process Log", font=("Helvetica", 16, "bold"), text_color="#ffffff")
        self.process_label.pack(pady=5)
        self.process_text = ctk.CTkTextbox(self.process_container, height=300, width=500, font=("Helvetica", 12))
        self.process_text.pack(pady=5, padx=10, fill="both", expand=True)

        

        self.feedback_button_frame = ctk.CTkFrame(self.feedback_frame, fg_color="#14532d")
        self.feedback_button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        self.stop_system_button = ctk.CTkButton(self.feedback_button_frame, text="Stop System", command=self.stop_system, fg_color="#166534", hover_color="#15803d")
        self.stop_system_button.pack(side="left", padx=10)
        self.start_system_again_button = ctk.CTkButton(self.feedback_button_frame, text="Start System", command=self.start_system_again, fg_color="#166534", hover_color="#15803d")
        self.start_system_again_button.pack(side="left", padx=10)
        self.start_system_again_button.pack_forget()  # Initially hidden
        self.exit_program_button = ctk.CTkButton(self.feedback_button_frame, text="Exit Program", command=self.exit_program, fg_color="#ff4d4d", hover_color="#cc0000")
        self.exit_program_button.pack(side="left", padx=10)

        # Variables for camera feed and AI process
        self.cap = None
        self.ai_process = None
        self.running = False
        self.ai_thread = None

        print("GUI initialization completed. Waiting for user interaction...")

    def detect_cameras(self):
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(str(i))
                cap.release()
        return available_cameras if available_cameras else ["0"]

    def detect_ports(self):
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports] if ports else ["COM7"]

    def connect_serial(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        try:
            self.ser = serial.Serial(self.com_port, 9600, timeout=1)
            time.sleep(2)
            self.feedback_label.configure(text=f"Connected to {self.com_port}")
            print(f"Serial connection established on {self.com_port}")
        except Exception as e:
            self.feedback_label.configure(text=f"Serial Error: {e}")
            print(f"Serial connection failed: {e}")

    def disconnect_serial(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.feedback_label.configure(text=f"Disconnected from {self.com_port}")
            self.ser = None
            print(f"Serial connection closed on {self.com_port}")

    def update_camera(self, value):
        self.camera_index = int(value)
        self.save_config()
        self.feedback_label.configure(text=f"Camera set to index {self.camera_index}")

    def update_port(self, value):
        self.com_port = value
        self.connect_serial()
        self.save_config()

    def update_servo1(self, value):
        self.servo1_angle = int(value)
        self.servo_value1_label.configure(text=f"Lock Angle 1: {self.servo1_angle}°")
        self.save_config()

    def update_servo2(self, value):
        self.servo2_angle = int(value)
        self.servo_value2_label.configure(text=f"Lock Angle 2: {self.servo2_angle}°")
        self.save_config()

    def update_unlock_angle1(self, value):
        self.unlock_angle1 = int(value)
        self.unlock_value1_label.configure(text=f"Unlock Angle 1: {self.unlock_angle1}°")
        self.save_config()

    def update_unlock_angle2(self, value):
        self.unlock_angle2 = int(value)
        self.unlock_value2_label.configure(text=f"Unlock Angle 2: {self.unlock_angle2}°")
        self.save_config()

    def test_servo(self):
        try:
            command = f"A{self.servo1_angle},{self.servo2_angle}\n"
            self.ser.write(command.encode())
            response = self.ser.readline().strip().decode(errors='replace')
            if response == "OK":
                self.feedback_label.configure(text="Servo angles updated successfully.")
            else:
                self.feedback_label.configure(text=f"Servo update failed: {response}")
        except Exception as e:
            self.feedback_label.configure(text=f"Error: {e}")

    def capture_face(self):
        try:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame from camera.")
                cap.release()
                return

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Load Haar cascade for face detection
            cascade_path = 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                messagebox.showerror("Error", "Could not load Haar cascade file.")
                cap.release()
                return

            # Detect faces in the grayscale image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if len(faces) == 0:
                messagebox.showerror("Error", "No face detected in the captured frame.")
                cap.release()
                return

            # Use the first detected face
            (x, y, w, h) = faces[0]
            # Extract the face ROI in grayscale
            face_roi = gray[y:y+h, x:x+w]
            self.captured_image = face_roi  # Store grayscale face ROI
            # Display the captured face (optional, for user feedback)
            cv2.imshow("Captured Face", face_roi)
            messagebox.showinfo("Info", "Face captured successfully.")
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {e}")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def train_face(self, image_path, person_name, dataset_folder="PersonFaceDataset"):
        """
        Process the captured image and save the trained data as a matrix in a YAML file.
        Args:
            image_path (str): Path to the captured image.
            person_name (str): Name of the person.
            dataset_folder (str): Folder where the data is stored.
        Returns:
            bool: True if training and saving were successful, False otherwise.
        """
        try:
            # Read the captured image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Could not read image at {image_path}")
                return False

            # Normalize the image to improve matching accuracy
            image = cv2.equalizeHist(image)  # Histogram equalization for better contrast
            image = cv2.resize(image, (100, 100))  # Resize to a standard size for consistency
            # Convert the image to a numpy array (matrix)
            image_matrix = np.array(image, dtype=np.uint8)
            # Load existing YAML data (if any)
            yaml_path = os.path.join(dataset_folder, "faces.yaml")
            fs = cv2.FileStorage(yaml_path, cv2.FileStorage_READ)

            # If the file doesn't exist or is empty, create a new one
            if not fs.isOpened():
                fs.release()
                fs = cv2.FileStorage(yaml_path, cv2.FileStorage_WRITE)
            else:
                # Read existing data to append
                fs.release()
                fs = cv2.FileStorage(yaml_path, cv2.FileStorage_APPEND)

            # Write the matrix to the YAML file under the person's name
            fs.write(f"face_{person_name}", image_matrix)
            fs.release()

            print(f"Trained data for {person_name} saved in {yaml_path}")
            return True
        except Exception as e:
            print(f"Error during training: {e}")
            return False

    def save_to_dataset(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name.")
            return
        if not hasattr(self, 'captured_image') or self.captured_image is None:
            messagebox.showerror("Error", "No face captured.")
            return

        try:
            # Replace spaces with underscores to match Ai_room_lock.py naming convention
            person_name = name.lower().replace(" ", "_")
            folder = "PersonFaceDataset"
            person_dir = os.path.join(folder, person_name)
            os.makedirs(person_dir, exist_ok=True)
            # Save the grayscale face ROI
            img_path = os.path.join(person_dir, f"{person_name}.jpg")
            cv2.imwrite(img_path, self.captured_image)
            
            # Train the face data and save it to YAML
            if self.train_face(img_path, person_name):
                messagebox.showinfo("Info", f"Saved and trained {person_name}'s face to dataset.")
            else:
                messagebox.showwarning("Warning", f"Saved {person_name}'s face but training failed.")
            
            # Reset captured image to prevent accidental reuse
            self.captured_image = None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save or train image: {e}")
            
    def update_delay(self):
        try:
            new_lock_delay = int(self.lock_delay_entry.get())
            if new_lock_delay < 0:
                raise ValueError("Lock delay cannot be negative.")
            self.lock_delay = new_lock_delay
            self.save_config()
            self.feedback_label.configure(text=f"Lock delay updated to {self.lock_delay} seconds.")
        except ValueError as e:
            self.feedback_label.configure(text=f"Error: Invalid lock delay value - {e}")

    def unlock_door(self):
        try:
            command = f"A{self.unlock_angle1},{self.unlock_angle2}\n"
            self.ser.write(command.encode())
            response = self.ser.readline().strip().decode(errors='replace')
            if response == "OK":
                self.feedback_label.configure(text="Door unlocked.")
                if self.feedback_frame.winfo_viewable():
                    self.process_text.insert("end", "Door unlocked.\n")
                engine.say("The door is now unlocked.")
                engine.runAndWait()
                if self.feedback_frame.winfo_viewable():
                    self.process_text.insert("end", "Speaking: The door is now unlocked.\n")
                time.sleep(self.lock_delay)
                self.lock_door(auto=True)
            else:
                self.feedback_label.configure(text="Unlock failed.")
                if self.feedback_frame.winfo_viewable():
                    self.process_text.insert("end", f"Unlock failed: {response}\n")
        except Exception as e:
            self.feedback_label.configure(text=f"Error: {e}")
            if self.feedback_frame.winfo_viewable():
                self.process_text.insert("end", f"Error: {e}\n")

    def lock_door(self, auto=False):
        try:
            command = f"A{self.servo1_angle},{self.servo2_angle}\n"
            self.ser.write(command.encode())
            response = self.ser.readline().strip().decode(errors='replace')
            if response == "OK":
                self.feedback_label.configure(text="Door locked automatically." if auto else "Door locked manually.")
                if self.feedback_frame.winfo_viewable():
                    self.process_text.insert("end", "Door locked " + ("automatically.\n" if auto else "manually.\n"))
                if not auto:
                    engine.say("The door is now locked.")
                    engine.runAndWait()
                    if self.feedback_frame.winfo_viewable():
                        self.process_text.insert("end", "Speaking: The door is now locked.\n")
            else:
                self.feedback_label.configure(text="Lock failed.")
                if self.feedback_frame.winfo_viewable():
                    self.process_text.insert("end", f"Lock failed: {response}\n")
        except Exception as e:
            self.feedback_label.configure(text=f"Error: {e}")
            if self.feedback_frame.winfo_viewable():
                self.process_text.insert("end", f"Error: {e}\n")

    def initialize_system(self):
        print("Initialize System button clicked. Starting initialization...")
        self.initialize_button.configure(state="disabled")
        self.feedback_label.configure(text="Initializing system...")

        def run_initialization():
            try:
                if not self.ser or not self.ser.is_open:
                    self.feedback_label.configure(text="Error: Serial port not connected")
                    print("Error: Serial port not connected")
                    return

                self.ser.write(b'arduino_ready\n')
                print("Sent 'arduino_ready' command to Arduino")
                response = self.ser.readline().strip().decode(errors='replace')
                print(f"Arduino response: '{response}'")

                if response == "OK":
                    self.feedback_label.configure(text="System initialized.")
                    if self.feedback_frame.winfo_viewable():
                        self.process_text.insert("end", "System initialized.\n")
                    def speak():
                        engine.say("The system is initialized.")
                        engine.runAndWait()
                        if self.feedback_frame.winfo_viewable():
                            self.process_text.insert("end", "Speaking: The system is initialized.\n")
                    threading.Thread(target=speak, daemon=True).start()
                else:
                    self.feedback_label.configure(text=f"Initialization failed: {response}")
                    if self.feedback_frame.winfo_viewable():
                        self.process_text.insert("end", f"Initialization failed: {response}\n")
            except Exception as e:
                self.feedback_label.configure(text=f"Error: {e}")
                if self.feedback_frame.winfo_viewable():
                    self.process_text.insert("end", f"Error: {e}\n")
                print(f"Error during initialization: {e}")
            finally:
                self.initialize_button.configure(state="normal")

        threading.Thread(target=run_initialization, daemon=True).start()

    def start_system(self):
        print("Start System button clicked. Starting AI Door Lock...")
        self.disconnect_serial()
        self.device_frame.grid_remove()
        self.servo_frame.grid_remove()
        self.control_frame.grid_remove()
        self.recognition_frame.grid_remove()
        self.start_frame.grid_remove()
        self.feedback_frame.grid()
        self.stop_system_button.pack(side="left", padx=10)
        self.start_system_again_button.pack_forget()
        self.exit_program_button.pack(side="left", padx=10)
        self.running = True
        self.ai_process = None  # Reset ai_process
        self.ai_thread = threading.Thread(target=self.run_ai_door_lock, daemon=True)
        self.ai_thread.start()

    def start_system_again(self):
        print("Start System Again button clicked. Restarting AI Door Lock...")
        self.process_text.delete("1.0", "end")
        self.start_system_again_button.pack_forget()
        self.stop_system_button.pack(side="left", padx=10)
        self.exit_program_button.pack(side="left", padx=10)
        self.running = True
        self.ai_process = None  # Reset ai_process
        self.ai_thread = threading.Thread(target=self.run_ai_door_lock, daemon=True)
        self.ai_thread.start()

    def run_ai_door_lock(self):
        print("Running AI Door Lock system...")
        self.process_text.insert("end", "Starting AI Door Lock system...\n")
        try:
            self.ai_process = Popen(['python', 'Ai_room_lock.py'], stdout=PIPE, stderr=PIPE, text=True)
            print("AI Door Lock subprocess started.")
            while self.running and self.ai_process.poll() is None:
                output = self.ai_process.stdout.readline()
                if output:
                    self.process_text.insert("end", output)
                    self.process_text.see("end")
            if self.ai_process.poll() is not None:
                stderr_output = self.ai_process.stderr.read()
                if stderr_output:
                    self.process_text.insert("end", f"Error: {stderr_output}\n")
                self.process_text.insert("end", "AI Door Lock system stopped.\n")
        except Exception as e:
            self.process_text.insert("end", f"Error running AI system: {e}\n")
            print(f"Error running AI system: {e}")
        finally:
            if self.ai_process is not None and self.ai_process.poll() is None:
                self.ai_process.terminate()
                self.ai_process.wait(timeout=2)
            self.ai_process = None

    def stop_system(self):
        print("Stop System button clicked. Stopping AI Door Lock...")
        self.running = False
        if self.ai_process is not None and self.ai_process.poll() is None:
            try:
                self.ai_process.terminate()
                self.ai_process.wait(timeout=2)
                print("AI Door Lock process terminated normally.")
            except Exception as e:
                print(f"Error terminating AI process: {e}")
                if self.ai_process.poll() is None:
                    self.ai_process.kill()
                    print("AI Door Lock process force killed.")
            finally:
                self.ai_process = None
        
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=2)
            print("AI thread joined.")
        
        # Ensure proper button visibility
        self.stop_system_button.pack_forget()
        self.feedback_button_frame.update()  # Force update of the frame
        self.start_system_again_button.pack(side="left", padx=10)
        self.exit_program_button.pack(side="left", padx=10)
        self.process_text.insert("end", "System stopped successfully.\n")
        self.update()  # Force GUI update

    def exit_program(self):
        print("Exit Program button clicked. Stopping AI Door Lock and exiting application...")
        self.running = False
        
        try:
            # Stop the AI process if it exists and is running
            if self.ai_process is not None:  # Check if ai_process exists
                if self.ai_process.poll() is None:  # Check if process is still running
                    print("Terminating AI process...")
                    self.ai_process.terminate()
                    self.ai_process.wait(timeout=2)
                    if self.ai_process.poll() is None:
                        print("Force killing AI process...")
                        self.ai_process.kill()
                    print("AI Door Lock process terminated.")
                self.ai_process = None

            # Join the AI thread if it exists and is running
            if self.ai_thread is not None and self.ai_thread.is_alive():
                print("Joining AI thread...")
                self.ai_thread.join(timeout=2)
                print("AI thread joined.")
                self.ai_thread = None

            # Close serial connection if open
            if self.ser is not None and self.ser.is_open:
                print("Closing serial connection...")
                self.ser.close()
                print(f"Serial connection closed on {self.com_port}")
                self.ser = None

            # Release camera if in use
            if self.cap is not None and self.cap.isOpened():
                print("Releasing camera...")
                self.cap.release()
                print("Camera released.")
                self.cap = None

            print("Destroying GUI...")
            self.destroy()
            print("Application exited successfully.")
            sys.exit(0)
        except Exception as e:
            print(f"Error during exit: {e}")
            sys.exit(1)

    def save_config(self):
        try:
            self.config = {
                "servo1_angle": self.servo1_angle,
                "servo2_angle": self.servo2_angle,
                "unlock_angle1": self.unlock_angle1,
                "unlock_angle2": self.unlock_angle2,
                "lock_delay": self.lock_delay,
                "com_port": self.com_port,
                "camera_index": self.camera_index
            }
            save_config(self.config)
        except ValueError:
            self.feedback_label.configure(text="Error: Invalid delay value")

if __name__ == "__main__":
    print("Starting GUI_control.py...")
    try:
        app = SmartDoorApp()
        print("GUI app created. Starting mainloop...")
        app.mainloop()
        print("Mainloop exited.")
    except Exception as e:
        print(f"Error in GUI_control.py: {e}")