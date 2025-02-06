#!/usr/bin/env python3
"""
Multi-Module GUI for Raspberry Pi using Tkinter:
  - Camera Feed with YOLO object detection:
      â¢ Overlays for each detected object:
          - Class and detection accuracy (F1 score)
          - Radar-derived distance (in meters)
          - Radar-derived speed (in m/s)
  - Thermal View (with a thermal colormap)
  - GPS Map Visualization using tkintermapview:
      â¢ Shows only the current location marker (no route history)
      â¢ Displays GPS coordinates and signal strength in an info label
  - FMCW Radar Display:
      â¢ Reads actual radar data from /dev/serial0 using custom commands
      â¢ Displays a simple radar visualization

Additional features:
  - A status bar showing CPU usage, camera FPS, and GPS signal quality.
  - Navigation via on-screen arrow buttons and keyboard arrow keys.
  - Screenshot and video recording in the Camera Feed module.

Prerequisites:
  pip install opencv-python ultralytics psutil pyserial pillow tkintermapview numpy
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import psutil
import serial
import time
import datetime
import threading
import math

# For the interactive OSM map
try:
    import tkintermapview
except ImportError:
    print("Error: tkintermapview package not found. Please install via 'pip install tkintermapview'")
    exit(1)

# Import YOLO model from ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. Please install via 'pip install ultralytics'")
    exit(1)

############################################################
# Global Constants (adjust as needed)
############################################################

CAMERA_WIDTH = 1280   # Adjust as per your camera resolution
CAMERA_HEIGHT = 720

############################################################
# Shared Application Data for Inter-Thread Communication
############################################################

class AppData:
    def _init_(self):
        self.latest_frame = None   # Latest raw camera frame (BGR numpy array)
        self.lock = threading.Lock()
        self.camera_fps = 0
        self.gps_signal = "N/A"
        self.radar_distance = -1.0  # Numeric (in meters); -1 indicates invalid
        self.radar_speed = -1.0     # Numeric (in m/s); -1 indicates invalid

############################################################
# Radar Communication Functions
############################################################

def calculate_checksum(data):
    return sum(data) & 0xFF

def send_distance_command(ser):
    frame_header = [0x53, 0x59]
    control_word = [0x08]
    command_word = [0x84]
    length_identification = [0x00, 0x01]
    data = [0x0F]
    data_for_checksum = frame_header + control_word + command_word + length_identification + data
    checksum = calculate_checksum(data_for_checksum)
    end_of_frame = [0x54, 0x43]
    command_packet = bytes(data_for_checksum + [checksum] + end_of_frame)
    ser.write(command_packet)

def receive_distance_response(ser):
    if ser.inWaiting() > 0:
        time.sleep(0.1)  # Allow some time for data arrival
        response = ser.read(ser.inWaiting())
        if len(response) >= 8:
            distance_code = response[6]
            distances = {
                0x00: 0.0,   # No movement
                0x01: 0.5,
                0x02: 1.0,
                0x03: 1.5,
                0x04: 2.0,
                0x05: 2.5,
                0x06: 3.0,
                0x07: 3.5,
                0x08: 4.0,
            }
            return distances.get(distance_code, -1.0)
    return -1.0

def send_speed_command(ser):
    frame_header = [0x53, 0x59]
    control_word = [0x08]
    command_word = [0x85]
    length_identification = [0x00, 0x01]
    data = [0x0F]
    data_for_checksum = frame_header + control_word + command_word + length_identification + data
    checksum = calculate_checksum(data_for_checksum)
    end_of_frame = [0x54, 0x43]
    command_packet = bytes(data_for_checksum + [checksum] + end_of_frame)
    ser.write(command_packet)

def receive_speed_response(ser):
    if ser.inWaiting() > 0:
        response = ser.read(ser.inWaiting())
        if len(response) >= 8:
            speed_code = response[6]
            speeds = {
                0x00: 0.0,
                0x01: 0.5,
                0x02: 1.0,
                0x03: 1.5,
                0x04: 2.0,
                0x05: 2.5,
                0x06: 3.0,
                0x07: 3.5,
                0x08: 4.0,
                0x09: 4.5,
                0x0A: 0.0,
                0x0B: -0.5,
                0x0C: -1.0,
                0x0D: -1.5,
                0x0E: -2.0,
                0x0F: -2.5,
                0x10: -3.0,
                0x11: -3.5,
                0x12: -4.0,
                0x13: -4.5,
                0x14: -5.0,
            }
            return speeds.get(speed_code, -1.0)
    return -1.0

def calculate_angle(x, width):
    center_offset = (x + width/2) - (CAMERA_WIDTH/2)
    normalized_x = center_offset / (CAMERA_WIDTH/2)
    angle = np.deg2rad(-normalized_x * 45)
    return angle

############################################################
# Global Camera Thread
############################################################

class GlobalCameraThread(threading.Thread):
    def _init_(self, app_data):
        super()._init_(daemon=True)
        self.app_data = app_data
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Unable to open camera.")
            self.running = False
        else:
            self.running = True
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.app_data.lock:
                    self.app_data.latest_frame = frame.copy()
            time.sleep(0.01)
        self.cap.release()
    def stop(self):
        self.running = False

############################################################
# YOLO Detection Thread
############################################################

class YOLODetectionThread(threading.Thread):
    def _init_(self, app_data, update_callback):
        """
        update_callback(image, fps): Called with a processed PIL Image and the current FPS.
        """
        super()._init_(daemon=True)
        self.app_data = app_data
        self.update_callback = update_callback
        self.running = True
        self.model = YOLO('yolov5s.pt')  # Downloads the model if needed.
        self.prev_time = time.time()
    def run(self):
        while self.running:
            frame = None
            with self.app_data.lock:
                if self.app_data.latest_frame is not None:
                    frame = self.app_data.latest_frame.copy()
            if frame is not None:
                results = self.model(frame, verbose=False)
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
                        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
                        classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []
                        for box, conf, cls in zip(boxes, confidences, classes):
                            x1, y1, x2, y2 = map(int, box)
                            class_name = self.model.names[int(cls)]
                            label = f"{class_name} F1: {conf*100:.1f}%"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                            cv2.putText(frame, label, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                            # Get radar data from AppData and format it.
                            distance = self.app_data.radar_distance
                            speed = self.app_data.radar_speed
                            if distance < 0:
                                distance_text = "Distance: N/A"
                            else:
                                distance_text = f"Distance: {distance:.2f}m"
                            if speed < -100:  # Using -1.0 as invalid marker here
                                speed_text = "Speed: N/A"
                            else:
                                speed_text = f"Speed: {speed:.2f} m/s"
                            cv2.putText(frame, distance_text, (x1, y1-30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                            cv2.putText(frame, speed_text, (x1, y1-50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                current_time = time.time()
                dt = current_time - self.prev_time
                fps = 1.0/dt if dt > 0 else 0
                self.prev_time = current_time
                with self.app_data.lock:
                    self.app_data.camera_fps = fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                self.update_callback(image, fps)
            else:
                time.sleep(0.01)
    def stop(self):
        self.running = False

############################################################
# Modified GPS Worker Thread (with robust error handling)
############################################################

class GPSWorkerThread(threading.Thread):
    def _init_(self, gps_callback):
        """
        gps_callback(lat, lon, signal): Called with latitude, longitude, and signal quality.
        """
        super()._init_(daemon=True)
        self.gps_callback = gps_callback
        self.running = True
        self.last_valid_location = None
        self.last_update_time = time.time()
        try:
            self.serial_port = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.5)
        except Exception as e:
            print(f"GPS Serial port error: {e}. Using simulated GPS data.")
            self.serial_port = None
    def parse_gprmc(self, sentence):
        try:
            parts = sentence.strip().split(',')
            if parts[0] == '$GPRMC' and parts[2] == 'A' and len(parts) >= 7:
                lat = self.nmea_to_decimal(parts[3], parts[4])
                lon = self.nmea_to_decimal(parts[5], parts[6])
                return lat, lon
        except Exception as e:
            print("Error parsing NMEA sentence:", e)
        return None
    def run(self):
        while self.running:
            if self.serial_port:
                try:
                    sentence = self.serial_port.readline().decode('ascii', errors='replace')
                    result = self.parse_gprmc(sentence)
                    if result:
                        lat, lon = result
                        self.last_valid_location = (lat, lon)
                        self.last_update_time = time.time()
                        self.gps_callback(lat, lon, "Good")
                    else:
                        if time.time() - self.last_update_time > 2.0:
                            lat = 37.7749 + np.random.uniform(-0.001, 0.001)
                            lon = -122.4194 + np.random.uniform(-0.001, 0.001)
                            self.gps_callback(lat, lon, "Simulated")
                            self.last_update_time = time.time()
                except Exception as ex:
                    print("GPS read error:", ex)
                    lat = 37.7749 + np.random.uniform(-0.001, 0.001)
                    lon = -122.4194 + np.random.uniform(-0.001, 0.001)
                    self.gps_callback(lat, lon, "Simulated")
                    time.sleep(1)
            else:
                lat = 37.7749 + np.random.uniform(-0.001, 0.001)
                lon = -122.4194 + np.random.uniform(-0.001, 0.001)
                self.gps_callback(lat, lon, "Simulated")
                time.sleep(1)
    def nmea_to_decimal(self, nmea_str, direction):
        try:
            if len(nmea_str) < 4:
                return 0.0
            deg_len = 2 if direction in ['N','S'] else 3
            degrees = float(nmea_str[:deg_len])
            minutes = float(nmea_str[deg_len:])
            decimal = degrees + minutes/60.0
            if direction in ['S','W']:
                decimal = -decimal
            return decimal
        except Exception as e:
            print("Error converting NMEA to decimal:", e)
            return 0.0
    def stop(self):
        self.running = False

############################################################
# Modified Radar Worker Thread (with numeric outputs)
############################################################

class RadarWorkerThread(threading.Thread):
    def _init_(self, radar_callback, app_data):
        """
        radar_callback(data): Called with a list of tuples (angle, distance) for radar display.
        app_data: Shared AppData instance to update radar readings.
        """
        super()._init_(daemon=True)
        self.radar_callback = radar_callback
        self.app_data = app_data
        self.running = True
        try:
            self.serial_port = serial.Serial(
                port='/dev/serial0',
                baudrate=115200,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1
            )
            print("Radar connected on /dev/serial0")
        except Exception as e:
            print(f"Error opening radar serial port: {e}")
            self.serial_port = None
    def run(self):
        while self.running:
            data = []
            if self.serial_port:
                try:
                    bbox_x = CAMERA_WIDTH // 2
                    bbox_width = 200
                    theta = calculate_angle(bbox_x, bbox_width)
                    send_distance_command(self.serial_port)
                    distance = receive_distance_response(self.serial_port)
                    send_speed_command(self.serial_port)
                    speed = receive_speed_response(self.serial_port)
                    self.app_data.radar_distance = distance
                    self.app_data.radar_speed = speed
                    angle_deg = -np.rad2deg(theta)
                    data = [(angle_deg, distance)]
                except Exception as e:
                    print("Radar read error:", e)
            else:
                time.sleep(0.1)
            self.radar_callback(data)
            time.sleep(0.5)
    def stop(self):
        self.running = False
        if self.serial_port:
            self.serial_port.close()

############################################################
# Camera Feed Frame with YOLO Object Detection
############################################################

class CameraFeedFrame(tk.Frame):
    def _init_(self, master, app_data, **kwargs):
        super()._init_(master, **kwargs)
        self.app_data = app_data
        self.current_image = None
        title = tk.Label(self, text="Camera Feed with Object Detection", font=("Helvetica", 16, "bold"))
        title.pack(pady=5)
        self.image_label = tk.Label(self, text="Waiting for camera feed...", bg="black")
        self.image_label.pack(expand=True, fill=tk.BOTH)
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=5)
        self.btn_screenshot = tk.Button(btn_frame, text="Capture Screenshot", command=self.capture_screenshot)
        self.btn_screenshot.pack(side=tk.LEFT, padx=5)
        self.btn_record = tk.Button(btn_frame, text="Start Recording", command=self.toggle_record)
        self.btn_record.pack(side=tk.LEFT, padx=5)
        self.recording = False
        self.video_writer = None
    def update_image(self, image, fps):
        self.current_image = image
        self.photo = ImageTk.PhotoImage(image=image)
        self.image_label.config(image=self.photo)
        if self.recording and self.video_writer is not None:
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)
    def capture_screenshot(self):
        if self.current_image is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            self.current_image.save(filename)
            print(f"Screenshot saved as {filename}")
    def toggle_record(self):
        if not self.recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.avi"
            if self.current_image is not None:
                width, height = self.current_image.size
            else:
                width, height = (640, 480)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            self.recording = True
            self.btn_record.config(text="Stop Recording")
            print(f"Started recording video to {filename}")
        else:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.btn_record.config(text="Start Recording")
            print("Stopped recording video.")

############################################################
# Thermal View Frame
############################################################

class ThermalViewFrame(tk.Frame):
    def _init_(self, master, app_data, **kwargs):
        super()._init_(master, **kwargs)
        self.app_data = app_data
        title = tk.Label(self, text="Thermal Camera Simulation", font=("Helvetica", 16, "bold"))
        title.pack(pady=5)
        self.image_label = tk.Label(self, text="Waiting for thermal feed...", bg="black")
        self.image_label.pack(expand=True, fill=tk.BOTH)
        self.update_thermal()
    def update_thermal(self):
        frame = None
        with self.app_data.lock:
            if self.app_data.latest_frame is not None:
                frame = self.app_data.latest_frame.copy()
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            rgb = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            self.photo = ImageTk.PhotoImage(image=image)
            self.image_label.config(image=self.photo)
        self.after(30, self.update_thermal)

############################################################
# Modified GPS Map Frame (only current location marker)
############################################################

class GPSMapFrame(tk.Frame):
    def _init_(self, master, **kwargs):
        super()._init_(master, **kwargs)
        title = tk.Label(self, text="GPS Map Visualization", font=("Helvetica", 16, "bold"))
        title.pack(pady=5)
        self.map_widget = tkintermapview.TkinterMapView(self, width=600, height=400, corner_radius=0)
        self.map_widget.pack(expand=True, fill=tk.BOTH)
        self.default_lat = 37.7749
        self.default_lon = -122.4194
        self.map_widget.set_position(self.default_lat, self.default_lon)
        self.map_widget.set_zoom(15)
        self.marker = self.map_widget.set_marker(self.default_lat, self.default_lon, text="Current location")
        self.info_label = tk.Label(self, text="Lat: N/A, Lon: N/A | Signal: N/A", font=("Helvetica", 12))
        self.info_label.pack(pady=5)
    def update_location(self, lat, lon, signal="N/A"):
        self.marker.set_position(lat, lon)
        self.map_widget.set_position(lat, lon)
        self.info_label.config(text=f"Lat: {lat:.6f}, Lon: {lon:.6f} | Signal: {signal}")

############################################################
# Radar Display Frame
############################################################

class RadarDisplayFrame(tk.Frame):
    def _init_(self, master, **kwargs):
        super()._init_(master, **kwargs)
        title = tk.Label(self, text="FMCW Radar Display", font=("Helvetica", 16, "bold"))
        title.pack(pady=5)
        self.canvas = tk.Canvas(self, width=400, height=400, bg="black")
        self.canvas.pack(expand=True, fill=tk.BOTH)
    def update_radar(self, data):
        self.canvas.delete("all")
        size = min(self.canvas.winfo_width(), self.canvas.winfo_height()) or 400
        center = size // 2
        for r in range(50, center, 50):
            self.canvas.create_oval(center - r, center - r, center + r, center + r, outline="green")
        self.canvas.create_line(center, center, center, 0, fill="green", width=2)
        for angle, distance in data:
            try:
                d = float(distance)
            except Exception:
                d = 0.0
            r = (d / 5.0) * center  # Scale for radar range of 5m.
            rad = math.radians(angle - 90)
            x = center + int(r * math.cos(rad))
            y = center + int(r * math.sin(rad))
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")

############################################################
# Main Application Class
############################################################

class MainApplication(tk.Tk):
    def _init_(self):
        super()._init_()
        self.title("Raspberry Pi Multi-Module GUI")
        self.geometry("900x700")
        self.app_data = AppData()
        self.container = tk.Frame(self)
        self.container.pack(expand=True, fill=tk.BOTH)
        self.frames = {}
        # Pages: 0 = Camera Feed, 1 = Thermal View, 2 = GPS Map, 3 = Radar Display.
        self.current_index = 0
        self.frames[0] = CameraFeedFrame(self.container, self.app_data)
        self.frames[1] = ThermalViewFrame(self.container, self.app_data)
        self.frames[2] = GPSMapFrame(self.container)
        self.frames[3] = RadarDisplayFrame(self.container)
        self.show_frame(0)
        self.status_bar = tk.Label(self, text="Status", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        nav_frame = tk.Frame(self)
        nav_frame.pack(side=tk.BOTTOM, pady=5)
        btn_left = tk.Button(nav_frame, text="â", command=lambda: self.navigate("left"))
        btn_left.pack(side=tk.LEFT, padx=5)
        btn_right = tk.Button(nav_frame, text="â", command=lambda: self.navigate("right"))
        btn_right.pack(side=tk.LEFT, padx=5)
        btn_up = tk.Button(nav_frame, text="â", command=lambda: self.navigate("up"))
        btn_up.pack(side=tk.LEFT, padx=5)
        btn_down = tk.Button(nav_frame, text="â", command=lambda: self.navigate("down"))
        btn_down.pack(side=tk.LEFT, padx=5)
        self.bind("<Left>", lambda event: self.navigate("left"))
        self.bind("<Right>", lambda event: self.navigate("right"))
        self.bind("<Up>", lambda event: self.navigate("up"))
        self.bind("<Down>", lambda event: self.navigate("down"))
        self.update_status_bar()
        self.start_threads()
    def show_frame(self, index):
        for widget in self.container.winfo_children():
            widget.pack_forget()
        self.current_index = index
        self.frames[index].pack(expand=True, fill=tk.BOTH)
    def navigate(self, direction):
        grid_map = {
            0: {"right": 1, "down": 2},
            1: {"left": 0, "down": 3},
            2: {"up": 0, "right": 3},
            3: {"up": 1, "left": 2}
        }
        if direction in grid_map[self.current_index]:
            new_index = grid_map[self.current_index][direction]
            self.show_frame(new_index)
            self.status_bar.config(text=f"Switched to page {new_index}")
    def update_status_bar(self):
        cpu = psutil.cpu_percent()
        fps = self.app_data.camera_fps
        gps_signal = self.app_data.gps_signal
        self.status_bar.config(text=f"CPU: {cpu}% | FPS: {fps:.1f} | GPS Signal: {gps_signal}")
        self.after(1000, self.update_status_bar)
    def start_threads(self):
        self.global_camera_thread = GlobalCameraThread(self.app_data)
        self.global_camera_thread.start()
        def yolo_callback(image, fps):
            self.after(0, self.frames[0].update_image, image, fps)
        self.yolo_thread = YOLODetectionThread(self.app_data, yolo_callback)
        self.yolo_thread.start()
        def gps_callback(lat, lon, signal):
            self.app_data.gps_signal = signal
            self.after(0, self.frames[2].update_location, lat, lon, signal)
        self.gps_thread = GPSWorkerThread(gps_callback)
        self.gps_thread.start()
        def radar_callback(data):
            self.after(0, self.frames[3].update_radar, data)
        self.radar_thread = RadarWorkerThread(radar_callback, self.app_data)
        self.radar_thread.start()
    def on_closing(self):
        self.global_camera_thread.stop()
        self.yolo_thread.stop()
        self.gps_thread.stop()
        self.radar_thread.stop()
        self.destroy()

############################################################
# Main Execution
############################################################

if _name_ == "_main_":
    app = MainApplication()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()