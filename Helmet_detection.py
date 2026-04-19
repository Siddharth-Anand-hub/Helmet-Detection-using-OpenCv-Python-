import numpy as np
import cv2
import imutils
import datetime
import urllib.request
import os
import winsound  # For Windows alert sound

class HelmetDetectionSystem:
    def __init__(self, confidence_threshold=0.5):
        """
        Helmet Detection System using YOLOv4-Tiny
        Detects persons and attempts to identify helmet presence
        """
        self.confidence_threshold = confidence_threshold
        
        # Model files
        self.weights_file = "yolov4-tiny.weights"
        self.config_file = "yolov4-tiny.cfg"
        self.names_file = "coco.names"
        
        print("=" * 80)
        print("HELMET DETECTION SYSTEM - Initializing...")
        print("=" * 80)
        
        # Download models if needed
        self._download_models()
        
        # Load class names
        with open(self.names_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Load YOLO network
        print("\nLoading detection model...")
        self.net = cv2.dnn.readNet(self.weights_file, self.config_file)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        print("✓ Model loaded successfully!")
        
        # Load Haar Cascade for head detection (helps identify helmet area)
        self.head_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Colors
        self.color_with_helmet = (0, 255, 0)      # Green
        self.color_without_helmet = (0, 0, 255)   # Red
        self.color_person = (255, 255, 0)         # Yellow
        
        # Statistics
        self.total_persons = 0
        self.persons_with_helmet = 0
        self.persons_without_helmet = 0
        self.violations = []
        
        # Alert settings
        self.alert_enabled = True
        self.last_alert_time = None
        self.alert_cooldown = 3  # seconds
    
    def _download_models(self):
        """Download model files if needed"""
        files_info = {
            self.names_file: 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
            self.config_file: 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
            self.weights_file: 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights'
        }
        
        for filename, url in files_info.items():
            if not os.path.exists(filename):
                print(f"\nDownloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename, 
                        reporthook=lambda c, b, t: print(f'\r  Progress: {min(100, int(c*b*100/t)) if t > 0 else 0}%', end=''))
                    print(f"\n✓ {filename} downloaded")
                except Exception as e:
                    print(f"\n✗ Failed: {e}")
                    raise
            else:
                print(f"✓ {filename} exists")
    
    def detect_persons(self, frame):
        """Detect persons in frame using YOLO"""
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only detect 'person' class (class_id = 0 in COCO)
                if class_id == 0 and confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        
        persons = []
        if len(indices) > 0:
            for i in indices.flatten():
                persons.append({
                    'box': boxes[i],
                    'confidence': confidences[i]
                })
        
        return persons
    
    def check_helmet(self, frame, person_box):
        """
        Check if person is wearing helmet using head region analysis
        Returns: (has_helmet: bool, confidence: float)
        """
        x, y, w, h = person_box
        
        # Ensure coordinates are within frame
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        # Extract upper 30% of person box (head region)
        head_region_height = int(h * 0.3)
        if head_region_height < 20:
            return False, 0.0
        
        head_region = frame[y:y+head_region_height, x:x+w]
        
        if head_region.size == 0:
            return False, 0.0
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Method 1: Detect face in head region
        faces = self.head_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
        
        # Method 2: Analyze head region colors and texture
        # Helmets typically have uniform colors and smooth texture
        # Calculate color variance
        color_variance = np.var(hsv[:, :, 1])  # Saturation variance
        brightness_mean = np.mean(hsv[:, :, 2])  # Value (brightness) mean
        
        # Method 3: Edge detection for helmet shape
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Decision logic (heuristic-based)
        confidence = 0.0
        has_helmet = False
        
        # If no face detected in head region, likely wearing helmet
        if len(faces) == 0:
            confidence += 0.4
        
        # High color uniformity suggests helmet
        if color_variance < 500:
            confidence += 0.3
        
        # Moderate edge density suggests helmet structure
        if 0.1 < edge_density < 0.4:
            confidence += 0.3
        
        has_helmet = confidence > 0.5
        
        return has_helmet, min(confidence, 1.0)
    
    def trigger_alert(self):
        """Trigger alert for helmet violation"""
        current_time = datetime.datetime.now()
        
        # Check cooldown
        if self.last_alert_time:
            time_diff = (current_time - self.last_alert_time).total_seconds()
            if time_diff < self.alert_cooldown:
                return
        
        # Play alert sound (Windows only)
        try:
            winsound.Beep(1000, 200)  # 1000 Hz for 200ms
        except:
            pass
        
        self.last_alert_time = current_time
    
    def process_frame(self, frame):
        """Process frame and detect helmets"""
        # Detect persons
        persons = self.detect_persons(frame)
        
        detections = []
        current_violations = 0
        
        for person in persons:
            x, y, w, h = person['box']
            
            # Check for helmet
            has_helmet, helmet_confidence = self.check_helmet(frame, person['box'])
            
            detection = {
                'box': person['box'],
                'person_confidence': person['confidence'],
                'has_helmet': has_helmet,
                'helmet_confidence': helmet_confidence
            }
            detections.append(detection)
            
            # Update statistics
            self.total_persons += 1
            if has_helmet:
                self.persons_with_helmet += 1
            else:
                self.persons_without_helmet += 1
                current_violations += 1
                
                # Log violation
                violation = {
                    'timestamp': datetime.datetime.now(),
                    'confidence': person['confidence']
                }
                self.violations.append(violation)
        
        # Trigger alert if violations detected
        if current_violations > 0 and self.alert_enabled:
            self.trigger_alert()
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw detection results on frame"""
        for detection in detections:
            x, y, w, h = detection['box']
            has_helmet = detection['has_helmet']
            person_conf = detection['person_confidence']
            helmet_conf = detection['helmet_confidence']
            
            # Choose color based on helmet status
            color = self.color_with_helmet if has_helmet else self.color_without_helmet
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Create label
            status = "HELMET: YES" if has_helmet else "HELMET: NO"
            label = f"{status} ({helmet_conf:.2f})"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y-35), (x+label_w+10, y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x+5, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw status indicator
            status_text = "✓ SAFE" if has_helmet else "✗ VIOLATION"
            cv2.putText(frame, status_text, (x+5, y+h+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def add_statistics_panel(self, frame, detections):
        """Add statistics panel to frame"""
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 280), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        # Title
        cv2.putText(frame, "HELMET DETECTION SYSTEM", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Separator
        cv2.line(frame, (20, 85), (430, 85), (100, 100, 100), 1)
        
        # Current detections
        persons_now = len(detections)
        with_helmet_now = sum(1 for d in detections if d['has_helmet'])
        without_helmet_now = persons_now - with_helmet_now
        
        cv2.putText(frame, "CURRENT FRAME:", (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.putText(frame, f"Persons Detected: {persons_now}", (30, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"With Helmet: {with_helmet_now}", (30, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Without Helmet: {without_helmet_now}", (30, 195),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Session statistics
        cv2.putText(frame, "SESSION STATS:", (20, 225),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        compliance_rate = 0
        if self.total_persons > 0:
            compliance_rate = (self.persons_with_helmet / self.total_persons) * 100
        
        cv2.putText(frame, f"Total Violations: {len(self.violations)}", (30, 255),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alert status
        alert_status = "ON" if self.alert_enabled else "OFF"
        alert_color = (0, 255, 0) if self.alert_enabled else (128, 128, 128)
        cv2.putText(frame, f"Alert: {alert_status}", (350, 255),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)
        
        return frame
    
    def save_violation_report(self):
        """Save violation report to file"""
        if not self.violations:
            print("No violations to report")
            return
        
        filename = f"helmet_violations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HELMET DETECTION SYSTEM - VIOLATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total Violations: {len(self.violations)}\n")
            f.write(f"Total Persons Detected: {self.total_persons}\n")
            f.write(f"Persons With Helmet: {self.persons_with_helmet}\n")
            f.write(f"Persons Without Helmet: {self.persons_without_helmet}\n\n")
            
            if self.total_persons > 0:
                compliance = (self.persons_with_helmet / self.total_persons) * 100
                f.write(f"Compliance Rate: {compliance:.2f}%\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("VIOLATION LOG:\n")
            f.write("-" * 80 + "\n\n")
            
            for i, violation in enumerate(self.violations, 1):
                f.write(f"Violation #{i}\n")
                f.write(f"  Time: {violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  Detection Confidence: {violation['confidence']:.2f}\n\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"✓ Violation report saved: {filename}")

def main():
    print("\n" + "=" * 80)
    print("HELMET DETECTION SYSTEM")
    print("Safety Monitoring & Compliance System")
    print("=" * 80)
    print("\nControls:")
    print("  'q' - Quit and generate report")
    print("  's' - Save screenshot")
    print("  'r' - Generate violation report")
    print("  'a' - Toggle audio alerts")
    print("  '+' - Increase detection sensitivity")
    print("  '-' - Decrease detection sensitivity")
    print("-" * 80 + "\n")
    
    try:
        detector = HelmetDetectionSystem(confidence_threshold=0.5)
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        return
    
    # Initialize camera
    print("\nStarting camera...")
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("✗ Could not open camera")
        return
    
    print("✓ Camera started!")
    print("\n" + "=" * 80)
    print("SYSTEM ACTIVE - Monitoring for helmet compliance...")
    print("=" * 80 + "\n")
    
    frame_count = 0
    fps_start = datetime.datetime.now()
    fps = 0
    
    while True:
        ret, frame = camera.read()
        
        if not ret:
            print("✗ Failed to capture frame")
            break
        
        # Resize frame
        frame = imutils.resize(frame, width=900)
        
        # Process frame
        detections = detector.process_frame(frame)
        
        # Draw detections
        frame = detector.draw_detections(frame, detections)
        frame = detector.add_statistics_panel(frame, detections)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps_end = datetime.datetime.now()
            time_diff = (fps_end - fps_start).total_seconds()
            fps = 30 / time_diff if time_diff > 0 else 0
            fps_start = fps_end
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Helmet Detection System", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nShutting down system...")
            break
        elif key == ord('s'):
            filename = f"helmet_detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Screenshot saved: {filename}")
        elif key == ord('r'):
            detector.save_violation_report()
        elif key == ord('a'):
            detector.alert_enabled = not detector.alert_enabled
            status = "ENABLED" if detector.alert_enabled else "DISABLED"
            print(f"✓ Audio alerts {status}")
        elif key == ord('+') or key == ord('='):
            detector.confidence_threshold = min(0.95, detector.confidence_threshold + 0.05)
            print(f"✓ Detection sensitivity: {detector.confidence_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            detector.confidence_threshold = max(0.2, detector.confidence_threshold - 0.05)
            print(f"✓ Detection sensitivity: {detector.confidence_threshold:.2f}")
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    
    # Final report
    print("\n" + "=" * 80)
    print("SESSION SUMMARY")
    print("=" * 80)
    print(f"Total Frames Processed: {frame_count:,}")
    print(f"Total Persons Detected: {detector.total_persons}")
    print(f"Persons With Helmet: {detector.persons_with_helmet}")
    print(f"Persons Without Helmet: {detector.persons_without_helmet}")
    print(f"Total Violations: {len(detector.violations)}")
    
    if detector.total_persons > 0:
        compliance = (detector.persons_with_helmet / detector.total_persons) * 100
        print(f"Helmet Compliance Rate: {compliance:.2f}%")
    
    print("=" * 80)
    
    # Auto-generate final report
    if detector.violations:
        print("\nGenerating final violation report...")
        detector.save_violation_report()

if __name__ == "__main__":
    main()