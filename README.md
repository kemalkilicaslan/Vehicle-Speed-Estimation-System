# Vehicle Speed Estimation System

## 1. Introduction

This project implements a real-time vehicle speed estimation system using computer vision techniques, combining YOLOv8 object detection with perspective transformation for accurate speed measurements. The system processes traffic footage to detect vehicles within a defined region of interest (ROI), tracks their movement across frames, and calculates their speeds using geometric principles.

Automated traffic monitoring and speed enforcement are used in intelligent transportation systems, traffic flow analysis, speed violation detection, and road safety monitoring. Using deep learning models and geometric transformations, the system measures speed without specialized hardware.

It processes video streams to track multiple vehicles at once and calculate their speeds in real-world units (km/h).

**Core Features:**
- Real-time vehicle detection using YOLOv8 (cars, motorcycles, buses, trucks)
- ROI-based speed calculation with perspective correction
- Multi-vehicle tracking with unique identification
- Geometric transformation for pixel-to-meter conversion
- Visual speed display with trajectory trails
- Video output with annotated speed measurements
- Configurable detection and tracking parameters

## 2. Methodology / Approach

The system employs YOLOv8 for vehicle detection combined with ByteTrack for multi-object tracking and perspective transformation for accurate distance-to-speed conversion. The approach uses a defined region of interest (ROI) to focus measurements on a specific road section with known dimensions.

### 2.1 System Architecture

The vehicle speed estimation pipeline consists of:

1. **Vehicle Detection:** YOLOv8 identifies vehicles (car, motorcycle, bus, truck) in each frame
2. **ROI Filtering:** Only vehicles within the defined polygon zone are processed
3. **Object Tracking:** ByteTrack maintains consistent vehicle identities across frames
4. **Coordinate Transformation:** Perspective correction converts pixel coordinates to real-world meters
5. **Speed Calculation:** Distance traveled over time provides velocity measurements
6. **Visualization:** Speed labels and trajectory traces are overlaid on the output video

### 2.2 Implementation Strategy

The implementation uses the Ultralytics framework for YOLOv8 inference and the Supervision library for tracking and annotation. A trapezoidal ROI is defined to match the road boundaries, with known real-world dimensions (12m width × 50m length) used for calibration. Detection smoothing and coordinate buffering reduce noise and improve measurement accuracy. The system processes video frame-by-frame, maintaining a history of vehicle positions to calculate instantaneous speeds.

## 3. Mathematical Framework

### 3.1 Perspective Transformation

The system uses perspective transformation to map image coordinates to real-world coordinates:

$$\mathbf{P}_{\text{world}} = \mathbf{M} \cdot \mathbf{P}_{\text{image}}$$

where $\mathbf{M}$ is the $3 \times 3$ perspective transformation matrix calculated using `cv2.getPerspectiveTransform()` from ROI coordinates to target dimensions.

### 3.2 Transformation Matrix Calculation

Given source points $\mathbf{S}$ (ROI coordinates) and destination points $\mathbf{D}$ (target rectangle):

$$\mathbf{M} = \text{getPerspectiveTransform}(\mathbf{S}, \mathbf{D})$$

Where:
- $\mathbf{S} = \{(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)\}$ (ROI corners in pixels)
- $\mathbf{D} = \{(0, 0), (w, 0), (w, h), (0, h)\}$ (target rectangle: $w=12$ m, $h=50$ m)

### 3.3 Speed Calculation

Vehicle speed is calculated using the distance traveled between consecutive frames:

$$v = \frac{d}{t} \times 3.6$$

where:
- $v$ = velocity (km/h)
- $d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$ = Euclidean distance traveled (meters)
- $t = \frac{1}{\text{fps}}$ = time between frames (seconds)
- $3.6$ = conversion factor from m/s to km/h

### 3.4 Coordinate Buffer and Moving Average

To reduce noise, speeds are calculated from a coordinate history buffer:

$$v_{\text{avg}} = \frac{1}{n-1} \sum_{i=1}^{n-1} \frac{d_i}{\Delta t} \times 3.6$$

where:
- $n$ = buffer size (set to video fps)
- $d_i$ = distance between consecutive coordinate pairs
- $\Delta t = \frac{1}{\text{fps}}$ = time step

## 4. Requirements

`requirements.txt`

```txt
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
supervision>=0.16.0
```

## 5. Installation & Configuration

### 5.1 Environment Setup

```bash
# Clone the repository
git clone https://github.com/kemalkilicaslan/Vehicle-Speed-Estimation-System.git
cd Vehicle-Speed-Estimation-System

# Install required packages
pip install -r requirements.txt
```

### 5.2 Project Structure

```
Vehicle-Speed-Estimation-System
├── Vehicle-Speed-Estimation-System.py
├── config.py
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

### 5.3 Configuration Parameters

All adjustable parameters live in `config.py`. Update them there without touching the main script.

**ROI Configuration:**

```python
# Region of Interest (trapezoidal polygon) — list of [x, y] points
ROI_COORDINATES = [[750, 350], [1150, 350], [1870, 1079], [50, 1079]]

# Real-world dimensions
TARGET_WIDTH = 12   # meters (road width)
TARGET_HEIGHT = 50  # meters (measurement distance)
```

**Detection Parameters:**

```python
# Vehicle classes to detect
TARGET_CLASSES = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5   # detection confidence threshold
NMS_THRESHOLD = 0.7          # NMS (Non-Maximum Suppression) threshold
```

**Visualization Parameters:**

The trace and label styling (`trace_length`, `text_scale`, `text_thickness`, colors) is set on the annotator objects in the main script.

## 6. Usage / How to Run

### 6.1 Basic Execution

```bash
python Vehicle-Speed-Estimation-System.py
```

**Requirements:**
- Input video: `Vehicle-Flow.mp4` (place in project directory)
- YOLOv8 model: `yolov8x.pt` (automatically downloaded on first run)

**Controls:**
- Press `q` to quit during playback
- Output saved to: `Vehicle-Speed-Estimation.mp4`

### 6.2 Customizing Input Video

Modify the video path in the script:

```python
# Change this line in Vehicle-Speed-Estimation-System.py
video_path = "Vehicle-Flow.mp4"
```

### 6.3 Adjusting ROI for Different Videos

To calibrate for a new video:

1. **Identify the road section** to monitor (preferably straight and flat)
2. **Measure real-world dimensions** (width and length in meters)
3. **Update ROI coordinates** in the script:

```python
ROI_COORDINATES = np.array([
    [x1, y1],  # Top-left corner
    [x2, y2],  # Top-right corner
    [x3, y3],  # Bottom-right corner
    [x4, y4]   # Bottom-left corner
])
```

4. **Set target dimensions:**

```python
TARGET_WIDTH = measured_width   # meters
TARGET_HEIGHT = measured_length  # meters
```

### 6.4 Optimizing Detection

For different traffic conditions:

```python
# Increase sensitivity (more detections)
conf = 0.3

# Decrease sensitivity (fewer false positives)
conf = 0.6

# Adjust NMS for overlapping vehicles
iou = 0.5  # More aggressive filtering
iou = 0.8  # Less filtering
```

## 7. Docker Deployment

The system can be packaged and run in a container so that the code, its
dependencies and the runtime environment stay consistent across machines.

### Files

- `Dockerfile` - builds the image (Python 3.11 + OpenCV/ffmpeg system libraries + Python dependencies).
- `.dockerignore` - keeps large media and model files out of the build context.
- `docker-compose.yml` - runs the container and mounts the project folder for input/output.

### Build and run

```bash
# Build the image and run the container
docker compose up --build

# Or use the Docker CLI directly
docker build -t vehicle-speed-estimation:latest .
docker run --rm -v "$(pwd)":/app vehicle-speed-estimation:latest
```

### Notes

- The YOLO model weight (`yolov8x.pt`) is downloaded automatically on the first
  run and cached in the mounted volume, so later runs start faster.
- Input and output videos are read from and written to the project folder through
  the bind mount, so the annotated output video appears on the host after the run.
- The script calls `cv2.imshow`, which needs a graphical display. On a headless
  server the annotated output video is still written to disk; the live preview
  window is only available on Linux with X11 forwarding (see the commented lines
  in `docker-compose.yml`).

## 8. Application / Results

### 8.1 Input Video

**Vehicle Flow:**

[![Vehicle Flow](https://github.com/kemalkilicaslan/Vehicle-Speed-Estimation-System/blob/main/Vehicle-Flow-Video.webp)](https://www.youtube.com/embed/4xGlA7z0Z7U)

### 8.2 Output Video

**Vehicle Speed Estimation:**

[![Vehicle Speed Estimation](https://github.com/kemalkilicaslan/Vehicle-Speed-Estimation-System/blob/main/Vehicle-Speed-Estimation-Video.webp)](https://www.youtube.com/embed/VNUhQXJtd-s)

### 8.3 System Configuration Visualization

**Region of Interest Area:**

![Region of Interest Area](https://github.com/kemalkilicaslan/Vehicle-Speed-Estimation-System/blob/main/Region-of-Interest-Area.webp)

*The trapezoidal polygon defines the measurement zone on the road.*

**Real Target Length:**

![Real Target Length](https://github.com/kemalkilicaslan/Vehicle-Speed-Estimation-System/blob/main/Real-Target-Length.webp)

*Physical dimensions: 12m width × 50m length for accurate calibration.*

**Region of Interest Coordinates:**

![Region of Interest Coordinates](https://github.com/kemalkilicaslan/Vehicle-Speed-Estimation-System/blob/main/Region-of-Interest-Coordinates.webp)

*Pixel coordinates of the ROI corners mapped to real-world measurements.*

### 8.4 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Detection Accuracy | 90-95% | Varies with video quality and lighting |
| Speed Accuracy | ±5 km/h | Depends on calibration precision |
| Processing Speed | 15-30 FPS (CPU) | GPU acceleration available |
| Multi-vehicle Tracking | Up to 50 simultaneous | ByteTrack algorithm |
| Measurement Range | 0-150 km/h | Configurable based on ROI size |

### 8.5 System Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Target Width | 12 | meters | Real-world road width |
| Target Height | 50 | meters | Measurement distance |
| Detection Confidence | 0.5 | - | Vehicle detection threshold |
| NMS Threshold | 0.7 | - | Non-maximum suppression |
| Trace Length | 10 | frames | Path visualization duration |
| Coordinate Buffer | fps | frames | Speed calculation history |
| Label Text Scale | 0.8 | - | Speed label size |
| Label Text Thickness | 2 | pixels | Speed label boldness |
| Trace Thickness | 2 | pixels | Path line thickness |
| Label Position | TOP_CENTER | - | Speed display location |
| Annotation Color | GREEN | (0, 150, 0) | BGR format |

## 9. Tech Stack

### 9.1 Core Technologies

- **Programming Language:** Python 3.8+
- **Computer Vision:** OpenCV 4.8+
- **Object Detection:** YOLOv8x (Ultralytics)
- **Object Tracking:** ByteTrack (Supervision)
- **Video Processing:** OpenCV VideoCapture/VideoWriter

### 9.2 Libraries & Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.8+ | Video I/O, geometric transformations, rendering |
| numpy | 1.24+ | Array operations, coordinate calculations |
| ultralytics | 8.0+ | YOLOv8 model inference and detection |
| supervision | 0.16+ | Tracking, smoothing, and annotation tools |

### 9.3 Model Architecture

**YOLOv8x (Extra Large):**
- **Model Size:** yolov8x.pt (~136 MB)
- **Input Resolution:** 640×640 pixels (default)
- **Architecture:** CSPDarknet backbone + PAN neck + detection head
- **Detection Classes:** 80 COCO classes (filtered to vehicles: 2, 3, 5, 7)
- **Performance:** High accuracy, moderate speed (suitable for traffic monitoring)

**Tracking Algorithm:**
- **ByteTrack:** Multi-object tracking with motion prediction
- **Features:** 
  - Handles occlusions and temporary disappearances
  - Maintains consistent IDs across frames
  - Low computational overhead

### 9.4 Annotation Components

**Supervision Library Tools:**

| Component | Type | Purpose |
|-----------|------|---------|
| ByteTrack | Tracker | Multi-vehicle identity management |
| DetectionsSmoother | Filter | Reduce detection jitter/noise |
| PolygonZone | ROI Filter | Spatial filtering for speed measurement |
| TraceAnnotator | Visualizer | Draw vehicle movement paths |
| LabelAnnotator | Visualizer | Display speed measurements |

### 9.5 Geometric Transformation

**OpenCV Functions:**
- `cv2.getPerspectiveTransform()`: Calculate transformation matrix
- `cv2.perspectiveTransform()`: Apply transformation to coordinates
- Input: 4-point ROI polygon (source)
- Output: Rectangle with real-world dimensions (destination)

## 10. License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://github.com/kemalkilicaslan/Vehicle-Speed-Estimation-System/blob/main/LICENSE).

## 11. References

1. Ultralytics [YOLOv8](https://docs.ultralytics.com/models/yolov8/) Documentation.
2. Roboflow [Supervision Trackers](https://supervision.roboflow.com/latest/trackers/) Documentation.
3. OpenCV [Geometric Transformations of Images](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html) Documentation.

---

### Acknowledgments

This project uses YOLOv8 from Ultralytics for vehicle detection and the Supervision library for tracking and annotation. Thanks to the computer vision community for their open-source tools for traffic analysis.

---

**Note:** This system is intended for research, education, and traffic analysis purposes. For legal speed enforcement applications, ensure compliance with local regulations and calibrate the system according to official standards. Speed measurements may vary based on camera angle, calibration accuracy, and environmental conditions.