# Vehicle Speed Estimation System — Configuration

# Region of Interest (ROI) polygon for speed estimation — list of [x, y] points
ROI_COORDINATES = [[750, 350], [1150, 350], [1870, 1079], [50, 1079]]

# Target dimensions for perspective transformation (meters)
TARGET_WIDTH = 12
TARGET_HEIGHT = 50

# Vehicle classes to detect (COCO IDs): car, motorcycle, bus, truck
TARGET_CLASSES = [2, 3, 5, 7]

# Detection confidence threshold and NMS threshold
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.7

# Model and video files
MODEL_PATH = "yolov8x.pt"
INPUT_VIDEO = "Vehicle-Flow.mp4"               # input video file name
OUTPUT_VIDEO = "Vehicle-Speed-Estimation.mp4"  # output video file name
