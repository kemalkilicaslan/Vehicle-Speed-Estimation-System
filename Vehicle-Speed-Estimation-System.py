# Vehicle Speed Estimation System

# Import the necessary libraries
import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
from ultralytics import YOLO
import config

# Region of Interest coordinates for speed estimation (loaded from config.py)
ROI_COORDINATES = np.array(config.ROI_COORDINATES, dtype=np.int32)

# Target dimensions for perspective transformation (loaded from config.py)
TARGET_WIDTH = config.TARGET_WIDTH
TARGET_HEIGHT = config.TARGET_HEIGHT
TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]])

# Vehicle classes for detection (loaded from config.py)
TARGET_CLASSES = config.TARGET_CLASSES  # car, motorcycle, bus, truck

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.array([]).reshape(0, 2)
        
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# Load YOLO model (path loaded from config.py)
model = YOLO(config.MODEL_PATH)

# Open video (paths loaded from config.py)
video_path = config.INPUT_VIDEO  # write the name of the video file here
output_file = config.OUTPUT_VIDEO  # extension the name and extension of the video file to be recorded

# Initialize video processing
video_info = sv.VideoInfo.from_video_path(video_path=video_path)
frames_generator = sv.get_video_frames_generator(source_path=video_path)

# Initialize tracking components
tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother()

# Initialize annotators
label_annotator = sv.LabelAnnotator(
    text_thickness=2,
    text_scale=0.8,
    text_position=sv.Position.TOP_CENTER,
    color=sv.Color.from_bgr_tuple((0, 150, 0))  
)

trace_annotator = sv.TraceAnnotator(
    thickness=2,
    trace_length=10,
    color=sv.Color.from_bgr_tuple((0, 150, 0))
)

# Initialize ROI zone and view transformer
polygon_zone = sv.PolygonZone(polygon=ROI_COORDINATES)
view_transformer = ViewTransformer(source=ROI_COORDINATES, target=TARGET)

# Track coordinates for speed calculation
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

# Create VideoSink for output
with sv.VideoSink(target_path=output_file, video_info=video_info) as sink:
    for frame in frames_generator:
        # Run YOLO inference
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter detections by class and confidence
        detections = detections[np.isin(detections.class_id, TARGET_CLASSES)]
        detections = detections[detections.confidence > config.CONFIDENCE_THRESHOLD]

        # Apply ROI filter - only detect objects inside the Region of Interest
        roi_mask = polygon_zone.trigger(detections)
        detections = detections[roi_mask]

        # Update tracker and smoother for objects inside ROI
        detections = tracker.update_with_detections(detections)
        detections = detections.with_nms(threshold=config.NMS_THRESHOLD)
        detections = smoother.update_with_detections(detections)

        # Get center points of detected vehicles
        center_points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)

        # Transform points for speed calculation using perspective transformation
        if len(center_points) > 0:
            points = view_transformer.transform_points(points=center_points).astype(int)
            
            # Store coordinates for each tracked object for speed calculation
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

        # Calculate speed and create labels for each vehicle
        labels = []
        for tracker_id in detections.tracker_id:
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time = len(coordinates[tracker_id]) / video_info.fps if len(coordinates[tracker_id]) > 0 else 1
            speed = distance / time * 3.6  # Convert to km/h
            labels.append(f"{int(speed)} km/h")
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Apply trace annotation to show vehicle paths
        annotated_frame = trace_annotator.annotate(annotated_frame, detections)
        
        # Apply label annotation to show speeds
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
        
        # Write the drawn frame to the video file to be saved
        sink.write_frame(frame=annotated_frame)
        
        # Show video with vehicle speed estimation
        cv2.imshow('Vehicle Speed Estimation', annotated_frame)
        
        # Switch off video when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release all open windows
cv2.destroyAllWindows()