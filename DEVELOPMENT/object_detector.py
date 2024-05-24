## detect arena boundary 
# --> opencv contour detection (opencv)
# --> Adaptive Thresholding and Morphological Operations
# // background subtraction, not a good approach because its purpose to detect moving object
# feature to add position of winner, position of loser, winner remaining spin duration, 
# distance of two object before battle ends, distance from the center in radius of both
# analyze the distance of each object to collision impact point


'''
#checkpoint 3 --> boundary detection + spin detection
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

video_path = 'data/sample_2.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('fine-tune-yolo/best.pt')
box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=1
)

# Set the color of the ellipse to green
ellipse_color = (0, 255, 0)
# Static IDs for the two beyblades
static_ids = [1, 2]
# Dictionary to store previous bounding box points for each object
prev_bbox_points = {1: None, 2: None}

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Get the dimensions of the frame
    height, width = frame.shape[:2]

    # Calculate the center of the frame
    center_x = width // 2
    center_y = height // 2

    # Define ellipse parameters
    axis_x = width // 2
    axis_y = height // 2
    angle = 0  # Angle in degrees
    ellipse_params = ((center_x, center_y), (axis_x, axis_y), angle)

    # Draw the ellipse
    cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), angle, 0, 360, ellipse_color, 2)

    # Get the ellipse points for pointPolygonTest
    ellipse_points = cv2.ellipse2Poly((center_x, center_y), (axis_x, axis_y), angle, 0, 360, 1).astype(np.int32)

    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Initialize a dictionary to store the spin status and object ID of each object
    obj_info = {}

    for obj_id in static_ids:
        # Check if the object has been detected
        if len(detections.xyxy) >= obj_id:
            det = detections.xyxy[obj_id - 1]  # Indexing starts from 0
            bbox = np.array(det[:4]).astype(int)  # Convert to integer numpy array

            # Check if the bounding box coordinates have changed from the previous frame
            if prev_bbox_points[obj_id] is not None and np.array_equal(prev_bbox_points[obj_id], bbox):
                spin_status = 'Spin=False'
            else:
                spin_status = 'Spin=True'

            # Store current bounding box points for the next iteration
            prev_bbox_points[obj_id] = bbox

            # Extract bounding box coordinates
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

            # Check if any point of the bounding box is inside the ellipse
            bbox_points = [(bbox_x1, bbox_y1), (bbox_x2, bbox_y1), (bbox_x1, bbox_y2), (bbox_x2, bbox_y2)]
            inside_ellipse = any(cv2.pointPolygonTest(ellipse_points, (int(point[0]), int(point[1])), False) >= 0 for point in bbox_points)

            # Draw bounding box
            bbox_color = (0, 255, 0) if inside_ellipse else (0, 255, 255)
            cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), bbox_color, 2)

            # Store spin status and object ID information
            obj_info[obj_id] = {'Spin': spin_status, 'ID': obj_id}

            # Draw spin status and object ID text
            text = f"ID: {obj_id}, {spin_status}"
            cv2.putText(frame, text, (bbox_x1, bbox_y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("yolov8", frame)

    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''

'''
checkpoint 5 --> able to track battle duration, winner id and collision count(need some tuning)
import cv2
import numpy as np
import csv
import time
from ultralytics import YOLO
import supervision as sv

video_path = 'data/sample_2.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('fine-tune-yolo/best.pt')
box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=1
)

# Set the color of the ellipse to green
ellipse_color = (0, 255, 0)
# Static IDs for the two beyblades
static_ids = [1, 2]
# Dictionary to store previous bounding box points for each object
prev_bbox_points = {1: None, 2: None}

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS of the video: {fps}")

# Battle tracking variables
battle_start_frame = None
battle_end_frame = None
battle_ended = False
winner_id = None

# Collision counter
collision_count = 0

# Function to save battle results to CSV
def save_battle_results(battle_duration, winner_id, collision_count):
    with open('battle_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Battle Duration (s)', 'Winner ID', 'Collision Count'])
        writer.writerow([battle_duration, winner_id, collision_count])
    print(f"Battle results saved: Duration - {battle_duration}s, Winner ID - {winner_id}, Collisions - {collision_count}")

# Function to check if two bounding boxes overlap
def is_collision(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

# Initialize frame counter
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        print("Failed to capture frame")
        break

    # Set battle start frame at the first frame
    if battle_start_frame is None:
        battle_start_frame = frame_count

    # Get the dimensions of the frame
    height, width = frame.shape[:2]

    # Calculate the center of the frame
    center_x = width // 2
    center_y = height // 2

    # Define ellipse parameters
    axis_x = 480
    axis_y = 420
    angle = 0  # Angle in degrees
    ellipse_params = ((center_x, center_y), (axis_x, axis_y), angle)

    # Draw the ellipse
    cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), angle, 0, 360, ellipse_color, 2)

    # Get the ellipse points for pointPolygonTest
    ellipse_points = cv2.ellipse2Poly((center_x, center_y), (axis_x, axis_y), angle, 0, 360, 1).astype(np.int32)

    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Initialize a dictionary to store the spin status and object ID of each object
    obj_info = {}
    bboxes = []

    for obj_id in static_ids:
        # Check if the object has been detected
        if len(detections.xyxy) >= obj_id:
            det = detections.xyxy[obj_id - 1]  # Indexing starts from 0
            bbox = np.array(det[:4]).astype(int)  # Convert to integer numpy array
            bboxes.append(bbox)

            # Check if the bounding box coordinates have changed from the previous frame
            if prev_bbox_points[obj_id] is not None and np.array_equal(prev_bbox_points[obj_id], bbox):
                spin_status = 'Spin=False'
            else:
                spin_status = 'Spin=True'

            # Store current bounding box points for the next iteration
            prev_bbox_points[obj_id] = bbox

            # Extract bounding box coordinates
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

            # Check if any point of the bounding box is inside the ellipse
            bbox_points = [(bbox_x1, bbox_y1), (bbox_x2, bbox_y1), (bbox_x1, bbox_y2), (bbox_x2, bbox_y2)]
            inside_ellipse = any(cv2.pointPolygonTest(ellipse_points, (int(point[0]), int(point[1])), False) >= 0 for point in bbox_points)

            # Draw bounding box
            bbox_color = (0, 255, 0) if inside_ellipse else (0, 255, 255)
            cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), bbox_color, 2)

            # Store spin status and object ID information
            obj_info[obj_id] = {'Spin': spin_status, 'ID': obj_id, 'Inside': inside_ellipse}

            # Draw spin status and object ID text
            text = f"ID: {obj_id}, {spin_status}"
            cv2.putText(frame, text, (bbox_x1, bbox_y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Check if the battle has ended
            if not inside_ellipse or spin_status == 'Spin=False':
                battle_ended = True
                winner_id = static_ids[1] if obj_id == static_ids[0] else static_ids[0]
                battle_end_frame = frame_count
                break

    # Check for collisions
    if len(bboxes) == 2:
        if is_collision(bboxes[0], bboxes[1]):
            collision_count += 1

    if battle_ended:
        # Calculate battle duration in seconds using FPS
        battle_duration = (battle_end_frame - battle_start_frame) / fps
        save_battle_results(battle_duration, winner_id, collision_count)
        break

    cv2.imshow("yolov8", frame)

    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''


'''
checkpoint 4 (valid) --> able to track battle duration and winner id
import cv2
import numpy as np
import csv
import time
from ultralytics import YOLO
import supervision as sv

video_path = 'data/sample_2.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('fine-tune-yolo/best.pt')
box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=1
)

# Set the color of the ellipse to green
ellipse_color = (0, 255, 0)
# Static IDs for the two beyblades
static_ids = [1, 2]
# Dictionary to store previous bounding box points for each object
prev_bbox_points = {1: None, 2: None}

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS of the video: {fps}")

# Battle tracking variables
battle_start_frame = None
battle_end_frame = None
battle_ended = False
winner_id = None

# Function to save battle results to CSV
def save_battle_results(battle_duration, winner_id):
    with open('battle_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Battle Duration (s)', 'Winner ID'])
        writer.writerow([battle_duration, winner_id])
    print(f"Battle results saved: Duration - {battle_duration}s, Winner ID - {winner_id}")

# Initialize frame counter
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        print("Failed to capture frame")
        break

    # Set battle start frame at the first frame
    if battle_start_frame is None:
        battle_start_frame = frame_count

    # Get the dimensions of the frame
    height, width = frame.shape[:2]

    # define the position and shape of the ellipse
    center_x = width // 2
    center_y = height // 2
    axis_x = width // 2
    axis_y = height // 2
    angle = 0  
    ellipse_params = ((center_x, center_y), (axis_x, axis_y), angle)

    # Draw the ellipse
    cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), angle, 0, 360, ellipse_color, 2)

    # Get the ellipse points for pointPolygonTest
    ellipse_points = cv2.ellipse2Poly((center_x, center_y), (axis_x, axis_y), angle, 0, 360, 1).astype(np.int32)

    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Initialize a dictionary to store the spin status and object ID of each object
    obj_info = {}

    for obj_id in static_ids:
        # Check if the object has been detected
        if len(detections.xyxy) >= obj_id:
            det = detections.xyxy[obj_id - 1]  # Indexing starts from 0
            bbox = np.array(det[:4]).astype(int)  # Convert to integer numpy array

            # Check if the bounding box coordinates have changed from the previous frame
            if prev_bbox_points[obj_id] is not None and np.array_equal(prev_bbox_points[obj_id], bbox):
                spin_status = 'Spin=False'
            else:
                spin_status = 'Spin=True'

            # Store current bounding box points for the next iteration
            prev_bbox_points[obj_id] = bbox

            # Extract bounding box coordinates
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

            # Check if any point of the bounding box is inside the ellipse
            bbox_points = [(bbox_x1, bbox_y1), (bbox_x2, bbox_y1), (bbox_x1, bbox_y2), (bbox_x2, bbox_y2)]
            inside_ellipse = any(cv2.pointPolygonTest(ellipse_points, (int(point[0]), int(point[1])), False) >= 0 for point in bbox_points)

            # Draw bounding box
            bbox_color = (0, 255, 0) if inside_ellipse else (0, 255, 255)
            cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), bbox_color, 2)

            # Store spin status and object ID information
            obj_info[obj_id] = {'Spin': spin_status, 'ID': obj_id, 'Inside': inside_ellipse}

            # Draw spin status and object ID text
            text = f"ID: {obj_id}, {spin_status}"
            cv2.putText(frame, text, (bbox_x1, bbox_y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Check if the battle has ended
            if not inside_ellipse or spin_status == 'Spin=False':
                battle_ended = True
                winner_id = static_ids[1] if obj_id == static_ids[0] else static_ids[0]
                battle_end_frame = frame_count
                break

    if battle_ended:
        # Calculate battle duration in seconds using FPS
        battle_duration = (battle_end_frame - battle_start_frame) / fps
        save_battle_results(battle_duration, winner_id)
        break

    cv2.imshow("yolov8", frame)

    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''


'''
#checkpoint 6 VALID --> properly log the battle duration, winner id, and collision count
import cv2
import numpy as np
import csv
import time
from ultralytics import YOLO
import supervision as sv

video_path = 'data/sample_2.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('fine-tune-yolo/best.pt')
box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=1
)

# Set the color of the ellipse to green
ellipse_color = (0, 255, 0)
# Static IDs for the two beyblades
static_ids = [1, 2]
# Dictionary to store previous bounding box points for each object
prev_bbox_points = {1: None, 2: None}

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS of the video: {fps}")

# Battle tracking variables
battle_start_frame = None
battle_end_frame = None
battle_ended = False
winner_id = None

# Collision counter
collision_count = 0

prev_collision_state = False
# Function to save battle results to CSV
def save_battle_results(battle_duration, winner_id, collision_count):
    with open('battle_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Battle Duration (s)', 'Winner ID', 'Collision Count'])
        writer.writerow([battle_duration, winner_id, collision_count])
    print(f"Battle results saved: Duration - {battle_duration}s, Winner ID - {winner_id}, Collisions - {collision_count}")

# Function to check if two bounding boxes overlap
def is_collision(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

# Initialize frame counter
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        print("Failed to capture frame")
        break

    # Set battle start frame at the first frame
    if battle_start_frame is None:
        battle_start_frame = frame_count

    # Get the dimensions of the frame
    height, width = frame.shape[:2]

    # Calculate the center of the frame
    center_x = width // 2
    center_y = height // 2

    # Define ellipse parameters
    axis_x = 480
    axis_y = 420
    angle = 0  # Angle in degrees
    ellipse_params = ((center_x, center_y), (axis_x, axis_y), angle)

    # Draw the ellipse
    cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), angle, 0, 360, ellipse_color, 2)

    # Get the ellipse points for pointPolygonTest
    ellipse_points = cv2.ellipse2Poly((center_x, center_y), (axis_x, axis_y), angle, 0, 360, 1).astype(np.int32)

    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Initialize a dictionary to store the spin status and object ID of each object
    obj_info = {}
    bboxes = []

    for obj_id in static_ids:
        # Check if the object has been detected
        if len(detections.xyxy) >= obj_id:
            det = detections.xyxy[obj_id - 1]  # Indexing starts from 0
            bbox = np.array(det[:4]).astype(int)  # Convert to integer numpy array
            bboxes.append(bbox)

            # Check if the bounding box coordinates have changed from the previous frame
            if prev_bbox_points[obj_id] is not None and np.array_equal(prev_bbox_points[obj_id], bbox):
                spin_status = 'Spin=False'
            else:
                spin_status = 'Spin=True'

            # Store current bounding box points for the next iteration
            prev_bbox_points[obj_id] = bbox

            # Extract bounding box coordinates
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

            # Check if any point of the bounding box is inside the ellipse
            bbox_points = [(bbox_x1, bbox_y1), (bbox_x2, bbox_y1), (bbox_x1, bbox_y2), (bbox_x2, bbox_y2)]
            inside_ellipse = any(cv2.pointPolygonTest(ellipse_points, (int(point[0]), int(point[1])), False) >= 0 for point in bbox_points)

            # Draw bounding box
            bbox_color = (0, 255, 0) if inside_ellipse else (0, 255, 255)
            cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), bbox_color, 2)

            # Store spin status and object ID information
            obj_info[obj_id] = {'Spin': spin_status, 'ID': obj_id, 'Inside': inside_ellipse}

            # Draw spin status and object ID text
            text = f"ID: {obj_id}, {spin_status}"
            cv2.putText(frame, text, (bbox_x1, bbox_y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            if len(bboxes) == 2:
                if is_collision(bboxes[0], bboxes[1]):
                    if not prev_collision_state:
                        collision_count += 1
                    prev_collision_state = True
                else:
                    prev_collision_state = False

            # Check if the battle has ended
            if not inside_ellipse or spin_status == 'Spin=False':
                battle_ended = True
                winner_id = static_ids[1] if obj_id == static_ids[0] else static_ids[0]
                battle_end_frame = frame_count
                break

    if battle_ended:
        # Calculate battle duration in seconds using FPS
        battle_duration = (battle_end_frame - battle_start_frame) / fps
        save_battle_results(battle_duration, winner_id, collision_count)
        break

    cv2.imshow("yolov8", frame)

    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''



#checkpoint 7 VALID --> properly log the battle duration, winner id, win reason,and collision count
import cv2
import numpy as np
import csv
import time
from ultralytics import YOLO
import supervision as sv

video_path = 'sample_2.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('best.pt')
box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=1
)

# Set the color of the ellipse to green
ellipse_color = (0, 255, 0)
# Static IDs for the two beyblades
static_ids = [1, 2]
# Dictionary to store previous bounding box points for each object
prev_bbox_points = {1: None, 2: None}

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS of the video: {fps}")

# Battle tracking variables
battle_start_frame = None
battle_end_frame = None
battle_ended = False
winner_id = None
battle_stop_factor = None

# Collision counter
collision_count = 0

prev_collision_state = False

# Function to save battle results to CSV
def save_battle_results(battle_duration, winner_id, collision_count, battle_stop_factor):
    with open('battle_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Battle Duration (s)', 'Winner ID', 'Collision Count', 'Battle Stop Factor'])
        writer.writerow([battle_duration, winner_id, collision_count, battle_stop_factor])
    print(f"Battle results saved: Duration - {battle_duration}s, Winner ID - {winner_id}, Collisions - {collision_count}, Stop Factor - {battle_stop_factor}")

# Function to check if two bounding boxes overlap
def is_collision(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

# Function to draw an ellipse on the frame
def draw_ellipse(frame, color):
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    axis_x, axis_y, angle = 480, 420, 0
    cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), angle, 0, 360, color, 2)
    ellipse_points = cv2.ellipse2Poly((center_x, center_y), (axis_x, axis_y), angle, 0, 360, 1).astype(np.int32)
    return ellipse_points

# Initialize frame counter
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        print("Failed to capture frame")
        break

    # Set battle start frame at the first frame
    if battle_start_frame is None:
        battle_start_frame = frame_count

    # Draw the ellipse and get the ellipse points
    ellipse_points = draw_ellipse(frame, ellipse_color)

    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Initialize a dictionary to store the spin status and object ID of each object
    obj_info = {}
    bboxes = []

    for obj_id in static_ids:
        # Check if the object has been detected
        if len(detections.xyxy) >= obj_id:
            det = detections.xyxy[obj_id - 1]  # Indexing starts from 0
            bbox = np.array(det[:4]).astype(int)  # Convert to integer numpy array
            bboxes.append(bbox)

            # Check if the bounding box coordinates have changed from the previous frame
            if prev_bbox_points[obj_id] is not None and np.array_equal(prev_bbox_points[obj_id], bbox):
                spin_status = 'Spin=False'
            else:
                spin_status = 'Spin=True'

            # Store current bounding box points for the next iteration
            prev_bbox_points[obj_id] = bbox

            # Extract bounding box coordinates
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

            # Check if any point of the bounding box is inside the ellipse
            bbox_points = [(bbox_x1, bbox_y1), (bbox_x2, bbox_y1), (bbox_x1, bbox_y2), (bbox_x2, bbox_y2)]
            inside_ellipse = any(cv2.pointPolygonTest(ellipse_points, (int(point[0]), int(point[1])), False) >= 0 for point in bbox_points)

            # Draw bounding box
            bbox_color = (0, 255, 0) if inside_ellipse else (0, 255, 255)
            cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), bbox_color, 2)

            # Store spin status and object ID information
            obj_info[obj_id] = {'Spin': spin_status, 'ID': obj_id, 'Inside': inside_ellipse}

            # Draw spin status and object ID text
            text = f"ID: {obj_id}, {spin_status}"
            cv2.putText(frame, text, (bbox_x1, bbox_y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            if len(bboxes) == 2:
                if is_collision(bboxes[0], bboxes[1]):
                    if not prev_collision_state:
                        collision_count += 1
                    prev_collision_state = True
                else:
                    prev_collision_state = False

            # Check if the battle has ended
            if not inside_ellipse:
                battle_ended = True
                winner_id = static_ids[1] if obj_id == static_ids[0] else static_ids[0]
                battle_end_frame = frame_count
                battle_stop_factor = 'Exited Arena'
                break
            elif spin_status == 'Spin=False':
                battle_ended = True
                winner_id = static_ids[1] if obj_id == static_ids[0] else static_ids[0]
                battle_end_frame = frame_count
                battle_stop_factor = 'Spinning Stops'
                break

    if battle_ended:
        # Calculate battle duration in seconds using FPS
        battle_duration = (battle_end_frame - battle_start_frame) / fps
        save_battle_results(battle_duration, winner_id, collision_count, battle_stop_factor)
        break

    cv2.imshow("yolov8", frame)

    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()




