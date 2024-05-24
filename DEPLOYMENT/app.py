import cv2
import numpy as np
import csv
import tempfile
import streamlit as st
from ultralytics import YOLO
import supervision as sv
import pandas as pd

# Define a function to capture a frame from the video
def capture_frame(video_path):
    '''
    function to read the frames of the video
    '''
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return None

def save_battle_results(battle_duration, winner_id, collision_count, battle_stop_factor):
    '''
    Function to save battle results to CSV
    '''
    with open('battle_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Battle Duration (s)', 'Winner ID', 'Collision Count', 'Battle Stop Factor'])
        writer.writerow([battle_duration, winner_id, collision_count, battle_stop_factor])
    print(f"Battle results saved: Duration - {battle_duration}s, Winner ID - {winner_id}, Collisions - {collision_count}, Stop Factor - {battle_stop_factor}")

def is_collision(bbox1, bbox2):
    '''
    Function to check if two bounding boxes overlap
    '''
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

def draw_ellipse(frame, ellipse_params, color):
    '''
    Function to draw an ellipse on the frame
    '''
    center, axes, angle = ellipse_params
    axes = (int(axes[0]), int(axes[1]))
    cv2.ellipse(frame, center, axes, angle, 0, 360, color, 2)
    ellipse_points = cv2.ellipse2Poly(center, axes, angle, 0, 360, 1).astype(np.int32)
    return ellipse_points


def update_detection_script(video_path, ellipse_params, model):
    '''
    Function to update the detection script to take the ellipse 
    specified by the user as area of interest
    '''
    cap = cv2.VideoCapture(video_path)
    static_ids = [1, 2]
    prev_bbox_points = {1: None, 2: None}

    # count the fps to properly count the duration until battle ended
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of the video: {fps}")

    battle_start_frame = None
    battle_end_frame = None
    battle_ended = False
    winner_id = None
    battle_stop_factor = None

    collision_count = 0
    prev_collision_state = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        frame_count += 1

        if not ret:
            print("Failed to capture frame")
            break

        if battle_start_frame is None:
            battle_start_frame = frame_count

        ellipse_points = draw_ellipse(frame, ellipse_params, (0, 255, 0))

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)

        obj_info = {}
        bboxes = []

        for obj_id in static_ids:
            if len(detections.xyxy) >= obj_id:
                det = detections.xyxy[obj_id - 1]
                bbox = np.array(det[:4]).astype(int)
                bboxes.append(bbox)
                
                # check the spin status by comparing if there are no changes of the bounding box within 2 frame
                if prev_bbox_points[obj_id] is not None and np.array_equal(prev_bbox_points[obj_id], bbox):
                    spin_status = 'Spin=False'
                else:
                    spin_status = 'Spin=True'

                prev_bbox_points[obj_id] = bbox

                # check whether any of the object is within the area of interest
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
                bbox_points = [(bbox_x1, bbox_y1), (bbox_x2, bbox_y1), (bbox_x1, bbox_y2), (bbox_x2, bbox_y2)]
                inside_ellipse = any(cv2.pointPolygonTest(ellipse_points, (int(point[0]), int(point[1])), False) >= 0 for point in bbox_points)

                bbox_color = (0, 255, 0) if inside_ellipse else (0, 255, 255)
                cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), bbox_color, 2)

                # update the details of each beyblade
                obj_info[obj_id] = {'Spin': spin_status, 'ID': obj_id, 'Inside': inside_ellipse}
                text = f"ID: {obj_id}, {spin_status}"
                cv2.putText(frame, text, (bbox_x1, bbox_y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # check if collision between 2 beyblade occur
                if len(bboxes) == 2:
                    if is_collision(bboxes[0], bboxes[1]):
                        if not prev_collision_state:
                            collision_count += 1
                        prev_collision_state = True
                    else:
                        prev_collision_state = False

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

        # save battle result if battle end condition is met                
        if battle_ended:
            battle_duration = (battle_end_frame - battle_start_frame) / fps
            save_battle_results(battle_duration, winner_id, collision_count, battle_stop_factor)
            break

        cv2.imshow("yolov8", frame)

        if cv2.waitKey(20) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    st.title("Beyblade Battle Detection with Custom Area of Interest")

    video_file = st.file_uploader("Upload a video file either in mp4/avi format", type=["mp4", "avi"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
        
        # capture a single frame from the uploaded video
        frame = capture_frame(video_path)
        if frame is not None:
            st.write("Draw an ellipse on the image to define the area of interest.")
            drawing = st.image(frame, caption="Draw an ellipse", use_column_width=True, channels="BGR")
            
            # use sliders to let user define custom area of interest
            center_x = st.slider("Center X", min_value=0, max_value=frame.shape[1], step=1, value=frame.shape[1] // 2)
            center_y = st.slider("Center Y", min_value=0, max_value=frame.shape[0], step=1, value=frame.shape[0] // 2)
            axis_x = st.slider("Axis X", min_value=1, max_value=frame.shape[1], step=1, value=frame.shape[1] // 4)
            axis_y = st.slider("Axis Y", min_value=1, max_value=frame.shape[0], step=1, value=frame.shape[0] // 4)
            angle = st.slider("Angle", min_value=0, max_value=360, step=1, value=0)

            ellipse_params = ((center_x, center_y), (axis_x, axis_y), angle)
            center, axes, angle = ellipse_params
            axes = (int(axes[0]), int(axes[1]))
            cv2.ellipse(frame, center, axes, angle, 0, 360, (0, 255, 0), 2)
            drawing.image(frame, caption="Draw an ellipse", use_column_width=True, channels="BGR")

            if st.button("Process Video"):
                model = YOLO('best.pt')
                update_detection_script(video_path, ellipse_params, model)
                st.write("Battle results saved to battle_results.csv")

                # Display the CSV file content
                df = pd.read_csv('battle_results.csv')
                st.dataframe(df)

if __name__ == "__main__":
    main()
