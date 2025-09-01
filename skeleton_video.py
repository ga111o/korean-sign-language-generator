import psycopg2
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Dict, Optional, Tuple
import logging


# MediaPipe Pose 랜드마크 연결선 정의 (33개 포인트)
POSE_CONNECTIONS = [
    # 얼굴 부분
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # 어깨와 팔
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (16, 18), (17, 19), (18, 20), (15, 21), (16, 22),
    # 몸통
    (11, 23), (12, 24), (23, 24),
    # 다리
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

# 손 랜드마크 연결선 정의 (21개 포인트)
HAND_CONNECTIONS = [
    # 엄지
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 검지
    (0, 5), (5, 6), (6, 7), (7, 8),
    # 중지
    (0, 9), (9, 10), (10, 11), (11, 12),
    # 약지
    (0, 13), (13, 14), (14, 15), (15, 16),
    # 소지
    (0, 17), (17, 18), (18, 19), (19, 20)
]

class SkeletonVideoVisualizer:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.frames_data = []
        
    def get_db_connection(self):
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logging.error(f"DB connection failed: {e}")
            raise
    
    def fetch_video_frames(self, video_id: int) -> List[Dict]:
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT 
            frame_index, 
            timestamp,
            pose_landmarks, 
            face_landmarks, 
            left_hand_landmarks, 
            right_hand_landmarks,
            normalized_pose_landmarks,
            normalized_left_hand_landmarks,
            normalized_right_hand_landmarks
        FROM video_frames 
        WHERE video_id = %s 
        ORDER BY frame_index ASC
        """
        
        cursor.execute(query, (video_id,))
        rows = cursor.fetchall()
        
        frames = []
        for row in rows:
            frame_data = {
                'frame_index': row[0],
                'timestamp': row[1],
                'pose_landmarks': pickle.loads(row[2]) if row[2] else None,
                'face_landmarks': pickle.loads(row[3]) if row[3] else None,
                'left_hand_landmarks': pickle.loads(row[4]) if row[4] else None,
                'right_hand_landmarks': pickle.loads(row[5]) if row[5] else None,
                'normalized_pose_landmarks': pickle.loads(row[6]) if row[6] else None,
                'normalized_left_hand_landmarks': pickle.loads(row[7]) if row[7] else None,
                'normalized_right_hand_landmarks': pickle.loads(row[8]) if row[8] else None,
            }
            frames.append(frame_data)
        
        cursor.close()
        conn.close()
        
        self.frames_data = frames
        logging.info(f"video_id {len(frames)}of {video_id}")
        return frames
    
    def draw_skeleton_2d(self, frame_data: Dict, image_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        width, height = image_size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # pose landmark draw
        if frame_data['pose_landmarks']:
            pose_points = []
            for landmark in frame_data['pose_landmarks']:
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                pose_points.append((x, y))
                # join point draw
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
            
            # skeleton connection draw
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if (start_idx < len(pose_points) and end_idx < len(pose_points) and
                    pose_points[start_idx][0] > 0 and pose_points[start_idx][1] > 0 and
                    pose_points[end_idx][0] > 0 and pose_points[end_idx][1] > 0):
                    cv2.line(image, pose_points[start_idx], pose_points[end_idx], (255, 255, 0), 2)
        
        if frame_data['left_hand_landmarks']:
            hand_points = []
            for landmark in frame_data['left_hand_landmarks']:
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                hand_points.append((x, y))
                cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
            
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if (start_idx < len(hand_points) and end_idx < len(hand_points) and
                    hand_points[start_idx][0] > 0 and hand_points[start_idx][1] > 0 and
                    hand_points[end_idx][0] > 0 and hand_points[end_idx][1] > 0):
                    cv2.line(image, hand_points[start_idx], hand_points[end_idx], (255, 0, 255), 1)
        
        if frame_data['right_hand_landmarks']:
            hand_points = []
            for landmark in frame_data['right_hand_landmarks']:
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                hand_points.append((x, y))
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if (start_idx < len(hand_points) and end_idx < len(hand_points) and
                    hand_points[start_idx][0] > 0 and hand_points[start_idx][1] > 0 and
                    hand_points[end_idx][0] > 0 and hand_points[end_idx][1] > 0):
                    cv2.line(image, hand_points[start_idx], hand_points[end_idx], (0, 255, 255), 1)
        
        cv2.putText(image, f"Frame: {frame_data['frame_index']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"Time: {frame_data['timestamp']:.2f}s", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return image
    
    def create_skeleton_video(self, output_path: str, fps: float = 30.0, image_size: Tuple[int, int] = (640, 480)):
        if not self.frames_data:
            raise ValueError("not self.frames_data")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, image_size)
        
        for i, frame_data in enumerate(self.frames_data):
            skeleton_image = self.draw_skeleton_2d(frame_data, image_size)
            out.write(skeleton_image)
            
            if i % 100 == 0:
                print(f"processing: {i}/{len(self.frames_data)} frames")
        
        out.release()
        print(f"skeleton video saved: {output_path}")
    
    def visualize_3d_skeleton(self, frame_index: int = 0):
        if not self.frames_data:
            raise ValueError("not self.frames_data")
        
        if frame_index >= len(self.frames_data):
            frame_index = 0
        
        frame_data = self.frames_data[frame_index]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if frame_data['pose_landmarks']:
            pose_landmarks = frame_data['pose_landmarks']
            
            # join point
            xs = [lm[0] for lm in pose_landmarks]
            ys = [lm[1] for lm in pose_landmarks]  
            zs = [lm[2] for lm in pose_landmarks]
            ax.scatter(xs, ys, zs, c='red', s=50, alpha=0.7)
            
            # skeleton connection
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                    x_vals = [pose_landmarks[start_idx][0], pose_landmarks[end_idx][0]]
                    y_vals = [pose_landmarks[start_idx][1], pose_landmarks[end_idx][1]]
                    z_vals = [pose_landmarks[start_idx][2], pose_landmarks[end_idx][2]]
                    ax.plot(x_vals, y_vals, z_vals, 'b-', linewidth=2)
        
        # hand landmark 3D draw
        for hand_landmarks, color, label in [
            (frame_data['left_hand_landmarks'], 'green', 'Left Hand'),
            (frame_data['right_hand_landmarks'], 'blue', 'Right Hand')
        ]:
            if hand_landmarks:
                xs = [lm[0] for lm in hand_landmarks]
                ys = [lm[1] for lm in hand_landmarks]
                zs = [lm[2] for lm in hand_landmarks]
                ax.scatter(xs, ys, zs, c=color, s=30, alpha=0.7, label=label)
                
                # hand connection
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                        x_vals = [hand_landmarks[start_idx][0], hand_landmarks[end_idx][0]]
                        y_vals = [hand_landmarks[start_idx][1], hand_landmarks[end_idx][1]]
                        z_vals = [hand_landmarks[start_idx][2], hand_landmarks[end_idx][2]]
                        ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(f'3D Skeleton - Frame {frame_data["frame_index"]}')
        ax.legend()
        
        plt.show()
    
    def play_skeleton_realtime(self, fps: float = 30.0, image_size: Tuple[int, int] = (640, 480)):
        if not self.frames_data:
            raise ValueError("not self.frames_data")
        
        frame_delay = int(1000 / fps)  # milliseconds
        
        for frame_data in self.frames_data:
            skeleton_image = self.draw_skeleton_2d(frame_data, image_size)
            cv2.imshow('Skeleton Visualization', skeleton_image)
            
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    VIDEO_ID = 1255

    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', '5432')
    }

    visualizer = SkeletonVideoVisualizer(db_config)
    
    video_id = VIDEO_ID
    frames = visualizer.fetch_video_frames(video_id)
    
    if frames:
        # 1. skeleton video file create
        visualizer.create_skeleton_video(f'skeleton_video_{video_id}.mp4', fps=30.0)
        
        # 2. 3D skeleton visualize (first frame)
        visualizer.visualize_3d_skeleton(frame_index=0)
        
        # 3. realtime play (OpenCV window)
        visualizer.play_skeleton_realtime(fps=30.0)
    else:
        print(f"no frames")

if __name__ == "__main__":
    main()
