import os
import json
import pickle
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# Load environment variables
load_dotenv()

def get_db_connection():
    """Create and return database connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT', '5432')
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

def create_tables():
    """Create tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 메인 비디오 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sign_language_videos (
                id SERIAL PRIMARY KEY,
                video_id INTEGER NOT NULL UNIQUE,
                provider_id INTEGER,
                acquisition_year SMALLINT,
                direction VARCHAR(10),
                type VARCHAR(30),
                filename VARCHAR(100),
                korean_text VARCHAR(200),

                -- Video Information
                video_path VARCHAR(100),
                fps REAL,
                total_frames INTEGER,
                duration REAL,
                processing_date TIMESTAMP,
                gpu_accelerated BOOLEAN,
                gpu_processing BOOLEAN,

                -- Statistics
                stats_total_frames INTEGER,
                frames_with_pose INTEGER,
                frames_with_face INTEGER,
                frames_with_left_hand INTEGER,
                frames_with_right_hand INTEGER,
                avg_left_hand_velocity REAL,
                avg_right_hand_velocity REAL,

                -- Hand usage distribution
                hand_usage_one_hand INTEGER,
                hand_usage_two_hands INTEGER,
                hand_usage_no_hands INTEGER,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 프레임별 상세 데이터 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_frames (
                id SERIAL PRIMARY KEY,
                video_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                timestamp REAL NOT NULL,

                -- 모든 랜드마크 데이터
                pose_landmarks BYTEA,
                face_landmarks BYTEA,
                left_hand_landmarks BYTEA,
                right_hand_landmarks BYTEA,
                normalized_pose_landmarks BYTEA,
                normalized_face_landmarks BYTEA,
                normalized_left_hand_landmarks BYTEA,
                normalized_right_hand_landmarks BYTEA,
                velocities BYTEA,

                -- Face features
                eyebrow_height_left REAL,
                eyebrow_height_right REAL,
                eye_openness_left REAL,
                eye_openness_right REAL,
                mouth_openness REAL,
                mouth_width REAL,

                -- Sign properties
                hand_usage VARCHAR(20),
                movement_type VARCHAR(20),
                repetition BOOLEAN,
                sign_type VARCHAR(20),

                -- 외래키
                FOREIGN KEY (video_id) REFERENCES sign_language_videos(video_id)
            )
        """)

        conn.commit()
        print("Tables created successfully")

    except Exception as e:
        conn.rollback()
        print(f"Error creating tables: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

# 테이블 존재 여부 확인 및 생성
def initialize_database():
    """Initialize database by creating tables if they don't exist"""
    try:
        create_tables()
    except Exception as e:
        print(f"Database initialization error: {e}")
        raise

def compress_landmarks(landmarks_data):
    """Compress landmarks data to binary format using pickle"""
    if not landmarks_data:
        return None
    return pickle.dumps(landmarks_data)

def parse_json_file(json_file_path):
    """Parse JSON file and extract data for database insertion"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract metadata
    metadata = data['metadata']
    video_info = data['video_info']
    statistics = data['statistics']
    frame_data = data['frame_data']
    
    # Prepare video data
    video_data = {
        'video_id': metadata['id'],
        'provider_id': metadata['provider_id'],
        'acquisition_year': metadata['acquisition_year'],
        'direction': metadata['direction'],
        'type': metadata['type'],
        'filename': metadata['filename'],
        'korean_text': metadata['korean_text'],
        'video_path': video_info['path'],
        'fps': video_info['fps'],
        'total_frames': video_info['total_frames'],
        'duration': video_info['duration'],
        'processing_date': datetime.fromisoformat(video_info['processing_date'].replace('Z', '+00:00')),
        'gpu_accelerated': video_info['gpu_accelerated'],
        'gpu_processing': video_info['gpu_processing'],
        'stats_total_frames': statistics['total_frames'],
        'frames_with_pose': statistics['frames_with_pose'],
        'frames_with_face': statistics['frames_with_face'],
        'frames_with_left_hand': statistics['frames_with_left_hand'],
        'frames_with_right_hand': statistics['frames_with_right_hand'],
        'avg_left_hand_velocity': statistics['average_hand_velocity']['left'],
        'avg_right_hand_velocity': statistics['average_hand_velocity']['right'],
        'hand_usage_one_hand': statistics['hand_usage_distribution']['one_hand'],
        'hand_usage_two_hands': statistics['hand_usage_distribution']['two_hands'],
        'hand_usage_no_hands': statistics['hand_usage_distribution']['no_hands']
    }
    
    # Prepare frame data
    frames = []
    for frame in frame_data:
        frame_dict = {
            'video_id': metadata['id'],
            'frame_index': frame['frame_index'],
            'timestamp': frame['timestamp'],
            'pose_landmarks': compress_landmarks(frame['landmarks'].get('pose')),
            'face_landmarks': compress_landmarks(frame['landmarks'].get('face')),
            'left_hand_landmarks': compress_landmarks(frame['landmarks'].get('left_hand')),
            'right_hand_landmarks': compress_landmarks(frame['landmarks'].get('right_hand')),
            'normalized_pose_landmarks': compress_landmarks(frame['normalized_landmarks'].get('pose')),
            'normalized_face_landmarks': compress_landmarks(frame['normalized_landmarks'].get('face')),
            'normalized_left_hand_landmarks': compress_landmarks(frame['normalized_landmarks'].get('left_hand')),
            'normalized_right_hand_landmarks': compress_landmarks(frame['normalized_landmarks'].get('right_hand')),
            'velocities': compress_landmarks(frame.get('velocities')),
            'eyebrow_height_left': frame['face_features']['eyebrow_height'][0],
            'eyebrow_height_right': frame['face_features']['eyebrow_height'][1],
            'eye_openness_left': frame['face_features']['eye_openness'][0],
            'eye_openness_right': frame['face_features']['eye_openness'][1],
            'mouth_openness': frame['face_features']['mouth_openness'],
            'mouth_width': frame['face_features']['mouth_width'],
            'hand_usage': frame['sign_properties']['hand_usage'],
            'movement_type': frame['sign_properties']['movement_type'],
            'repetition': frame['sign_properties']['repetition'],
            'sign_type': frame['sign_properties']['sign_type']
        }
        frames.append(frame_dict)
    
    return video_data, frames

def insert_video_data(video_data):
    """Insert video data into sign_language_videos table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        insert_query = """
            INSERT INTO sign_language_videos (
                video_id, provider_id, acquisition_year, direction, type, filename, korean_text,
                video_path, fps, total_frames, duration, processing_date, gpu_accelerated, gpu_processing,
                stats_total_frames, frames_with_pose, frames_with_face, frames_with_left_hand, frames_with_right_hand,
                avg_left_hand_velocity, avg_right_hand_velocity, hand_usage_one_hand, hand_usage_two_hands, hand_usage_no_hands
            ) VALUES (
                %(video_id)s, %(provider_id)s, %(acquisition_year)s, %(direction)s, %(type)s, %(filename)s, %(korean_text)s,
                %(video_path)s, %(fps)s, %(total_frames)s, %(duration)s, %(processing_date)s, %(gpu_accelerated)s, %(gpu_processing)s,
                %(stats_total_frames)s, %(frames_with_pose)s, %(frames_with_face)s, %(frames_with_left_hand)s, %(frames_with_right_hand)s,
                %(avg_left_hand_velocity)s, %(avg_right_hand_velocity)s, %(hand_usage_one_hand)s, %(hand_usage_two_hands)s, %(hand_usage_no_hands)s
            )
            ON CONFLICT (video_id) DO UPDATE SET
                provider_id = EXCLUDED.provider_id,
                acquisition_year = EXCLUDED.acquisition_year,
                direction = EXCLUDED.direction,
                type = EXCLUDED.type,
                filename = EXCLUDED.filename,
                korean_text = EXCLUDED.korean_text,
                video_path = EXCLUDED.video_path,
                fps = EXCLUDED.fps,
                total_frames = EXCLUDED.total_frames,
                duration = EXCLUDED.duration,
                processing_date = EXCLUDED.processing_date,
                gpu_accelerated = EXCLUDED.gpu_accelerated,
                gpu_processing = EXCLUDED.gpu_processing,
                stats_total_frames = EXCLUDED.stats_total_frames,
                frames_with_pose = EXCLUDED.frames_with_pose,
                frames_with_face = EXCLUDED.frames_with_face,
                frames_with_left_hand = EXCLUDED.frames_with_left_hand,
                frames_with_right_hand = EXCLUDED.frames_with_right_hand,
                avg_left_hand_velocity = EXCLUDED.avg_left_hand_velocity,
                avg_right_hand_velocity = EXCLUDED.avg_right_hand_velocity,
                hand_usage_one_hand = EXCLUDED.hand_usage_one_hand,
                hand_usage_two_hands = EXCLUDED.hand_usage_two_hands,
                hand_usage_no_hands = EXCLUDED.hand_usage_no_hands
        """
        
        cursor.execute(insert_query, video_data)
        conn.commit()
        print(f"Video data inserted/updated for video_id: {video_data['video_id']}")
        
    except Exception as e:
        conn.rollback()
        print(f"Error inserting video data: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def insert_frame_data(frames):
    """Insert frame data into video_frames table"""
    if not frames:
        return
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Delete existing frames for this video_id first
        video_id = frames[0]['video_id']
        cursor.execute("DELETE FROM video_frames WHERE video_id = %s", (video_id,))
        
        insert_query = """
            INSERT INTO video_frames (
                video_id, frame_index, timestamp,
                pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks,
                normalized_pose_landmarks, normalized_face_landmarks, normalized_left_hand_landmarks, normalized_right_hand_landmarks,
                velocities,
                eyebrow_height_left, eyebrow_height_right, eye_openness_left, eye_openness_right,
                mouth_openness, mouth_width,
                hand_usage, movement_type, repetition, sign_type
            ) VALUES (
                %(video_id)s, %(frame_index)s, %(timestamp)s,
                %(pose_landmarks)s, %(face_landmarks)s, %(left_hand_landmarks)s, %(right_hand_landmarks)s,
                %(normalized_pose_landmarks)s, %(normalized_face_landmarks)s, %(normalized_left_hand_landmarks)s, %(normalized_right_hand_landmarks)s,
                %(velocities)s,
                %(eyebrow_height_left)s, %(eyebrow_height_right)s, %(eye_openness_left)s, %(eye_openness_right)s,
                %(mouth_openness)s, %(mouth_width)s,
                %(hand_usage)s, %(movement_type)s, %(repetition)s, %(sign_type)s
            )
        """
        
        cursor.executemany(insert_query, frames)
        conn.commit()
        print(f"Inserted {len(frames)} frames for video_id: {video_id}")
        
    except Exception as e:
        conn.rollback()
        print(f"Error inserting frame data: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def process_json_file(json_file_path):
    """Process a single JSON file and insert data into database"""
    try:
        print(f"Processing: {json_file_path}")
        video_data, frames = parse_json_file(json_file_path)
        
        # Insert video data
        insert_video_data(video_data)
        
        # Insert frame data
        insert_frame_data(frames)
        
        print(f"Successfully processed: {json_file_path}")
        
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        raise

def process_all_json_files(directory_path="/mnt/f/signlanguage/output/"):
    """Process all JSON files in the specified directory"""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory does not exist: {directory_path}")
        return
    
    json_files = list(directory.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in: {directory_path}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Initialize database first
    initialize_database()
    
    processed = 0
    failed = 0
    
    for json_file in json_files:
        try:
            process_json_file(json_file)
            processed += 1
        except Exception as e:
            print(f"Failed to process {json_file}: {e}")
            failed += 1
    
    print(f"\nProcessing completed:")
    print(f"  Successfully processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(json_files)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific file
        json_file_path = sys.argv[1]
        initialize_database()
        process_json_file(json_file_path)
    else:
        # Process all files in default directory
        process_all_json_files()
