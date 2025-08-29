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

if __name__ == "__main__":
    initialize_database()