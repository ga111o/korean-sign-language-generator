import os
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any

def setup_logging():
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = os.path.join('.', 'logs', 'extractor')
    os.makedirs(log_dir, exist_ok=True)
    # 로그 파일명 생성 (현재 시간 기준)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"sign_language_processing_{timestamp}.log")
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 콘솔 출력도 유지
        ]
    )
    return log_filename

def load_csv_data(csv_path: str) -> pd.DataFrame:
    """CSV 파일 로드"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"오류: {csv_path} 파일이 존재하지 않습니다.")
    
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"CSV 파일 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        logging.error(f"CSV 파일 읽기 실패: {e}")
        raise

def filter_front_data(df: pd.DataFrame) -> pd.DataFrame:
    """정면 데이터만 필터링"""
    front_data = df[df['방향'] == '정면'].copy()
    logging.info(f"정면 데이터: {len(front_data)}개")
    
    if len(front_data) == 0:
        raise ValueError("정면 데이터가 없습니다.")
    
    return front_data

def find_video_files(dataset_path: str) -> Dict[str, str]:
    """dataset 폴더에서 비디오 파일 찾기"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"오류: {dataset_path} 폴더가 존재하지 않습니다.")
    
    # 지원하는 비디오 파일 확장자
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mts', '.MOV', '.MTS']
    
    video_files = {}
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in [ext.lower() for ext in video_extensions]:
                # 확장자를 제외한 파일명
                base_name = os.path.splitext(filename)[0]
                video_files[base_name] = file_path
    
    logging.info(f"dataset 폴더의 비디오 파일: {len(video_files)}개")
    return video_files

def get_processable_files(front_data: pd.DataFrame, video_files: Dict[str, str], output_dir: str) -> List[Dict[str, Any]]:
    """처리 가능한 파일 목록 생성"""
    processable_files = []
    
    for idx, row in front_data.iterrows():
        filename = row['파일명']
        base_name = os.path.splitext(filename)[0]
        
        if base_name in video_files:
            # 이미 JSON 파일이 존재하는지 확인
            output_path = os.path.join(output_dir, f"{base_name}_skeleton_data.json")
            if os.path.exists(output_path):
                logging.info(f"이미 처리된 파일 건너뛰기: {filename}")
                continue
                
            processable_files.append({
                'row_data': row,
                'video_path': video_files[base_name],
                'base_name': base_name
            })
        else:
            logging.warning(f"파일을 찾을 수 없음: {filename}")
    
    logging.info(f"처리 가능한 정면 비디오: {len(processable_files)}개")
    
    if len(processable_files) == 0:
        raise ValueError("처리할 수 있는 정면 비디오가 없습니다.")
    
    return processable_files

def create_metadata(row_data: pd.Series) -> Dict[str, Any]:
    """메타데이터 생성"""
    return {
        'id': int(row_data['번호']),
        'provider_id': int(row_data['언어 제공자 ID']),
        'acquisition_year': int(row_data['취득연도']),
        'direction': row_data['방향'],
        'type': row_data['타입(단어/문장)'],
        'filename': row_data['파일명'],
        'korean_text': row_data['한국어']
    }

def log_processing_info(file_info: Dict[str, Any], index: int, total: int):
    """처리 정보 로깅"""
    row_data = file_info['row_data']
    video_path = file_info['video_path']
    
    logging.info(f"\n====================== 비디오 {index}/{total} 처리 중 ======================")
    logging.info(f"  번호: {row_data['번호']}")
    logging.info(f"  언어 제공자 ID: {row_data['언어 제공자 ID']}")
    logging.info(f"  타입: {row_data['타입(단어/문장)']}")
    logging.info(f"  한국어: {row_data['한국어']}")
    logging.info(f"  파일: {os.path.basename(video_path)}")

def log_statistics(stats: Dict[str, Any]):
    """통계 정보 로깅"""
    logging.info(f"  포즈: {stats['frames_with_pose']} | 얼굴: {stats['frames_with_face']} | 왼손: {stats['frames_with_left_hand']} | 오른손: {stats['frames_with_right_hand']}")
    logging.info(f"  손 사용 분포: {stats['hand_usage_distribution']}")
    logging.info(f"  평균 손 속도: 왼손: {stats['average_hand_velocity']['left']:.4f} | 오른손: {stats['average_hand_velocity']['right']:.4f}")

def log_completion_info(processable_files: List[Dict[str, Any]], output_dir: str, log_filename: str):
    """완료 정보 로깅"""
    logging.info(f"\n====================== done! ======================")
    success_count = len([f for f in processable_files if os.path.exists(os.path.join(output_dir, f"{f['base_name']}_skeleton_data.json"))])
    logging.info(f"성공적으로 처리된 파일: {success_count}개")
    logging.info(f"출력 디렉토리: {output_dir}")
    logging.info(f"로그 파일: {log_filename}")
    
    print(f"\n처리 완료! 로그 파일이 생성되었습니다: {log_filename}") 