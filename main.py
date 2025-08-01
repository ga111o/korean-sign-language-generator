import os
import json
import logging
from extractor import SignLanguageSkeletonExtractor
from utils import (
    setup_logging, load_csv_data, filter_front_data, find_video_files,
    get_processable_files, create_metadata, log_processing_info,
    log_statistics, log_completion_info
)

def main():
    # 로깅 설정
    log_filename = setup_logging()
    
    # 스켈레톤 추출기 초기화
    extractor = SignLanguageSkeletonExtractor()
    
    # CSV 파일 읽기
    csv_path = "./signlanguage_total.csv"
    df = load_csv_data(csv_path)
    
    # "정면" 데이터만 필터링
    front_data = filter_front_data(df)
    
    # dataset 폴더 경로 설정
    dataset_path = "./dataset"
    output_dir = "output"
    
    # dataset 폴더에서 모든 비디오 파일 찾기
    video_files = find_video_files(dataset_path)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 정면 데이터에서 처리할 수 있는 파일 찾기
    processable_files = get_processable_files(front_data, video_files, output_dir)
    
    # 각 비디오 파일 처리
    for i, file_info in enumerate(processable_files, 1):
        row_data = file_info['row_data']
        video_path = file_info['video_path']
        base_name = file_info['base_name']
        
        log_processing_info(file_info, i, len(processable_files))
        
        # 출력 파일명 생성
        output_path = os.path.join(output_dir, f"{base_name}_skeleton_data.json")
        
        try:
            # 비디오 처리 및 데이터 추출
            result_data = extractor.process_video(video_path, output_path)
            
            # 메타데이터를 최상단에 영어로 추가
            metadata = create_metadata(row_data)
            
            # 메타데이터를 최상단에 배치
            result_data_with_metadata = {
                'metadata': metadata,
                'video_info': result_data['video_info'],
                'frame_data': result_data['frame_data'],
                'statistics': result_data['statistics']
            }
            
            # 메타데이터가 포함된 결과를 다시 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data_with_metadata, f, ensure_ascii=False, indent=2)
            
            # 통계 정보 출력
            stats = result_data['statistics']
            log_statistics(stats)
            
        except Exception as e:
            logging.error(f"  오류 발생: {e}")
            continue
    
    log_completion_info(processable_files, output_dir, log_filename)

if __name__ == "__main__":
    main()
