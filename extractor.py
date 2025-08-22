import cv2
import mediapipe as mp
import numpy as np
import json
import os
import pandas as pd
from typing import List, Dict, Any, Tuple
import math
from datetime import datetime
import logging
from config import MEDIAPIPE_CONFIG, FRAME_PROCESSING_CONFIG

class SignLanguageSkeletonExtractor:
    def __init__(self):
        # MediaPipe 초기화
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

        # 더 안정적인 MediaPipe 설정 사용
        mediapipe_config = MEDIAPIPE_CONFIG.copy()
        # 세그멘테이션 관련 문제를 방지하기 위해 세그멘테이션 비활성화
        mediapipe_config['enable_segmentation'] = False

        try:
            self.holistic = self.mp_holistic.Holistic(**mediapipe_config)
        except Exception as e:
            logging.error(f"MediaPipe 초기화 실패, 세그멘테이션 없이 재시도: {e}")
            # 세그멘테이션 없이 재시도
            mediapipe_config['enable_segmentation'] = False
            self.holistic = self.mp_holistic.Holistic(**mediapipe_config)

        # 프레임 크기 추적 변수들
        self.expected_frame_size = None
        self.frame_size_tolerance = FRAME_PROCESSING_CONFIG['frame_size_tolerance']
        self.max_consecutive_errors = FRAME_PROCESSING_CONFIG['max_consecutive_errors']

        # 데이터 저장을 위한 리스트 초기화
        self.reset_data()

    def validate_frame(self, frame: np.ndarray) -> Tuple[bool, str]:
        """프레임 유효성 검사 및 크기 일관성 확인"""
        if frame is None or frame.size == 0:
            return False, "프레임이 비어있거나 유효하지 않습니다"

        height, width = frame.shape[:2]

        # 첫 번째 프레임인 경우 기준 크기 설정
        if self.expected_frame_size is None:
            self.expected_frame_size = (height, width)
            return True, "기준 프레임 크기 설정됨"

        # 프레임 크기 일관성 확인
        expected_height, expected_width = self.expected_frame_size
        height_diff = abs(height - expected_height) / expected_height
        width_diff = abs(width - expected_width) / expected_width

        if height_diff > self.frame_size_tolerance or width_diff > self.frame_size_tolerance:
            return False, f"프레임 크기 불일치: 예상 {self.expected_frame_size}, 실제 ({height}, {width})"

        return True, "프레임 유효함"

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임을 일관된 크기로 조정"""
        if self.expected_frame_size is None:
            return frame

        expected_height, expected_width = self.expected_frame_size

        # 보간 방법 설정
        interpolation = cv2.INTER_LINEAR
        if FRAME_PROCESSING_CONFIG['resize_interpolation'] == 'cubic':
            interpolation = cv2.INTER_CUBIC
        elif FRAME_PROCESSING_CONFIG['resize_interpolation'] == 'nearest':
            interpolation = cv2.INTER_NEAREST

        return cv2.resize(frame, (expected_width, expected_height), interpolation=interpolation)

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """프레임 전처리 (노이즈 제거, 밝기 조정 등)"""
        try:
            # 프레임이 비어있거나 유효하지 않은 경우
            if frame is None or frame.size == 0:
                return frame, False

            # 프레임 크기 검증 및 조정
            is_valid, validation_message = self.validate_frame(frame)
            if not is_valid:
                logging.warning(f"프레임 전처리 중: {validation_message}")
                frame = self.resize_frame(frame)

            # 밝기와 대비 조정 (선택적)
            # 프레임이 너무 어둡거나 밝은 경우 조정
            if frame.dtype == np.uint8:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)

                # 밝기 조정 (너무 어두운 경우)
                if np.mean(l) < 50:
                    l = cv2.add(l, 30)
                elif np.mean(l) > 200:
                    l = cv2.subtract(l, 20)

                lab = cv2.merge([l, a, b])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            return frame, True

        except Exception as e:
            logging.error(f"프레임 전처리 중 오류: {e}")
            return frame, False

    def reset_data(self):
        """데이터 저장 변수들 초기화"""
        self.frame_data = []
        self.pose_landmarks_history = []
        self.face_landmarks_history = []
        self.left_hand_landmarks_history = []
        self.right_hand_landmarks_history = []
        # 프레임 크기 추적 변수도 초기화
        self.expected_frame_size = None
    
    def extract_landmarks(self, image: np.ndarray) -> Dict[str, Any]:
        """이미지에서 랜드마크 추출"""
        try:
            # 프레임 전처리
            processed_image, success = self.preprocess_frame(image)
            if not success:
                logging.warning("프레임 전처리 실패, 원본 프레임 사용")
                processed_image = image

            # BGR을 RGB로 변환
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

            # MediaPipe 처리
            results = self.holistic.process(rgb_image)

        except Exception as e:
            logging.error(f"랜드마크 추출 중 오류 발생: {e}")
            # 오류 발생 시 빈 결과 반환
            return {
                'pose': None,
                'face': None,
                'left_hand': None,
                'right_hand': None
            }
        
        landmarks_data = {
            'pose': None,
            'face': None,
            'left_hand': None,
            'right_hand': None
        }
        
        # 포즈 랜드마크 추출 (33개 점)
        if results.pose_landmarks:
            landmarks_data['pose'] = [
                [lm.x, lm.y, lm.z, lm.visibility] 
                for lm in results.pose_landmarks.landmark
            ]
        
        # 얼굴 랜드마크 추출 (468개 점)
        if results.face_landmarks:
            landmarks_data['face'] = [
                [lm.x, lm.y, lm.z] 
                for lm in results.face_landmarks.landmark
            ]
        
        # 왼손 랜드마크 추출 (21개 점)
        if results.left_hand_landmarks:
            landmarks_data['left_hand'] = [
                [lm.x, lm.y, lm.z] 
                for lm in results.left_hand_landmarks.landmark
            ]
        
        # 오른손 랜드마크 추출 (21개 점)
        if results.right_hand_landmarks:
            landmarks_data['right_hand'] = [
                [lm.x, lm.y, lm.z] 
                for lm in results.right_hand_landmarks.landmark
            ]
        
        return landmarks_data
    
    def calculate_velocity(self, current_landmarks: List[List[float]], 
                          previous_landmarks: List[List[float]], 
                          fps: float) -> List[List[float]]:
        """관절 속도 계산"""
        if not current_landmarks or not previous_landmarks:
            return None
        
        velocities = []
        dt = 1.0 / fps  # 시간 간격
        
        for i, (curr, prev) in enumerate(zip(current_landmarks, previous_landmarks)):
            if len(curr) >= 3 and len(prev) >= 3:  # x, y, z 좌표가 있는 경우
                vx = (curr[0] - prev[0]) / dt
                vy = (curr[1] - prev[1]) / dt
                vz = (curr[2] - prev[2]) / dt
                magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
                velocities.append([vx, vy, vz, magnitude])
            else:
                velocities.append([0.0, 0.0, 0.0, 0.0])
        
        return velocities
    
    def calculate_acceleration(self, current_velocity: List[List[float]], 
                             previous_velocity: List[List[float]], 
                             fps: float) -> List[List[float]]:
        """관절 가속도 계산"""
        if not current_velocity or not previous_velocity:
            return None
        
        accelerations = []
        dt = 1.0 / fps
        
        for curr_vel, prev_vel in zip(current_velocity, previous_velocity):
            if len(curr_vel) >= 4 and len(prev_vel) >= 4:
                ax = (curr_vel[0] - prev_vel[0]) / dt
                ay = (curr_vel[1] - prev_vel[1]) / dt
                az = (curr_vel[2] - prev_vel[2]) / dt
                magnitude = math.sqrt(ax**2 + ay**2 + az**2)
                accelerations.append([ax, ay, az, magnitude])
            else:
                accelerations.append([0.0, 0.0, 0.0, 0.0])
        
        return accelerations
    
    def normalize_landmarks(self, landmarks: List[List[float]], 
                           reference_point: List[float] = None) -> List[List[float]]:
        """랜드마크 정규화 (수어에 특화된 정규화)"""
        if not landmarks:
            return None
        
        normalized = []
        
        # 어깨 중심점을 기준점으로 사용 (포즈의 경우)
        if reference_point is None and len(landmarks) > 12:  # 포즈 랜드마크인 경우
            # 양쪽 어깨의 중점을 기준점으로 설정
            left_shoulder = landmarks[11]  # 왼쪽 어깨
            right_shoulder = landmarks[12]  # 오른쪽 어깨
            reference_point = [
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2,
                (left_shoulder[2] + right_shoulder[2]) / 2
            ]
        elif reference_point is None:
            # 손목을 기준점으로 사용 (손 랜드마크인 경우)
            reference_point = landmarks[0] if landmarks else [0, 0, 0]
        
        # 각 랜드마크를 기준점 대비 상대 위치로 정규화
        for landmark in landmarks:
            if len(landmark) >= 3:
                norm_x = landmark[0] - reference_point[0]
                norm_y = landmark[1] - reference_point[1] 
                norm_z = landmark[2] - reference_point[2]
                
                # 거리 기반 정규화
                distance = math.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
                
                normalized_landmark = [norm_x, norm_y, norm_z, distance]
                if len(landmark) > 3:  # visibility 정보가 있는 경우
                    normalized_landmark.append(landmark[3])
                
                normalized.append(normalized_landmark)
            else:
                normalized.append(landmark)
        
        return normalized
    
    def extract_hand_features(self, hand_landmarks: List[List[float]]) -> Dict[str, Any]:
        """손 모양 및 움직임 특징 추출"""
        if not hand_landmarks:
            return None
        
        features = {
            'finger_angles': [],
            'hand_openness': 0.0,
            'palm_direction': [0.0, 0.0, 0.0],
            'finger_spread': 0.0
        }
        
        # 손가락 각도 계산 (각 손가락의 굽힘 정도)
        finger_tip_indices = [4, 8, 12, 16, 20]  # 엄지, 검지, 중지, 약지, 소지 끝
        finger_pip_indices = [3, 6, 10, 14, 18]  # 각 손가락의 PIP 관절
        
        for tip_idx, pip_idx in zip(finger_tip_indices, finger_pip_indices):
            if tip_idx < len(hand_landmarks) and pip_idx < len(hand_landmarks):
                tip = hand_landmarks[tip_idx]
                pip = hand_landmarks[pip_idx]
                wrist = hand_landmarks[0]
                
                # 손목-PIP-끝 각도 계산
                vec1 = [pip[0] - wrist[0], pip[1] - wrist[1]]
                vec2 = [tip[0] - pip[0], tip[1] - pip[1]]
                
                dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
                magnitude1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
                magnitude2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
                
                if magnitude1 > 0 and magnitude2 > 0:
                    angle = math.acos(max(-1, min(1, dot_product / (magnitude1 * magnitude2))))
                    features['finger_angles'].append(math.degrees(angle))
                else:
                    features['finger_angles'].append(0.0)
        
        # 손 벌림 정도 계산
        if len(hand_landmarks) > 20:
            distances = []
            for i in range(1, len(finger_tip_indices)):
                p1 = hand_landmarks[finger_tip_indices[i-1]]
                p2 = hand_landmarks[finger_tip_indices[i]]
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                distances.append(dist)
            features['finger_spread'] = np.mean(distances) if distances else 0.0
        
        return features
    
    def extract_face_features(self, face_landmarks: List[List[float]]) -> Dict[str, Any]:
        """얼굴 표정 특징 추출"""
        if not face_landmarks:
            return None
        
        features = {
            'eyebrow_height': [0.0, 0.0],  # 왼쪽, 오른쪽 눈썹 높이
            'eye_openness': [0.0, 0.0],   # 왼쪽, 오른쪽 눈 열림 정도
            'mouth_openness': 0.0,        # 입 열림 정도
            'mouth_width': 0.0,           # 입 너비
        }
        
        # 눈썹 높이 (대략적인 인덱스 사용)
        if len(face_landmarks) > 70:
            left_eyebrow = face_landmarks[70]
            right_eyebrow = face_landmarks[107]
            nose_bridge = face_landmarks[8]
            
            features['eyebrow_height'][0] = nose_bridge[1] - left_eyebrow[1]
            features['eyebrow_height'][1] = nose_bridge[1] - right_eyebrow[1]
        
        # 입 특징 (대략적인 인덱스 사용)
        if len(face_landmarks) > 300:
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            left_mouth = face_landmarks[61]
            right_mouth = face_landmarks[291]
            
            features['mouth_openness'] = abs(upper_lip[1] - lower_lip[1])
            features['mouth_width'] = abs(left_mouth[0] - right_mouth[0])
        
        return features
    
    def classify_sign_properties(self, pose_landmarks: List[List[float]], 
                               left_hand: List[List[float]], 
                               right_hand: List[List[float]]) -> Dict[str, Any]:
        """수어 언어학적 속성 분류"""
        properties = {
            'hand_usage': 'unknown',      # 'one_hand', 'two_hands', 'no_hands'
            'movement_type': 'unknown',   # 'circular', 'linear', 'static'
            'repetition': False,          # 반복 동작 여부
            'sign_type': 'unknown'        # 'simple', 'compound'
        }
        
        # 손 사용 분류
        left_active = left_hand is not None
        right_active = right_hand is not None
        
        if left_active and right_active:
            properties['hand_usage'] = 'two_hands'
        elif left_active or right_active:
            properties['hand_usage'] = 'one_hand'
        else:
            properties['hand_usage'] = 'no_hands'
        
        # 움직임 타입은 시계열 분석이 필요하므로 여기서는 기본값 유지
        
        return properties
    
    def process_video(self, video_path: str, output_path: str = None) -> Dict[str, Any]:
        """비디오 처리 메인 함수"""
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        # 비디오 정보 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"{video_path} | FPS: {fps} | 총 프레임: {total_frames}")
        
        self.reset_data()
        frame_idx = 0
        
        # 이전 프레임 데이터 저장용 변수
        prev_landmarks = None
        prev_velocities = None
        
        consecutive_errors = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임이 비어있는지 확인
            if frame is None or frame.size == 0:
                logging.warning(f"빈 프레임 발견 (프레임 {frame_idx}), 건너뜀")
                frame_idx += 1
                continue

            try:
                # 랜드마크 추출
                current_landmarks = self.extract_landmarks(frame)
                consecutive_errors = 0  # 성공 시 오류 카운트 리셋

            except Exception as e:
                consecutive_errors += 1
                logging.error(f"프레임 {frame_idx} 처리 중 오류: {e}")

                if consecutive_errors >= self.max_consecutive_errors:
                    logging.error(f"연속 {self.max_consecutive_errors}프레임 오류로 인해 비디오 처리 중단")
                    break

                # 오류 발생 시 빈 랜드마크로 계속 진행
                current_landmarks = {
                    'pose': None,
                    'face': None,
                    'left_hand': None,
                    'right_hand': None
                }
            
            # 프레임 데이터 초기화
            frame_data = {
                'frame_index': frame_idx,
                'timestamp': frame_idx / fps,
                'landmarks': {},
                'normalized_landmarks': {},
                'velocities': {},
                'accelerations': {},
                'hand_features': {},
                'face_features': {},
                'sign_properties': {}
            }
            
            # 각 부위별 처리
            for part in ['pose', 'face', 'left_hand', 'right_hand']:
                landmarks = current_landmarks[part]
                
                if landmarks:
                    # 원본 랜드마크 저장
                    frame_data['landmarks'][part] = landmarks
                    
                    # 정규화된 랜드마크 저장
                    normalized = self.normalize_landmarks(landmarks)
                    frame_data['normalized_landmarks'][part] = normalized
                    
                    # 속도 계산 (이전 프레임이 있는 경우)
                    if prev_landmarks and prev_landmarks[part]:
                        velocity = self.calculate_velocity(
                            landmarks, prev_landmarks[part], fps
                        )
                        frame_data['velocities'][part] = velocity
                        
                        # 가속도 계산 (이전 속도가 있는 경우)
                        if (prev_velocities and prev_velocities.get(part) and 
                            velocity):
                            acceleration = self.calculate_acceleration(
                                velocity, prev_velocities[part], fps
                            )
                            frame_data['accelerations'][part] = acceleration
            
            # 손 특징 추출
            if current_landmarks['left_hand']:
                frame_data['hand_features']['left'] = self.extract_hand_features(
                    current_landmarks['left_hand']
                )
            
            if current_landmarks['right_hand']:
                frame_data['hand_features']['right'] = self.extract_hand_features(
                    current_landmarks['right_hand']
                )
            
            # 얼굴 특징 추출
            if current_landmarks['face']:
                frame_data['face_features'] = self.extract_face_features(
                    current_landmarks['face']
                )
            
            # 수어 속성 분류
            frame_data['sign_properties'] = self.classify_sign_properties(
                current_landmarks['pose'],
                current_landmarks['left_hand'],
                current_landmarks['right_hand']
            )
            
            # 프레임 데이터 저장
            self.frame_data.append(frame_data)
            
            # 이전 프레임 데이터 업데이트
            prev_landmarks = current_landmarks.copy()
            prev_velocities = frame_data['velocities'].copy()
            
            frame_idx += 1
        
        cap.release()
        
        # 결과 데이터 구성
        result_data = {
            'video_info': {
                'path': video_path,
                'fps': fps,
                'total_frames': total_frames,
                'duration': total_frames / fps,
                'processing_date': datetime.now().isoformat()
            },
            'frame_data': self.frame_data,
            'statistics': self._calculate_statistics()
        }
        
        # 결과 저장
        if output_path:
            self.save_data(result_data, output_path)
        
        return result_data
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """처리된 데이터의 통계 정보 계산"""
        if not self.frame_data:
            return {}
        
        stats = {
            'total_frames': len(self.frame_data),
            'frames_with_pose': 0,
            'frames_with_face': 0,
            'frames_with_left_hand': 0,
            'frames_with_right_hand': 0,
            'average_hand_velocity': {'left': 0.0, 'right': 0.0},
            'hand_usage_distribution': {'one_hand': 0, 'two_hands': 0, 'no_hands': 0}
        }
        
        total_left_velocity = []
        total_right_velocity = []
        
        for frame in self.frame_data:
            # 각 부위 검출 프레임 수 계산
            if frame['landmarks'].get('pose'):
                stats['frames_with_pose'] += 1
            if frame['landmarks'].get('face'):
                stats['frames_with_face'] += 1
            if frame['landmarks'].get('left_hand'):
                stats['frames_with_left_hand'] += 1
            if frame['landmarks'].get('right_hand'):
                stats['frames_with_right_hand'] += 1
            
            # 손 사용 분포 계산
            hand_usage = frame['sign_properties'].get('hand_usage', 'unknown')
            if hand_usage in stats['hand_usage_distribution']:
                stats['hand_usage_distribution'][hand_usage] += 1
            
            # 속도 데이터 수집
            if frame['velocities'].get('left_hand'):
                velocities = [v[3] for v in frame['velocities']['left_hand'] if len(v) > 3]
                total_left_velocity.extend(velocities)
            
            if frame['velocities'].get('right_hand'):
                velocities = [v[3] for v in frame['velocities']['right_hand'] if len(v) > 3]
                total_right_velocity.extend(velocities)
        
        # 평균 속도 계산
        if total_left_velocity:
            stats['average_hand_velocity']['left'] = np.mean(total_left_velocity)
        if total_right_velocity:
            stats['average_hand_velocity']['right'] = np.mean(total_right_velocity)
        
        return stats
    
    def save_data(self, data: Dict[str, Any], output_path: str):
        """데이터를 JSON 파일로 저장"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                        
        except Exception as e:
            logging.error(f"데이터 저장 실패: {e}")
    
    def load_data(self, input_path: str) -> Dict[str, Any]:
        """JSON 파일에서 데이터 로드"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logging.info(f"데이터 로드 완료: {input_path}")
            return data
            
        except Exception as e:
            logging.error(f"데이터 로드 실패: {e}")
            return {} 