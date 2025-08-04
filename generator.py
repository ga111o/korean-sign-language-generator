import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import math
from torch.nn.utils.rnn import pad_sequence
import cv2
from tqdm import tqdm
import logging
from datetime import datetime

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging():
    """Setup logging configuration for generator."""
    # Create logs directory if it doesn't exist
    log_dir = './logs/generator/'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'generator_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

# Initialize logger
logger = setup_logging()

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

class SignLanguageDataset(Dataset):
    """Dataset class for sign language data with extracted landmarks - Fixed version."""
    
    def __init__(self, json_dir: str, max_frames: int = 150, max_text_length: int = 128):
        self.json_dir = json_dir
        self.max_frames = max_frames
        self.max_text_length = max_text_length
        self.data_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        # Landmark dimensions
        self.pose_dim = 33 * 4
        self.face_dim = 468 * 3
        self.hand_dim = 21 * 3
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        
        # Compute normalization statistics
        self._compute_normalization_stats()
        
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization with improved stability."""
        logger.info("Computing normalization statistics...")
        
        pose_data = []
        face_data = []
        hand_data = []
        
        # Sample a subset for statistics computation
        sample_size = min(100, len(self.data_files))
        
        for i in range(sample_size):
            json_path = os.path.join(self.json_dir, self.data_files[i])
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                frames = data['frame_data'][:self.max_frames]
                
                for frame in frames:
                    # Collect pose data
                    pose_landmarks = frame['landmarks'].get('pose', [])
                    if pose_landmarks:
                        pose_flat = [coord for landmark in pose_landmarks for coord in landmark]
                        if len(pose_flat) >= self.pose_dim:
                            pose_data.append(pose_flat[:self.pose_dim])
                    
                    # Collect face data
                    face_landmarks = frame['landmarks'].get('face', [])
                    if face_landmarks:
                        face_flat = [coord for landmark in face_landmarks for coord in landmark[:3]]
                        if len(face_flat) >= self.face_dim:
                            face_data.append(face_flat[:self.face_dim])
                    
                    # Collect hand data
                    for hand_key in ['left_hand', 'right_hand']:
                        hand_landmarks = frame['landmarks'].get(hand_key, [])
                        if hand_landmarks:
                            hand_flat = [coord for landmark in hand_landmarks for coord in landmark[:3]]
                            if len(hand_flat) >= self.hand_dim:
                                hand_data.append(hand_flat[:self.hand_dim])
                                
            except Exception as e:
                logger.error(f"Error processing {self.data_files[i]}: {e}")
                continue
        
        # Compute statistics with improved numerical stability
        if pose_data:
            pose_array = np.array(pose_data, dtype=np.float32)
            # Clip extreme values before computing statistics
            pose_array = np.clip(pose_array, -10, 10)
            self.pose_mean = np.mean(pose_array, axis=0)
            self.pose_std = np.std(pose_array, axis=0) + 1e-4  # Increased epsilon
            # Ensure std is not too small
            self.pose_std = np.maximum(self.pose_std, 0.1)
        else:
            self.pose_mean = np.zeros(self.pose_dim, dtype=np.float32)
            self.pose_std = np.ones(self.pose_dim, dtype=np.float32)
            
        if face_data:
            face_array = np.array(face_data, dtype=np.float32)
            face_array = np.clip(face_array, -10, 10)
            self.face_mean = np.mean(face_array, axis=0)
            self.face_std = np.std(face_array, axis=0) + 1e-4
            self.face_std = np.maximum(self.face_std, 0.1)
        else:
            self.face_mean = np.zeros(self.face_dim, dtype=np.float32)
            self.face_std = np.ones(self.face_dim, dtype=np.float32)
            
        if hand_data:
            hand_array = np.array(hand_data, dtype=np.float32)
            hand_array = np.clip(hand_array, -10, 10)
            self.hand_mean = np.mean(hand_array, axis=0)
            self.hand_std = np.std(hand_array, axis=0) + 1e-4
            self.hand_std = np.maximum(self.hand_std, 0.1)
        else:
            self.hand_mean = np.zeros(self.hand_dim, dtype=np.float32)
            self.hand_std = np.ones(self.hand_dim, dtype=np.float32)
        
        logger.info("Normalization statistics computed with improved stability.")
    
    def _normalize_data(self, data, mean, std):
        """Normalize data using computed statistics with clipping."""
        normalized = (data - mean) / std
        # Clip normalized values to prevent extreme values
        normalized = np.clip(normalized, -5, 5)
        return normalized
    
    def _validate_tensor(self, tensor, name):
        """Validate tensor for NaN/Inf values with more aggressive fixing."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"NaN/Inf found in {name}, replacing with zeros")
            tensor = torch.zeros_like(tensor)
        # Additional clipping for safety
        tensor = torch.clamp(tensor, -10, 10)
        return tensor
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        json_path = os.path.join(self.json_dir, self.data_files[idx])
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        try:
            text = data['metadata']['korean_text']
        except:
            logger.warning(f"No text found for {self.data_files[idx]}")
            text = "빈 텍스트"
        
        frames = data['frame_data']
        
        # Pre-allocate numpy arrays instead of lists - MAJOR FIX
        max_len = min(len(frames), self.max_frames)
        pose_sequence = np.zeros((max_len, self.pose_dim), dtype=np.float32)
        face_sequence = np.zeros((max_len, self.face_dim), dtype=np.float32)
        left_hand_sequence = np.zeros((max_len, self.hand_dim), dtype=np.float32)
        right_hand_sequence = np.zeros((max_len, self.hand_dim), dtype=np.float32)
        
        for i, frame in enumerate(frames[:max_len]):
            # Extract and normalize pose landmarks
            pose_landmarks = frame['landmarks'].get('pose', [])
            if pose_landmarks:
                pose_flat = [coord for landmark in pose_landmarks for coord in landmark]
                pose_flat = pose_flat[:self.pose_dim] + [0] * max(0, self.pose_dim - len(pose_flat))
            else:
                pose_flat = [0] * self.pose_dim
            
            pose_flat = np.array(pose_flat[:self.pose_dim], dtype=np.float32)
            pose_normalized = self._normalize_data(pose_flat, self.pose_mean, self.pose_std)
            pose_sequence[i] = pose_normalized
            
            # Extract and normalize face landmarks
            face_landmarks = frame['landmarks'].get('face', [])
            if face_landmarks:
                face_flat = [coord for landmark in face_landmarks for coord in landmark[:3]]
                face_flat = face_flat[:self.face_dim] + [0] * max(0, self.face_dim - len(face_flat))
            else:
                face_flat = [0] * self.face_dim
            
            face_flat = np.array(face_flat[:self.face_dim], dtype=np.float32)
            face_normalized = self._normalize_data(face_flat, self.face_mean, self.face_std)
            face_sequence[i] = face_normalized
            
            # Extract and normalize hand landmarks
            for hand_key, hand_seq in [('left_hand', left_hand_sequence), ('right_hand', right_hand_sequence)]:
                hand_landmarks = frame['landmarks'].get(hand_key, [])
                if hand_landmarks:
                    hand_flat = [coord for landmark in hand_landmarks for coord in landmark[:3]]
                    hand_flat = hand_flat[:self.hand_dim] + [0] * max(0, self.hand_dim - len(hand_flat))
                else:
                    hand_flat = [0] * self.hand_dim
                
                hand_flat = np.array(hand_flat[:self.hand_dim], dtype=np.float32)
                hand_normalized = self._normalize_data(hand_flat, self.hand_mean, self.hand_std)
                hand_seq[i] = hand_normalized
        
        # Convert to tensors directly from numpy arrays - MAJOR FIX
        pose_tensor = self._validate_tensor(torch.from_numpy(pose_sequence), "pose")
        face_tensor = self._validate_tensor(torch.from_numpy(face_sequence), "face")
        left_hand_tensor = self._validate_tensor(torch.from_numpy(left_hand_sequence), "left_hand")
        right_hand_tensor = self._validate_tensor(torch.from_numpy(right_hand_sequence), "right_hand")
        
        # Tokenize text
        text_tokens = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'text_tokens': text_tokens,
            'pose': pose_tensor,
            'face': face_tensor,
            'left_hand': left_hand_tensor,
            'right_hand': right_hand_tensor,
            'metadata': data['metadata'],
            'statistics': data['statistics']
        }

# =============================================================================
# Attention Mechanisms
# =============================================================================

class StarAttention(nn.Module):
    """Star attention mechanism for feature enhancement."""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
        # Star node parameters
        self.star_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.star_key = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Expand star nodes
        star_q = self.star_query.expand(batch_size, 1, -1)
        star_k = self.star_key.expand(batch_size, 1, -1)
        
        # Compute queries, keys, values - Fixed to maintain consistent dimensions
        q = self.query(torch.cat([star_q, x], dim=1))
        k = self.key(torch.cat([star_k, x], dim=1))
        v = self.value(torch.cat([star_k, x], dim=1))  # Changed from [x, x] to [star_k, x]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len + 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len + 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len + 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len + 1, self.d_model
        )
        
        # Remove star node from output and return original sequence
        output = self.output(attn_output[:, 1:, :])
        
        return output


class MotorAttention(nn.Module):
    """Motor attention for fine-grained movement control - Fixed version."""
    
    def __init__(self, d_model: int, feature_dim: int):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        
        # Project motor signals to match feature dimension if needed
        self.motor_projection = nn.Linear(d_model, feature_dim) if d_model != feature_dim else nn.Identity()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, features, motor_signals):
        # Project motor signals to feature space
        motor_projected = self.motor_projection(motor_signals)
        
        # Apply motor attention
        attn_output, _ = self.attention(features, motor_projected, motor_projected)
        
        # Residual connection and normalization
        output = self.norm(features + attn_output)
        
        return output

# =============================================================================
# Core Model Components
# =============================================================================

class SemanticUnderstandingModule(nn.Module):
    """LLM-enhanced text encoder for semantic understanding."""
    
    def __init__(self, model_name: str = 'klue/bert-base', d_model: int = 512):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert_model.config.hidden_size, d_model)
        self.semantic_decomposer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=3
        )
        
    def forward(self, text_tokens):
        # BERT encoding
        bert_output = self.bert_model(**text_tokens)
        text_features = bert_output.last_hidden_state
        
        # Project to model dimension
        projected_features = self.projection(text_features)
        
        # Semantic decomposition
        semantic_features = self.semantic_decomposer(projected_features)
        
        return semantic_features

class HierarchicalMotionPlanner(nn.Module):
    """Hierarchical motion planning network - Fixed version."""
    
    def __init__(self, d_model: int, pose_dim: int, face_dim: int, hand_dim: int):
        super().__init__()
        self.d_model = d_model
        
        # Global sequence planner
        self.global_planner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=6
        )
        
        # Shared CNN layers
        self.conv1 = nn.Conv1d(d_model, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Separate LSTM layers for each modality
        self.pose_lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.face_lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.left_hand_lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.right_hand_lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, bidirectional=True)
        
        # Temporal coherence module
        self.temporal_lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, bidirectional=True)
        self.temporal_projection = nn.Linear(d_model * 2, d_model)
        
        # Motor attention modules - Fixed to use LSTM output dimension (128)
        self.pose_motor_attention = MotorAttention(d_model, 128)
        self.face_motor_attention = MotorAttention(d_model, 128) 
        self.hand_motor_attention = MotorAttention(d_model, 128)
    
    def _apply_cnn_lstm(self, features, lstm_layer):
        """Apply CNN-BiLSTM processing to features."""
        # features: (batch, d_model, seq_len)
        x = self.relu1(self.conv1(features))
        x = self.relu2(self.conv2(x))
        
        # Transpose for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # Apply LSTM and return only the output
        lstm_output, _ = lstm_layer(x)
        return lstm_output
    
    def forward(self, semantic_features, target_length: int):
        batch_size = semantic_features.shape[0]
        
        # Global planning
        global_plan = self.global_planner(semantic_features)
        
        # Temporal modeling
        temporal_features, _ = self.temporal_lstm(global_plan)
        temporal_features = self.temporal_projection(temporal_features)
        
        # Interpolate to target length
        if temporal_features.shape[1] != target_length:
            temporal_features = F.interpolate(
                temporal_features.transpose(1, 2),
                size=target_length,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # Prepare features for CNN processing: (batch, features, seq_len)
        cnn_input = temporal_features.transpose(1, 2)
        
        # Apply CNN-BiLSTM for each modality
        pose_refined = self._apply_cnn_lstm(cnn_input, self.pose_lstm)
        face_refined = self._apply_cnn_lstm(cnn_input, self.face_lstm)
        left_hand_refined = self._apply_cnn_lstm(cnn_input, self.left_hand_lstm)
        right_hand_refined = self._apply_cnn_lstm(cnn_input, self.right_hand_lstm)
        
        # Apply motor attention - Fixed to use refined features as both input and motor signals
        pose_output = self.pose_motor_attention(pose_refined, temporal_features)
        face_output = self.face_motor_attention(face_refined, temporal_features)
        left_hand_output = self.hand_motor_attention(left_hand_refined, temporal_features)
        right_hand_output = self.hand_motor_attention(right_hand_refined, temporal_features)
        
        return {
            'pose': pose_output,
            'face': face_output,
            'left_hand': left_hand_output,
            'right_hand': right_hand_output,
            'global_features': temporal_features
        }


class MultiKeypointGenerator(nn.Module):
    """Multi-keypoint generation framework with fixed tensor shapes."""
    
    def __init__(self, d_model: int, pose_dim: int, face_dim: int, hand_dim: int):
        super().__init__()
        
        self.d_model = d_model
        
        # Add projection layers to map from LSTM output (128) to d_model (512)
        self.pose_projection = nn.Linear(128, d_model)
        self.face_projection = nn.Linear(128, d_model)
        self.left_hand_projection = nn.Linear(128, d_model)
        self.right_hand_projection = nn.Linear(128, d_model)
        
        # Output projection layers
        self.pose_head = nn.Linear(d_model, pose_dim)
        self.face_head = nn.Linear(d_model, face_dim)
        self.left_hand_head = nn.Linear(d_model, hand_dim)
        self.right_hand_head = nn.Linear(d_model, hand_dim)
        
        # Hierarchical importance weighting
        self.importance_weights = nn.Parameter(torch.ones(4))  # pose, face, left_hand, right_hand
        
        # Star attention for feature enhancement
        self.star_attention = StarAttention(d_model)
        
    def forward(self, motion_features):
        # Project features from 128 to d_model (512) before star attention
        pose_projected = self.pose_projection(motion_features['pose'])  # [batch, seq_len, d_model]
        face_projected = self.face_projection(motion_features['face'])
        left_hand_projected = self.left_hand_projection(motion_features['left_hand'])
        right_hand_projected = self.right_hand_projection(motion_features['right_hand'])
        
        # Apply star attention to enhance features
        pose_features = self.star_attention(pose_projected)  # [batch, seq_len, d_model]
        face_features = self.star_attention(face_projected)
        left_hand_features = self.star_attention(left_hand_projected)
        right_hand_features = self.star_attention(right_hand_projected)
        
        # Get batch and sequence dimensions
        batch_size, seq_len, d_model = pose_features.shape
        
        # Flatten tensors for linear layers (time-distributed application)
        pose_features_flat = pose_features.reshape(batch_size * seq_len, d_model)
        face_features_flat = face_features.reshape(batch_size * seq_len, d_model)
        left_hand_features_flat = left_hand_features.reshape(batch_size * seq_len, d_model)
        right_hand_features_flat = right_hand_features.reshape(batch_size * seq_len, d_model)
        
        # Apply linear transformations
        pose_keypoints_flat = self.pose_head(pose_features_flat)  # [batch*seq_len, pose_dim]
        face_keypoints_flat = self.face_head(face_features_flat)
        left_hand_keypoints_flat = self.left_hand_head(left_hand_features_flat)
        right_hand_keypoints_flat = self.right_hand_head(right_hand_features_flat)
        
        # Reshape back to [batch, seq_len, output_dim]
        pose_keypoints = pose_keypoints_flat.view(batch_size, seq_len, -1)
        face_keypoints = face_keypoints_flat.view(batch_size, seq_len, -1)
        left_hand_keypoints = left_hand_keypoints_flat.view(batch_size, seq_len, -1)
        right_hand_keypoints = right_hand_keypoints_flat.view(batch_size, seq_len, -1)
        
        # Apply hierarchical importance weighting
        weights = F.softmax(self.importance_weights, dim=0)
        
        return {
            'pose': pose_keypoints * weights[0],
            'face': face_keypoints * weights[1],
            'left_hand': left_hand_keypoints * weights[2],
            'right_hand': right_hand_keypoints * weights[3],
            'weights': weights
        }


class ContrastiveLearningModule(nn.Module):
    """Multi-positive contrastive learning module."""
    
    def __init__(self, d_model: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2)
        )
        
    def forward(self, features, labels=None):
        # Project features
        projected = self.projector(features.mean(dim=1))  # Global pooling
        projected = F.normalize(projected, dim=-1)
        
        if labels is not None:
            # Compute contrastive loss
            similarity_matrix = torch.matmul(projected, projected.T) / self.temperature
            
            # Create positive mask (same labels)
            positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            positive_mask.fill_diagonal_(0)  # Remove self-similarity
            
            # Compute InfoNCE loss
            exp_sim = torch.exp(similarity_matrix)
            pos_sim = exp_sim * positive_mask
            neg_sim = exp_sim * (1 - positive_mask)
            
            loss = -torch.log(pos_sim.sum(dim=1) / (pos_sim.sum(dim=1) + neg_sim.sum(dim=1) + 1e-8))
            loss = loss.mean()
            
            return projected, loss
        
        return projected, None

# =============================================================================
# Main MHAG Model
# =============================================================================

class MHAGModel(nn.Module):
    """Multi-Modal Hierarchical Attention Generator (MHAG) - Fixed version."""
    
    def __init__(self, 
                 d_model: int = 512,
                 pose_dim: int = 132,
                 face_dim: int = 1404,
                 hand_dim: int = 63):
        super().__init__()
        
        self.d_model = d_model
        self.pose_dim = pose_dim
        self.face_dim = face_dim
        self.hand_dim = hand_dim
        
        # Core modules
        self.semantic_module = SemanticUnderstandingModule(d_model=d_model)
        self.motion_planner = HierarchicalMotionPlanner(d_model, pose_dim, face_dim, hand_dim)
        self.keypoint_generator = MultiKeypointGenerator(d_model, pose_dim, face_dim, hand_dim)
        self.contrastive_module = ContrastiveLearningModule(d_model)
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Apply proper initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights properly to prevent NaN."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller gain for stability
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0, std=0.02)
    
    def forward(self, text_tokens, target_length: int, labels=None):
        # Semantic understanding
        semantic_features = self.semantic_module(text_tokens)
        
        # Validate semantic features
        if torch.isnan(semantic_features).any():
            logger.warning("NaN in semantic features")
            semantic_features = torch.nan_to_num(semantic_features, nan=0.0)
        
        # Cross-modal attention alignment
        aligned_features, _ = self.cross_modal_attention(
            semantic_features, semantic_features, semantic_features
        )
        
        # Validate aligned features
        if torch.isnan(aligned_features).any():
            logger.warning("NaN in aligned features")
            aligned_features = torch.nan_to_num(aligned_features, nan=0.0)
        
        # Hierarchical motion planning
        motion_features = self.motion_planner(aligned_features, target_length)
        
        # Validate motion features
        for key, value in motion_features.items():
            if torch.isnan(value).any():
                logger.warning(f"NaN in motion features - {key}")
                motion_features[key] = torch.nan_to_num(value, nan=0.0)
        
        # Multi-keypoint generation
        keypoint_outputs = self.keypoint_generator(motion_features)
        
        # Validate keypoint outputs
        for key, value in keypoint_outputs.items():
            if key != 'weights' and torch.isnan(value).any():
                logger.warning(f"NaN in keypoint outputs - {key}")
                keypoint_outputs[key] = torch.nan_to_num(value, nan=0.0)
        
        # Contrastive learning
        contrastive_features, contrastive_loss = self.contrastive_module(
            motion_features['global_features'], labels
        )
        
        return {
            'keypoints': keypoint_outputs,
            'semantic_features': semantic_features,
            'motion_features': motion_features,
            'contrastive_features': contrastive_features,
            'contrastive_loss': contrastive_loss
        }

# =============================================================================
# Training and Inference
# =============================================================================

class MHAGTrainer:
    """Training pipeline for MHAG model - Fixed version with better stability."""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Use even smaller learning rate for stability
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-5,  # Reduced further
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Use cosine annealing instead of OneCycleLR for stability
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-7
        )

    def compute_reconstruction_loss(self, predicted, target):
        """Compute MSE loss for keypoint reconstruction with enhanced stability."""
        losses = {}
        total_loss = 0
        valid_modalities = 0
        
        for modality in ['pose', 'face', 'left_hand', 'right_hand']:
            if modality in predicted and modality in target:
                pred_seq = predicted[modality]
                target_seq = target[modality]
                
                # Handle sequence length mismatch
                min_len = min(pred_seq.shape[1], target_seq.shape[1])
                pred_seq = pred_seq[:, :min_len, :]
                target_seq = target_seq[:, :min_len, :]
                
                # Aggressive clipping and NaN handling
                pred_seq = torch.clamp(pred_seq, -10, 10)
                target_seq = torch.clamp(target_seq, -10, 10)
                
                pred_seq = torch.nan_to_num(pred_seq, nan=0.0, posinf=1.0, neginf=-1.0)
                target_seq = torch.nan_to_num(target_seq, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Use MSE instead of Huber for more stability
                loss = F.mse_loss(pred_seq, target_seq, reduction='mean')
                
                # Additional validation
                if torch.isfinite(loss) and loss < 1000:  # Reject extremely large losses
                    losses[f'{modality}_loss'] = loss
                    total_loss += loss
                    valid_modalities += 1
                else:
                    logger.warning(f"Invalid/extreme loss for {modality}: {loss}, setting to 0")
                    losses[f'{modality}_loss'] = torch.tensor(0.0, device=pred_seq.device)
        
        # Average loss across valid modalities
        if valid_modalities > 0:
            total_loss = total_loss / valid_modalities
        else:
            total_loss = torch.tensor(0.0, device=self.device)
        
        # Final safety check
        if not torch.isfinite(total_loss):
            total_loss = torch.tensor(0.0, device=self.device)
        
        return total_loss, losses
    
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            # Move data to device
            text_tokens = {k: v.to(self.device).squeeze(1) for k, v in batch['text_tokens'].items()}
            
            target_keypoints = {
                'pose': batch['pose'].to(self.device),
                'face': batch['face'].to(self.device),
                'left_hand': batch['left_hand'].to(self.device),
                'right_hand': batch['right_hand'].to(self.device)
            }
            
            # Additional input validation
            for key, value in target_keypoints.items():
                if torch.isnan(value).any() or torch.isinf(value).any():
                    logger.warning(f"Invalid values in target {key}, replacing with zeros")
                    target_keypoints[key] = torch.zeros_like(value)
            
            target_length = target_keypoints['pose'].shape[1]
            labels = torch.arange(len(batch['text'])).to(self.device)
            
            # Forward pass
            outputs = self.model(text_tokens, target_length, labels)
            
            # Validate all outputs before loss computation
            for key, value in outputs['keypoints'].items():
                if key != 'weights' and (torch.isnan(value).any() or torch.isinf(value).any()):
                    print(f"Warning: Invalid model output in {key}, replacing with zeros")
                    outputs['keypoints'][key] = torch.zeros_like(value)
            
            # Compute reconstruction loss
            recon_loss, detailed_losses = self.compute_reconstruction_loss(
                outputs['keypoints'], target_keypoints
            )
            
            # More conservative contrastive loss handling
            total_loss = recon_loss
            if (outputs['contrastive_loss'] is not None and 
                torch.isfinite(outputs['contrastive_loss']) and 
                outputs['contrastive_loss'] < 100):
                total_loss += 0.01 * outputs['contrastive_loss']  # Reduced weight
            
            # Final safety check for total loss
            if not torch.isfinite(total_loss) or total_loss > 1000:
                logger.warning(f"Invalid total loss: {total_loss}, skipping batch")
                return {
                    'total_loss': 0.0,
                    'reconstruction_loss': 0.0,
                    'contrastive_loss': 0.0,
                    **{k: 0.0 for k in detailed_losses.keys()}
                }
            
            # Backward pass with additional safety
            try:
                total_loss.backward()
            except RuntimeError as e:
                logger.error(f"Error in backward pass: {e}, skipping batch")
                return {
                    'total_loss': 0.0,
                    'reconstruction_loss': 0.0,
                    'contrastive_loss': 0.0,
                    **{k: 0.0 for k in detailed_losses.keys()}
                }
            
            # More aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            # Check for valid gradients
            valid_gradients = True
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        valid_gradients = False
                        break
            
            if valid_gradients:
                self.optimizer.step()
            else:
                logger.warning("Skipping optimizer step due to invalid gradients")
            
            self.scheduler.step()
            
            # Remove the print statement that was causing issues
            return {
                'total_loss': float(total_loss.item()),
                'reconstruction_loss': float(recon_loss.item()),
                'contrastive_loss': float(outputs['contrastive_loss'].item()) if outputs['contrastive_loss'] is not None else 0.0,
                **{k: float(v.item()) for k, v in detailed_losses.items()}
            }
            
        except Exception as e:
            logger.error(f"Error in train_step: {e}")
            return {
                'total_loss': 0.0,
                'reconstruction_loss': 0.0,
                'contrastive_loss': 0.0,
                'pose_loss': 0.0,
                'face_loss': 0.0,
                'left_hand_loss': 0.0,
                'right_hand_loss': 0.0
            }

    def train_epoch(self, dataloader):
        """Train for one epoch and return average losses."""
        total_losses = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Train step
            losses = self.train_step(batch)
            
            # Accumulate losses (only valid ones)
            for key, value in losses.items():
                if isinstance(value, (int, float)) and not math.isnan(value) and math.isfinite(value):
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value
            
            num_batches += 1
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                total_loss_val = losses.get('total_loss', 0)
                logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                      f"Total Loss: {total_loss_val:.6f}")
        
        # Calculate average losses
        avg_losses = {}
        if num_batches > 0:
            for key, total_value in total_losses.items():
                avg_losses[key] = total_value / num_batches
        
        return avg_losses

class MHAGInference:
    """Inference pipeline for MHAG model."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_sign_language(self, text: str, target_length: int = 100):
        """Generate sign language from text input."""
        with torch.no_grad():
            # Tokenize input text
            text_tokens = self.tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
            
            # Generate keypoints
            outputs = self.model(text_tokens, target_length)
            
            # Convert to numpy for easier handling
            keypoints = {}
            for modality, data in outputs['keypoints'].items():
                if modality != 'weights':
                    keypoints[modality] = data.cpu().numpy()
            
            return keypoints

# =============================================================================
# Usage Example and Main Training Loop
# =============================================================================

def main():
    """Main training and inference example."""
    
    # Configuration
    config = {
        'json_dir': './output/extractor/',
        'd_model': 512,
        'batch_size': 8,
        'num_epochs': 50,
        'max_frames': 150,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Using device: {config['device']}")
    
    # Initialize dataset and dataloader
    dataset = SignLanguageDataset(
        json_dir=config['json_dir'],
        max_frames=config['max_frames']
    )
    
    def collate_fn(batch):
        """Custom collate function to handle variable length sequences."""
        # Handle text tokens
        text_tokens = {}
        for key in batch[0]['text_tokens'].keys():
            text_tokens[key] = torch.cat([item['text_tokens'][key] for item in batch], dim=0)
        
        # Handle sequences - pad to max length in batch
        pose_sequences = [item['pose'] for item in batch]
        face_sequences = [item['face'] for item in batch]
        left_hand_sequences = [item['left_hand'] for item in batch]
        right_hand_sequences = [item['right_hand'] for item in batch]
        
        # Pad sequences
        pose_padded = pad_sequence(pose_sequences, batch_first=True, padding_value=0)
        face_padded = pad_sequence(face_sequences, batch_first=True, padding_value=0)
        left_hand_padded = pad_sequence(left_hand_sequences, batch_first=True, padding_value=0)
        right_hand_padded = pad_sequence(right_hand_sequences, batch_first=True, padding_value=0)
        
        return {
            'text': [item['text'] for item in batch],
            'text_tokens': text_tokens,
            'pose': pose_padded,
            'face': face_padded,
            'left_hand': left_hand_padded,
            'right_hand': right_hand_padded,
            'metadata': [item['metadata'] for item in batch],
            'statistics': [item['statistics'] for item in batch]
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Initialize model
    model = MHAGModel(d_model=config['d_model'])
    
    # Initialize trainer
    trainer = MHAGTrainer(model, device=config['device'])
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train epoch
        avg_losses = trainer.train_epoch(dataloader)
        
        # Print epoch results
        logger.info(f"Average losses:")
        for key, value in avg_losses.items():
            logger.info(f"  {key}: {value:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'mhag_checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'losses': avg_losses
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Inference example
    logger.info("\n" + "="*50)
    logger.info("Inference Example")
    logger.info("="*50)
    
    # Initialize inference engine
    inference_engine = MHAGInference(
        model=model,
        tokenizer=dataset.tokenizer,
        device=config['device']
    )
    
    # Generate sign language for sample text
    sample_texts = [
        "안녕하세요",
        "감사합니다",
        "수화를 배우고 싶습니다"
    ]
    
    for text in sample_texts:
        logger.info(f"\nGenerating sign language for: '{text}'")
        keypoints = inference_engine.generate_sign_language(text, target_length=60)
        
        logger.info(f"Generated keypoint sequences:")
        for modality, data in keypoints.items():
            logger.info(f"  {modality}: {data.shape}")
        
        # Save generated keypoints (optional)
        output_file = f"generated_{text.replace(' ', '_')}.npz"
        np.savez(output_file, **keypoints)
        logger.info(f"Keypoints saved to: {output_file}")

def load_and_inference(checkpoint_path: str, text: str):
    """Load trained model and perform inference."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model
    model = MHAGModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    
    # Initialize inference engine
    inference_engine = MHAGInference(
        model=model,
        tokenizer=tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Generate sign language
    keypoints = inference_engine.generate_sign_language(text)
    
    return keypoints

if __name__ == "__main__":
    main()

    # Example usage for loading and inference
    keypoints = load_and_inference('mhag_checkpoint_epoch_50.pth', '탈골')
    logger.info("Generated keypoints shape:", {k: v.shape for k, v in keypoints.items()})
