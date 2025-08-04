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

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

class SignLanguageDataset(Dataset):
    """Dataset class for sign language data with extracted landmarks."""
    
    def __init__(self, json_dir: str, max_frames: int = 150, max_text_length: int = 128):
        self.json_dir = json_dir
        self.max_frames = max_frames
        self.max_text_length = max_text_length
        self.data_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        # Landmark dimensions
        self.pose_dim = 33 * 4  # 33 landmarks with 4 values each (x, y, z, visibility)
        self.face_dim = 468 * 3  # 468 landmarks with 3 values each (x, y, z)
        self.hand_dim = 21 * 3   # 21 landmarks with 3 values each (x, y, z)
        
        # Initialize tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load JSON data
        json_path = os.path.join(self.json_dir, self.data_files[idx])
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
                
        # Extract text
        try:
            text = data['metadata']['korean_text']
        except:
            print(f"No text found for {self.data_files[idx]}")
            print(data['metadata'])

        
        # Process frame data
        frames = data['frame_data']
        pose_sequence = []
        face_sequence = []
        left_hand_sequence = []
        right_hand_sequence = []
        
        for frame in frames[:self.max_frames]:
            # Extract pose landmarks
            pose_landmarks = frame['landmarks'].get('pose', [])
            pose_flat = [coord for landmark in pose_landmarks for coord in landmark] if pose_landmarks else [0] * self.pose_dim
            pose_sequence.append(pose_flat[:self.pose_dim])
            
            # Extract face landmarks
            face_landmarks = frame['landmarks'].get('face', [])
            face_flat = [coord for landmark in face_landmarks for coord in landmark[:3]] if face_landmarks else [0] * self.face_dim
            face_sequence.append(face_flat[:self.face_dim])
            
            # Extract hand landmarks
            left_hand = frame['landmarks'].get('left_hand', [])
            right_hand = frame['landmarks'].get('right_hand', [])
            
            left_hand_flat = [coord for landmark in left_hand for coord in landmark[:3]] if left_hand else [0] * self.hand_dim
            right_hand_flat = [coord for landmark in right_hand for coord in landmark[:3]] if right_hand else [0] * self.hand_dim
            
            left_hand_sequence.append(left_hand_flat[:self.hand_dim])
            right_hand_sequence.append(right_hand_flat[:self.hand_dim])
        
        # Convert to tensors
        pose_tensor = torch.FloatTensor(pose_sequence)
        face_tensor = torch.FloatTensor(face_sequence)
        left_hand_tensor = torch.FloatTensor(left_hand_sequence)
        right_hand_tensor = torch.FloatTensor(right_hand_sequence)
        
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
    """Multi-Modal Hierarchical Attention Generator (MHAG)."""
    
    def __init__(self, 
                 d_model: int = 512,
                 pose_dim: int = 132,  # 33 * 4
                 face_dim: int = 1404,  # 468 * 3
                 hand_dim: int = 63):   # 21 * 3
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
        
    def forward(self, text_tokens, target_length: int, labels=None):
        # Semantic understanding
        semantic_features = self.semantic_module(text_tokens)
        
        # Cross-modal attention alignment
        aligned_features, _ = self.cross_modal_attention(
            semantic_features, semantic_features, semantic_features
        )
        
        # Hierarchical motion planning
        motion_features = self.motion_planner(aligned_features, target_length)
        
        # Multi-keypoint generation
        keypoint_outputs = self.keypoint_generator(motion_features)
        
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
    """Training pipeline for MHAG model."""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def compute_reconstruction_loss(self, predicted, target):
        """Compute MSE loss for keypoint reconstruction."""
        losses = {}
        total_loss = 0
        
        for modality in ['pose', 'face', 'left_hand', 'right_hand']:
            if modality in predicted and modality in target:
                # Handle sequence length mismatch
                pred_seq = predicted[modality]
                target_seq = target[modality]
                
                min_len = min(pred_seq.shape[1], target_seq.shape[1])
                pred_seq = pred_seq[:, :min_len, :]
                target_seq = target_seq[:, :min_len, :]
                
                loss = F.mse_loss(pred_seq, target_seq)
                losses[f'{modality}_loss'] = loss
                total_loss += loss
        
        return total_loss, losses
    
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        text_tokens = {k: v.to(self.device).squeeze(1) for k, v in batch['text_tokens'].items()}
        
        target_keypoints = {
            'pose': batch['pose'].to(self.device),
            'face': batch['face'].to(self.device),
            'left_hand': batch['left_hand'].to(self.device),
            'right_hand': batch['right_hand'].to(self.device)
        }
        
        target_length = target_keypoints['pose'].shape[1]
        
        # Create labels for contrastive learning (using text for simplicity)
        labels = torch.arange(len(batch['text'])).to(self.device)
        
        # Forward pass
        outputs = self.model(text_tokens, target_length, labels)
        
        # Compute reconstruction loss
        recon_loss, detailed_losses = self.compute_reconstruction_loss(
            outputs['keypoints'], target_keypoints
        )
        
        # Total loss
        total_loss = recon_loss
        if outputs['contrastive_loss'] is not None:
            total_loss += 0.1 * outputs['contrastive_loss']
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'contrastive_loss': outputs['contrastive_loss'].item() if outputs['contrastive_loss'] is not None else 0,
            **{k: v.item() for k, v in detailed_losses.items()}
        }
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        total_losses = {}
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            losses = self.train_step(batch)
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value
            
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'recon': f"{losses['reconstruction_loss']:.4f}"
            })
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        self.scheduler.step()
        
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
    
    print(f"Using device: {config['device']}")
    
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
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    print("Starting training...")
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train epoch
        avg_losses = trainer.train_epoch(dataloader)
        
        # Print epoch results
        print(f"Average losses:")
        for key, value in avg_losses.items():
            print(f"  {key}: {value:.6f}")
        
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
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Inference example
    print("\n" + "="*50)
    print("Inference Example")
    print("="*50)
    
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
        print(f"\nGenerating sign language for: '{text}'")
        keypoints = inference_engine.generate_sign_language(text, target_length=60)
        
        print(f"Generated keypoint sequences:")
        for modality, data in keypoints.items():
            print(f"  {modality}: {data.shape}")
        
        # Save generated keypoints (optional)
        output_file = f"generated_{text.replace(' ', '_')}.npz"
        np.savez(output_file, **keypoints)
        print(f"Keypoints saved to: {output_file}")

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
    print("Generated keypoints shape:", {k: v.shape for k, v in keypoints.items()})
