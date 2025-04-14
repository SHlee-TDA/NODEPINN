#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
모델 학습 관련 코드
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Any, Optional
from pathlib import Path
import wandb

from src.training.loss import total_loss
from src.models.ode_rhs import get_ode_rhs


class Trainer:
    """
    모델 학습을 위한 Trainer 클래스
    """
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 config: Dict[str, Any],
                 device: Optional[str] = None):
        """
        Args:
            model (torch.nn.Module): 모델
            train_loader (DataLoader): 학습 데이터 로더
            val_loader (DataLoader): 검증 데이터 로더
            optimizer (torch.optim.Optimizer): 옵티마이저
            config (Dict[str, Any]): 학습 설정
            device (Optional[str], optional): 학습 디바이스 (기본값: config에서 가져옴)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        
        self.device = device or config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 학습 설정
        self.num_epochs = config.get('num_epochs', 100)
        self.save_dir = Path(config.get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project=config.get('project_name', 'lotka_volterra'),
                      config=config)
        
        # ODE 유형에 따른 우변 함수 가져오기
        self.ode_type = 'lotka_volterra'  # TODO: config에서 가져오기
        self.ode_rhs_fn = get_ode_rhs(self.ode_type)
        
        # 메트릭 저장용
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에폭의 학습을 수행합니다.
        
        Returns:
            Dict[str, float]: 학습 메트릭
        """
        self.model.train()
        total_loss = 0
        total_param_loss = 0
        total_traj_loss = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # 데이터를 디바이스로 이동
            x = batch['x'].to(self.device)
            true_params = batch['params'].to(self.device)
            true_traj = batch['traj'].to(self.device)
            
            # forward pass
            pred_params, pred_traj = self.model(x)
            
            # 손실 계산
            losses = self.model.compute_loss(pred_params, pred_traj,
                                          true_params, true_traj)
            
            # 역전파
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # 메트릭 업데이트
            total_loss += losses['total_loss'].item()
            total_param_loss += losses['param_loss'].item()
            total_traj_loss += losses['traj_loss'].item()
        
        num_batches = len(self.train_loader)
        return {
            'train_loss': total_loss / num_batches,
            'train_param_loss': total_param_loss / num_batches,
            'train_traj_loss': total_traj_loss / num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """검증을 수행합니다.
        
        Returns:
            Dict[str, float]: 검증 메트릭
        """
        self.model.eval()
        total_loss = 0
        total_param_loss = 0
        total_traj_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                x = batch['x'].to(self.device)
                true_params = batch['params'].to(self.device)
                true_traj = batch['traj'].to(self.device)
                
                pred_params, pred_traj = self.model(x)
                losses = self.model.compute_loss(pred_params, pred_traj,
                                              true_params, true_traj)
                
                total_loss += losses['total_loss'].item()
                total_param_loss += losses['param_loss'].item()
                total_traj_loss += losses['traj_loss'].item()
        
        num_batches = len(self.val_loader)
        return {
            'val_loss': total_loss / num_batches,
            'val_param_loss': total_param_loss / num_batches,
            'val_traj_loss': total_traj_loss / num_batches
        }
    
    def train(self):
        """전체 학습 과정을 수행합니다."""
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # 학습
            train_metrics = self.train_epoch()
            
            # 검증
            val_metrics = self.validate()
            
            # 메트릭 로깅
            metrics = {**train_metrics, **val_metrics}
            if self.use_wandb:
                wandb.log(metrics, step=epoch)
            
            # 모델 저장
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.model.save(self.save_dir / 'best_model.pt')
            
            # 에폭별 모델 저장
            self.model.save(self.save_dir / f'model_epoch_{epoch}.pt')
            
            # 진행 상황 출력
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print("-" * 50)
        
        if self.use_wandb:
            wandb.finish()
    
    def plot_training_curves(self, save_path=None):
        """
        학습 및 검증 손실 곡선 시각화
        
        Args:
            save_path (str, optional): 저장 경로
        """
        epochs = range(1, len(self.train_metrics['loss']) + 1)
        
        plt.figure(figsize=(12, 8))
        
        # 전체 손실
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.train_metrics['loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.val_metrics['loss'], 'r-', label='Validation Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 물리 손실
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.train_metrics['physics_loss'], 'b-', label='Train Physics Loss')
        plt.plot(epochs, self.val_metrics['physics_loss'], 'r-', label='Val Physics Loss')
        plt.title('Physics Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 데이터 손실
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.train_metrics['data_loss'], 'b-', label='Train Data Loss')
        plt.plot(epochs, self.val_metrics['data_loss'], 'r-', label='Val Data Loss')
        plt.title('Data Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 파라미터 손실
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.train_metrics['parameter_loss'], 'b-', label='Train Parameter Loss')
        plt.plot(epochs, self.val_metrics['parameter_loss'], 'r-', label='Val Parameter Loss')
        plt.title('Parameter Estimation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"학습 곡선 저장됨: {save_path}")
        
        plt.show() 