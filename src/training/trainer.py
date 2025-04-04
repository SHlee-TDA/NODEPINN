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

from src.training.loss import total_loss
from src.models.ode_rhs import get_ode_rhs


class Trainer:
    """
    모델 학습을 위한 Trainer 클래스
    """
    def __init__(self, vector_field, param_estimator, config, device):
        """
        Args:
            vector_field (nn.Module): ODE 벡터 필드 모델
            param_estimator (nn.Module): 파라미터 추정 모델
            config (dict): 학습 설정
            device (torch.device): 학습 디바이스
        """
        self.vector_field = vector_field
        self.param_estimator = param_estimator
        self.config = config
        self.device = device
        
        # ODE 유형에 따른 우변 함수 가져오기
        self.ode_type = 'lotka_volterra'  # TODO: config에서 가져오기
        self.ode_rhs_fn = get_ode_rhs(self.ode_type)
        
        # 옵티마이저 설정
        self.setup_optimizer()
        
        # 텐서보드 설정
        log_dir = os.path.join(config['experiment']['log_dir'], config['experiment']['name'])
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 베스트 모델 저장용
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.checkpoint_dir = os.path.join(config['experiment']['checkpoint_dir'], 
                                          config['experiment']['name'])
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 메트릭 저장용
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
    
    def setup_optimizer(self):
        """옵티마이저와 스케줄러 설정"""
        # 학습할 모든 파라미터
        parameters = list(self.vector_field.parameters()) + list(self.param_estimator.parameters())
        
        # 옵티마이저 선택
        optim_config = self.config['training']
        if optim_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                parameters, 
                lr=optim_config['learning_rate'],
                weight_decay=optim_config['weight_decay']
            )
        elif optim_config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                parameters, 
                lr=optim_config['learning_rate'],
                momentum=0.9,
                weight_decay=optim_config['weight_decay']
            )
        elif optim_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                parameters, 
                lr=optim_config['learning_rate'],
                weight_decay=optim_config['weight_decay']
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optim_config['optimizer']}")
        
        # 학습률 스케줄러 설정
        if optim_config['scheduler'] == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=optim_config['lr_step_size'],
                gamma=optim_config['lr_gamma']
            )
        elif optim_config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=optim_config['n_epochs']
            )
        else:
            self.scheduler = None
    
    def prepare_batch(self, batch):
        """
        배치 데이터 준비
        
        Args:
            batch (tuple): (trajectory, time_points, parameters, initial_conditions)
            
        Returns:
            tuple: 처리된 배치 데이터
        """
        trajectories, time_points, parameters, initial_conditions = batch
        
        # 장치 이동
        trajectories = trajectories.to(self.device)
        time_points = time_points.to(self.device)
        parameters = parameters.to(self.device)
        initial_conditions = initial_conditions.to(self.device)
        
        return trajectories, time_points, parameters, initial_conditions
    
    def train_step(self, batch):
        """
        단일 학습 스텝 수행
        
        Args:
            batch (tuple): 배치 데이터
            
        Returns:
            float: 학습 손실
            dict: 손실 구성요소
        """
        self.vector_field.train()
        self.param_estimator.train()
        
        # 데이터 준비
        trajectories, time_points, parameters, initial_conditions = self.prepare_batch(batch)
        
        # 순전파
        # 1. 파라미터 추정
        theta_pred = self.param_estimator(trajectories)
        
        # 2. 예측된 파라미터를 사용하여 trajectory 계산
        # (실제로는 Neural ODE를 사용하여 trajectory 계산)
        # 간단한 구현을 위해 여기서는 실제 trajectory 사용
        y_pred = trajectories  # TODO: Neural ODE 솔버 구현
        
        # 손실 계산
        loss, loss_components = total_loss(
            y_pred=y_pred,
            y_true=trajectories,
            t_grid=time_points[0],  # 모든 배치가 같은 시간 그리드 사용
            theta_pred=theta_pred,
            theta_true=parameters,
            ode_rhs_fn=self.ode_rhs_fn,
            config=self.config['loss'],
            model=self.param_estimator
        )
        
        # 역전파 및 옵티마이저 스텝
        self.optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑 (선택적)
        if self.config['training']['gradient_clip_val'] > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.vector_field.parameters()) + list(self.param_estimator.parameters()),
                self.config['training']['gradient_clip_val']
            )
        
        self.optimizer.step()
        
        return loss.item(), loss_components
    
    def validate(self, val_loader):
        """
        검증 세트에서 모델 평가
        
        Args:
            val_loader (DataLoader): 검증 데이터 로더
            
        Returns:
            float: 평균 검증 손실
            dict: 평균 손실 구성요소
        """
        self.vector_field.eval()
        self.param_estimator.eval()
        
        total_loss = 0.0
        loss_components_sum = defaultdict(float)
        
        with torch.no_grad():
            for batch in val_loader:
                # 데이터 준비
                trajectories, time_points, parameters, initial_conditions = self.prepare_batch(batch)
                
                # 파라미터 추정
                theta_pred = self.param_estimator(trajectories)
                
                # 간단한 구현을 위해 실제 trajectory 사용
                y_pred = trajectories  # TODO: Neural ODE 솔버 구현
                
                # 손실 계산
                loss, loss_components = total_loss(
                    y_pred=y_pred,
                    y_true=trajectories,
                    t_grid=time_points[0],
                    theta_pred=theta_pred,
                    theta_true=parameters,
                    ode_rhs_fn=self.ode_rhs_fn,
                    config=self.config['loss']
                )
                
                total_loss += loss.item()
                
                # 손실 구성요소 누적
                for k, v in loss_components.items():
                    loss_components_sum[k] += v
        
        # 평균 손실 및 구성요소 계산
        avg_loss = total_loss / len(val_loader)
        avg_components = {k: v / len(val_loader) for k, v in loss_components_sum.items()}
        
        return avg_loss, avg_components
    
    def train(self, train_loader, val_loader, n_epochs=None):
        """
        모델 학습 수행
        
        Args:
            train_loader (DataLoader): 학습 데이터 로더
            val_loader (DataLoader): 검증 데이터 로더
            n_epochs (int, optional): 에폭 수 (기본값: config에서 가져옴)
            
        Returns:
            dict: 학습 및 검증 메트릭
        """
        if n_epochs is None:
            n_epochs = self.config['training']['n_epochs']
        
        patience = self.config['training']['early_stopping_patience']
        
        # 학습 루프
        for epoch in range(n_epochs):
            # 학습 단계
            epoch_loss = 0.0
            epoch_components = defaultdict(float)
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
            for batch in progress_bar:
                # 학습 스텝
                loss, loss_components = self.train_step(batch)
                
                # 손실 누적
                epoch_loss += loss
                for k, v in loss_components.items():
                    epoch_components[k] += v
                
                # 프로그레스바 업데이트
                progress_bar.set_postfix({
                    'loss': loss,
                    'param_loss': loss_components['parameter_loss']
                })
            
            # 에폭 평균 손실
            epoch_loss /= len(train_loader)
            epoch_components = {k: v / len(train_loader) for k, v in epoch_components.items()}
            
            # 검증 단계
            val_loss, val_components = self.validate(val_loader)
            
            # 학습률 스케줄러 스텝
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 텐서보드에 로깅
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            for k, v in epoch_components.items():
                self.writer.add_scalar(f'Components/train_{k}', v, epoch)
            
            for k, v in val_components.items():
                self.writer.add_scalar(f'Components/val_{k}', v, epoch)
            
            # 메트릭 저장
            self.train_metrics['loss'].append(epoch_loss)
            self.val_metrics['loss'].append(val_loss)
            
            for k, v in epoch_components.items():
                self.train_metrics[k].append(v)
            
            for k, v in val_components.items():
                self.val_metrics[k].append(v)
            
            # 모델 체크포인트 저장
            if (epoch + 1) % self.config['experiment']['save_freq'] == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # 베스트 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"Epoch {epoch+1}: 새로운 최고 성능 모델 저장 (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                print(f"Epoch {epoch+1}: train_loss: {epoch_loss:.6f}, val_loss: {val_loss:.6f}")
            
            # 조기 종료
            if patience > 0 and self.patience_counter >= patience:
                print(f"Epoch {epoch+1}: 성능 향상 없음. {patience}회 동안 훈련 중단.")
                break
        
        # 텐서보드 마무리
        self.writer.close()
        
        return {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        모델 체크포인트 저장
        
        Args:
            epoch (int): 현재 에폭
            is_best (bool): 최고 성능 모델 여부
        """
        checkpoint = {
            'epoch': epoch,
            'vector_field_state_dict': self.vector_field.state_dict(),
            'param_estimator_state_dict': self.param_estimator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'config': self.config,
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pt'))
        else:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    def load_checkpoint(self, checkpoint_path):
        """
        체크포인트에서 모델 로드
        
        Args:
            checkpoint_path (str): 체크포인트 파일 경로
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.vector_field.load_state_dict(checkpoint['vector_field_state_dict'])
        self.param_estimator.load_state_dict(checkpoint['param_estimator_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_metrics = checkpoint['train_metrics']
        self.val_metrics = checkpoint['val_metrics']
        
        print(f"체크포인트 로드 완료: 에폭 {checkpoint['epoch']+1}, 최고 검증 손실 {self.best_val_loss:.6f}")
        
        return checkpoint['epoch']
    
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