#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
유틸리티 함수 모듈
"""

import os
import random
import numpy as np
import torch
import yaml
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt


def set_seed(seed):
    """
    재현성을 위한 시드 설정
    
    Args:
        seed (int): 시드 값
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"시드가 {seed}로 설정되었습니다.")


def load_config(config_path):
    """
    YAML 설정 파일 로드
    
    Args:
        config_path (str): 설정 파일 경로
        
    Returns:
        dict: 설정 데이터
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_device(device_name=None):
    """
    학습 장치 준비
    
    Args:
        device_name (str, optional): 'cuda' 또는 'cpu' 또는 None (자동)
        
    Returns:
        torch.device: 준비된 장치
    """
    if device_name is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    print(f"사용 장치: {device}")
    return device


def setup_logger(name, log_dir, log_file, level=logging.INFO):
    """
    로깅 설정
    
    Args:
        name (str): 로거 이름
        log_dir (str): 로그 디렉터리
        log_file (str): 로그 파일 이름
        level (logging.Level): 로깅 레벨
        
    Returns:
        logging.Logger: 로거 객체
    """
    # 로그 디렉터리 생성
    os.makedirs(log_dir, exist_ok=True)
    
    # 로거 설정
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # 핸들러 중복 방지
    if not logger.handlers:
        # 콘솔 출력 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 파일 출력 핸들러
        file_path = os.path.join(log_dir, log_file)
        file_handler = RotatingFileHandler(file_path, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_directory(directory):
    """
    디렉터리 생성
    
    Args:
        directory (str): 생성할 디렉터리 경로
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"디렉터리 생성: {directory}")


def get_model_summary(model):
    """
    모델 요약 정보 반환
    
    Args:
        model (nn.Module): 요약할 모델
        
    Returns:
        str: 모델 요약 문자열
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return (f"모델: {model.__class__.__name__}\n"
            f"총 파라미터 수: {params:,}")


def check_gradient_flow(named_parameters, save_path=None):
    """
    그래디언트 흐름 확인을 위한 그래프 생성
    
    Args:
        named_parameters: 모델의 named_parameters() 반환값
        save_path (str, optional): 그래프 저장 경로
    """
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())
    
    plt.figure(figsize=(15, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color='c')
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.3, lw=1, color='b')
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color='k')
    plt.xticks(np.arange(len(ave_grads)), layers, rotation='vertical')
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=max(max_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def compute_gradient_norm(model):
    """
    모델의 전체 그래디언트 노름 계산
    
    Args:
        model (nn.Module): 대상 모델
        
    Returns:
        float: 그래디언트 L2 노름
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    return total_norm


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=None):
    """
    데이터셋을 학습/검증/테스트 세트로 분할
    
    Args:
        dataset (list): 분할할 데이터셋
        train_ratio (float): 학습 데이터 비율
        val_ratio (float): 검증 데이터 비율
        test_ratio (float): 테스트 데이터 비율
        seed (int, optional): 랜덤 시드
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "비율의 합이 1이 되어야 합니다"
    
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    test_data = [dataset[i] for i in test_indices]
    
    return train_data, val_data, test_data 