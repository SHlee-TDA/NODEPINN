#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
파라미터 추정 모델 정의 (trajectory -> parameter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class MLP_Estimator(nn.Module):
    """
    다층 퍼셉트론 기반 파라미터 추정기
    """
    def __init__(self, input_dim, seq_len, hidden_dims, output_dim, activation='relu', 
                 use_batch_norm=False, dropout_rate=0.0):
        super(MLP_Estimator, self).__init__()
        
        # 입력층 (시퀀스를 평탄화하여 처리)
        self.input_dim = input_dim
        self.seq_len = seq_len
        flat_dim = input_dim * seq_len
        
        layers = []
        prev_dim = flat_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            else:
                raise ValueError(f"지원하지 않는 활성화 함수: {activation}")
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 출력층
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 상태 시퀀스 (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: 예측된 파라미터 (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        # 형태 변환: (batch_size, seq_len, input_dim) -> (batch_size, seq_len * input_dim)
        x_flat = x.reshape(batch_size, -1)
        
        return self.net(x_flat)


class LSTM_Estimator(nn.Module):
    """
    LSTM 기반 파라미터 추정기
    """
    def __init__(self, input_dim, hidden_dims, output_dim, n_layers=2, 
                 bidirectional=True, dropout_rate=0.0):
        super(LSTM_Estimator, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if n_layers > 1 else 0
        )
        
        # LSTM의 마지막 출력 차원
        lstm_output_dim = hidden_dims[0] * (2 if bidirectional else 1)
        
        # 후처리 MLP
        mlp_layers = []
        prev_dim = lstm_output_dim
        
        for i in range(1, len(hidden_dims)):
            mlp_layers.append(nn.Linear(prev_dim, hidden_dims[i]))
            mlp_layers.append(nn.ReLU())
            if dropout_rate > 0:
                mlp_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dims[i]
        
        # 출력층
        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 상태 시퀀스 (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: 예측된 파라미터 (batch_size, output_dim)
        """
        # LSTM 통과
        output, (h_n, c_n) = self.lstm(x)
        
        # 마지막 시간 단계 출력 또는 모든 은닉 상태의 평균 사용
        # 여기서는 마지막 타임스텝의 출력 사용
        last_output = output[:, -1, :]
        
        # MLP를 통한 후처리
        return self.mlp(last_output)


class CNN_Estimator(nn.Module):
    """
    1D CNN 기반 파라미터 추정기
    """
    def __init__(self, input_dim, hidden_dims, output_dim, kernel_sizes=[3, 5, 7], 
                 use_batch_norm=True, dropout_rate=0.0):
        super(CNN_Estimator, self).__init__()
        
        # 다양한 커널 크기를 가진 1D CNN 병렬 처리
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dims[0]//len(kernel_sizes), 
                     kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # 배치 정규화
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dims[0])
        
        # 후처리 MLP
        mlp_layers = []
        prev_dim = hidden_dims[0]
        
        for i in range(1, len(hidden_dims)):
            mlp_layers.append(nn.Linear(prev_dim, hidden_dims[i]))
            mlp_layers.append(nn.ReLU())
            if dropout_rate > 0:
                mlp_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dims[i]
        
        # 출력층
        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 상태 시퀀스 (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: 예측된 파라미터 (batch_size, output_dim)
        """
        # CNN은 입력 채널이 마지막 차원에 있어야 함
        # 형태 변환: (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 각 CNN 통과 후 특징 결합
        conv_results = []
        for conv in self.convs:
            # 활성화 및 최대 풀링
            result = F.relu(conv(x))
            result = F.adaptive_max_pool1d(result, 1).squeeze(-1)
            conv_results.append(result)
        
        # 다양한 커널의 결과 결합
        combined = torch.cat(conv_results, dim=1)
        
        if self.use_batch_norm:
            combined = self.bn(combined)
        
        # MLP 통과
        return self.mlp(combined)


class Transformer_Estimator(nn.Module):
    """
    Transformer 기반 파라미터 추정기
    """
    def __init__(self, input_dim, hidden_dims, output_dim, n_heads=4, n_layers=2,
                 dropout_rate=0.1):
        super(Transformer_Estimator, self).__init__()
        
        # 입력 임베딩
        self.embedding = nn.Linear(input_dim, hidden_dims[0])
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(hidden_dims[0], dropout_rate)
        
        # Transformer 인코더 레이어
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dims[0],
            nhead=n_heads,
            dim_feedforward=hidden_dims[0] * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=n_layers
        )
        
        # 후처리 MLP
        mlp_layers = []
        prev_dim = hidden_dims[0]
        
        for i in range(1, len(hidden_dims)):
            mlp_layers.append(nn.Linear(prev_dim, hidden_dims[i]))
            mlp_layers.append(nn.ReLU())
            if dropout_rate > 0:
                mlp_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dims[i]
        
        # 출력층
        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): 상태 시퀀스 (batch_size, seq_len, input_dim)
            mask (torch.Tensor, optional): 어텐션 마스크 (선택적)
            
        Returns:
            torch.Tensor: 예측된 파라미터 (batch_size, output_dim)
        """
        # 입력 임베딩 및 위치 인코딩
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # Transformer 인코더 통과
        x = self.transformer_encoder(x, mask)
        
        # 시퀀스 평균 (또는 마지막 토큰 사용)
        x = x.mean(dim=1)
        
        # MLP를 통한 최종 파라미터 예측
        return self.mlp(x)


class PositionalEncoding(nn.Module):
    """
    Transformer를 위한 위치 인코딩
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def create_param_estimator(config, input_dim, seq_len, output_dim):
    """
    설정에 따라 적절한 파라미터 추정 모델 생성
    
    Args:
        config (dict): 모델 설정
        input_dim (int): 입력 상태 벡터 차원
        seq_len (int): 시퀀스 길이
        output_dim (int): 출력 파라미터 차원
        
    Returns:
        nn.Module: 파라미터 추정 모델
    """
    model_type = config['type']
    hidden_dims = config['hidden_dims']
    
    if model_type == 'mlp':
        return MLP_Estimator(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=config['activation'],
            use_batch_norm=config['use_batch_norm'],
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'lstm':
        return LSTM_Estimator(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            n_layers=config['n_layers'],
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'cnn':
        return CNN_Estimator(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_batch_norm=config['use_batch_norm'],
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'transformer':
        return Transformer_Estimator(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dropout_rate=config['dropout_rate']
        )
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}") 