#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural ODE 벡터 필드 모델 정의
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    다층 퍼셉트론 (MLP) 벡터 필드 모델
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh', 
                 use_batch_norm=False, dropout_rate=0.0):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
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
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                raise ValueError(f"지원하지 않는 활성화 함수: {activation}")
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 출력층
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, y):
        """
        Neural ODE 벡터 필드 함수.
        
        Args:
            t (torch.Tensor): 시간점 (일반적으로 scalar이나 Neural ODE에서 요구됨)
            y (torch.Tensor): 상태 벡터 (batch_size, state_dim)
            
        Returns:
            torch.Tensor: dy/dt 벡터장 (batch_size, state_dim)
        """
        if len(y.shape) == 1:
            y = y.unsqueeze(0)  # (state_dim,) -> (1, state_dim)
        
        # 시간 t를 특징 벡터에 통합 (선택적)
        # t_expanded = t.expand(y.shape[0], 1)
        # ty = torch.cat([t_expanded, y], dim=1)
        
        # 현재는 자율적(autonomous) 시스템으로 간주 (t 무시)
        return self.net(y)


class ResNet(nn.Module):
    """
    ResNet 스타일 블록을 사용한 벡터 필드
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh', 
                 use_batch_norm=False, dropout_rate=0.0):
        super(ResNet, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"지원하지 않는 활성화 함수: {activation}")
        
        # ResNet 블록 생성
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            block = ResBlock(
                hidden_dims[i], hidden_dims[i+1], 
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate
            )
            self.res_blocks.append(block)
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, t, y):
        """
        Neural ODE 벡터 필드 함수 (ResNet 버전)
        """
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        
        h = self.activation(self.input_layer(y))
        
        for res_block in self.res_blocks:
            h = res_block(h)
        
        return self.output_layer(h)


class ResBlock(nn.Module):
    """
    ResNet 블록
    """
    def __init__(self, input_dim, hidden_dim, activation='tanh', use_batch_norm=False, dropout_rate=0.0):
        super(ResBlock, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)  # 원래 차원으로 환원
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
            
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(input_dim)
            
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
            
    def forward(self, x):
        identity = x
        
        out = self.linear1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        
        if self.dropout_rate > 0:
            out = self.dropout(out)
            
        out = self.linear2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
            
        out += identity  # 스킵 연결
        out = self.activation(out)
        
        return out


def create_vector_field(config, state_dim):
    """
    설정에 따라 적절한 벡터 필드 모델 생성
    
    Args:
        config (dict): 모델 설정
        state_dim (int): 상태 벡터 차원
        
    Returns:
        nn.Module: 벡터 필드 모델
    """
    model_type = config['type']
    hidden_dims = config['hidden_dims']
    activation = config['activation']
    use_batch_norm = config['use_batch_norm']
    dropout_rate = config['dropout_rate']
    
    if model_type == 'mlp':
        return MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=state_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
    elif model_type == 'resnet':
        return ResNet(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=state_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}") 