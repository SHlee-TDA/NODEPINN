#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PINN 프레임워크에 사용되는 다양한 손실 함수 정의
"""

import torch
import torch.autograd as autograd


def physics_loss(y_pred, t_grid, theta_pred, ode_rhs_fn, batch_size=None):
    """
    물리 기반 손실: ODE 잔차 (오차) 손실 계산
    
    dy/dt = f(y, t, theta)의 제약 조건을 적용
    autograd로 dy/dt 계산 후 RHS(f)와 비교
    
    Args:
        y_pred (torch.Tensor): 예측된 trajectory (batch_size, seq_len, state_dim)
        t_grid (torch.Tensor): 시간 그리드 (seq_len,)
        theta_pred (torch.Tensor): 예측된 파라미터 (batch_size, param_dim)
        ode_rhs_fn (callable): ODE 우변 함수
        batch_size (int, optional): 배치 크기 (샘플링 사용 시)
        
    Returns:
        torch.Tensor: 평균 물리 손실 (스칼라)
    """
    # 필요한 경우 배치에서 일부 샘플만 선택
    if batch_size is not None and batch_size < y_pred.shape[0]:
        indices = torch.randperm(y_pred.shape[0])[:batch_size]
        y_pred = y_pred[indices]
        theta_pred = theta_pred[indices]
    
    # 복원된 경사 계산을 위해 requires_grad 설정
    if not t_grid.requires_grad:
        t_grid = t_grid.detach().requires_grad_(True)
    
    total_loss = 0.0
    batch_size = y_pred.shape[0]
    seq_len = y_pred.shape[1]
    
    # 시간 그리드에 대한 단일 배치 경사 계산
    for i in range(seq_len):
        # i번째 시간 지점에서의 상태
        y_t = y_pred[:, i, :]
        t = t_grid[i]
        
        # y에 대한 t의 명시적인 미분 계산
        y_t.requires_grad_(True)
        
        # 미분 dy/dt 계산
        dy_dt = []
        for j in range(batch_size):
            # 각 배치 요소에 대해 개별적으로 미분
            # 참고: 배치 처리를 위한 더 효율적인 구현이 가능하지만, 명확성을 위해 루프 사용
            y_j = y_t[j:j+1]
            
            # 1차 미분 계산 (j번째 샘플에 대한)
            grad_outputs = torch.ones_like(y_j)
            grad_y = autograd.grad(
                outputs=y_j,
                inputs=t,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            dy_dt.append(grad_y)
        
        # 배치의 모든 미분 결합
        dy_dt = torch.stack(dy_dt)
        
        # ODE 우변 계산
        rhs = ode_rhs_fn(y_t, theta_pred, t)
        
        # 잔차(residual) 계산: dy/dt - f(y, t, theta)
        residual = dy_dt - rhs
        
        # MSE 손실 계산
        loss = torch.mean(residual ** 2)
        total_loss += loss
    
    # 시퀀스 길이로 정규화
    return total_loss / seq_len


def data_loss(y_pred, y_true, reduction='mean'):
    """
    데이터 기반 손실: 예측과 실제 trajectory 사이의 MSE
    
    Args:
        y_pred (torch.Tensor): 예측된 trajectory (batch_size, seq_len, state_dim)
        y_true (torch.Tensor): 실제 trajectory (batch_size, seq_len, state_dim)
        reduction (str): 손실 감소 방법 ('mean', 'sum', 'none')
        
    Returns:
        torch.Tensor: 데이터 손실
    """
    if reduction == 'mean':
        return torch.mean((y_pred - y_true) ** 2)
    elif reduction == 'sum':
        return torch.sum((y_pred - y_true) ** 2)
    elif reduction == 'none':
        return (y_pred - y_true) ** 2
    else:
        raise ValueError(f"지원하지 않는 reduction 유형: {reduction}")


def parameter_loss(theta_pred, theta_true, reduction='mean'):
    """
    파라미터 추정 손실: 예측과 실제 파라미터 사이의 MSE
    
    Args:
        theta_pred (torch.Tensor): 예측된 파라미터 (batch_size, param_dim)
        theta_true (torch.Tensor): 실제 파라미터 (batch_size, param_dim)
        reduction (str): 손실 감소 방법 ('mean', 'sum', 'none')
        
    Returns:
        torch.Tensor: 파라미터 손실
    """
    if reduction == 'mean':
        return torch.mean((theta_pred - theta_true) ** 2)
    elif reduction == 'sum':
        return torch.sum((theta_pred - theta_true) ** 2)
    elif reduction == 'none':
        return (theta_pred - theta_true) ** 2
    else:
        raise ValueError(f"지원하지 않는 reduction 유형: {reduction}")


def regularization_loss(model, reg_type='l2'):
    """
    모델 가중치에 대한 정규화 손실
    
    Args:
        model (nn.Module): 정규화할 신경망 모델
        reg_type (str): 정규화 유형 ('l1', 'l2')
        
    Returns:
        torch.Tensor: 정규화 손실
    """
    reg_loss = 0.0
    
    for param in model.parameters():
        if reg_type == 'l2':
            reg_loss += torch.sum(param ** 2)
        elif reg_type == 'l1':
            reg_loss += torch.sum(torch.abs(param))
        else:
            raise ValueError(f"지원하지 않는 정규화 유형: {reg_type}")
    
    return reg_loss


def total_loss(y_pred, y_true, t_grid, theta_pred, theta_true, ode_rhs_fn, 
               config, model=None):
    """
    전체 손실 함수: 물리 + 데이터 + 파라미터 + 정규화
    
    Args:
        y_pred (torch.Tensor): 예측된 trajectory
        y_true (torch.Tensor): 실제 trajectory
        t_grid (torch.Tensor): 시간 그리드
        theta_pred (torch.Tensor): 예측된 파라미터
        theta_true (torch.Tensor): 실제 파라미터
        ode_rhs_fn (callable): ODE 우변 함수
        config (dict): 손실 가중치 등의 설정
        model (nn.Module, optional): 정규화를 위한 모델
        
    Returns:
        torch.Tensor: 전체 손실
        dict: 각 손실 구성요소
    """
    # 각 손실 구성요소 계산
    phys_loss = physics_loss(y_pred, t_grid, theta_pred, ode_rhs_fn)
    d_loss = data_loss(y_pred, y_true)
    param_loss = parameter_loss(theta_pred, theta_true)
    
    # 가중치 적용
    weighted_phys_loss = config['physics_weight'] * phys_loss
    weighted_data_loss = config['data_weight'] * d_loss
    
    # 정규화 손실 (선택적)
    reg_loss = 0.0
    if model is not None and config['reg_weight'] > 0:
        reg_loss = config['reg_weight'] * regularization_loss(model)
    
    # 전체 손실
    total = weighted_phys_loss + weighted_data_loss + param_loss + reg_loss
    
    # 각 구성요소 반환 (디버깅용)
    loss_components = {
        'physics_loss': phys_loss.item(),
        'data_loss': d_loss.item(),
        'parameter_loss': param_loss.item(),
        'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
        'total_loss': total.item()
    }
    
    return total, loss_components 