#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
다양한 ODE 시스템의 우변(Right-Hand Side) 함수 정의
미분방정식 dy/dt = f(y, t, theta)에서 f를 정의하는 모듈
"""

import torch
import numpy as np


def lotka_volterra_rhs(y, theta, t=None):
    """
    Lotka-Volterra 시스템의 우변 함수
    
    dy1/dt = alpha * y1 - beta * y1 * y2
    dy2/dt = -gamma * y2 + delta * y1 * y2
    
    Args:
        y (torch.Tensor): 상태 벡터 [y1, y2] (batch_size, 2)
        theta (torch.Tensor): 파라미터 벡터 [alpha, beta, gamma, delta] (batch_size, 4)
        t (torch.Tensor, optional): 시간 (사용되지 않음, 자율계)
    
    Returns:
        torch.Tensor: 상태 미분 벡터 [dy1/dt, dy2/dt] (batch_size, 2)
    """
    alpha, beta, gamma, delta = theta.unbind(dim=1)
    
    y1, y2 = y.unbind(dim=1)
    
    dy1_dt = alpha * y1 - beta * y1 * y2
    dy2_dt = -gamma * y2 + delta * y1 * y2
    
    return torch.stack([dy1_dt, dy2_dt], dim=1)


def sir_model_rhs(y, theta, t=None):
    """
    SIR 전염병 모델의 우변 함수
    
    dS/dt = -beta * S * I
    dI/dt = beta * S * I - gamma * I
    dR/dt = gamma * I
    
    Args:
        y (torch.Tensor): 상태 벡터 [S, I, R] (batch_size, 3)
        theta (torch.Tensor): 파라미터 벡터 [beta, gamma] (batch_size, 2)
        t (torch.Tensor, optional): 시간 (사용되지 않음, 자율계)
    
    Returns:
        torch.Tensor: 상태 미분 벡터 [dS/dt, dI/dt, dR/dt] (batch_size, 3)
    """
    beta, gamma = theta.unbind(dim=1)
    
    S, I, R = y.unbind(dim=1)
    
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    
    return torch.stack([dS_dt, dI_dt, dR_dt], dim=1)


def lorenz_system_rhs(y, theta, t=None):
    """
    Lorenz 시스템의 우변 함수
    
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    
    Args:
        y (torch.Tensor): 상태 벡터 [x, y, z] (batch_size, 3)
        theta (torch.Tensor): 파라미터 벡터 [sigma, rho, beta] (batch_size, 3)
        t (torch.Tensor, optional): 시간 (사용되지 않음, 자율계)
    
    Returns:
        torch.Tensor: 상태 미분 벡터 [dx/dt, dy/dt, dz/dt] (batch_size, 3)
    """
    sigma, rho, beta = theta.unbind(dim=1)
    
    x, y, z = y.unbind(dim=1)
    
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    
    return torch.stack([dx_dt, dy_dt, dz_dt], dim=1)


def pendulum_rhs(y, theta, t=None):
    """
    비선형 진자의 우변 함수
    
    d(theta)/dt = omega
    d(omega)/dt = -g/l * sin(theta) - b*omega
    
    Args:
        y (torch.Tensor): 상태 벡터 [theta, omega] (batch_size, 2)
        theta (torch.Tensor): 파라미터 벡터 [g/l, b] (batch_size, 2)
        t (torch.Tensor, optional): 시간 (사용되지 않음, 자율계)
    
    Returns:
        torch.Tensor: 상태 미분 벡터 [d(theta)/dt, d(omega)/dt] (batch_size, 2)
    """
    g_over_l, b = theta.unbind(dim=1)
    
    theta_val, omega = y.unbind(dim=1)
    
    dtheta_dt = omega
    domega_dt = -g_over_l * torch.sin(theta_val) - b * omega
    
    return torch.stack([dtheta_dt, domega_dt], dim=1)


def harmonic_oscillator_rhs(y, theta, t=None):
    """
    조화 진동자의 우변 함수
    
    dx/dt = v
    dv/dt = -k/m * x - b/m * v
    
    Args:
        y (torch.Tensor): 상태 벡터 [x, v] (batch_size, 2)
        theta (torch.Tensor): 파라미터 벡터 [k/m, b/m] (batch_size, 2)
        t (torch.Tensor, optional): 시간 (사용되지 않음, 자율계)
    
    Returns:
        torch.Tensor: 상태 미분 벡터 [dx/dt, dv/dt] (batch_size, 2)
    """
    k_over_m, b_over_m = theta.unbind(dim=1)
    
    x, v = y.unbind(dim=1)
    
    dx_dt = v
    dv_dt = -k_over_m * x - b_over_m * v
    
    return torch.stack([dx_dt, dv_dt], dim=1)


def get_ode_rhs(ode_type):
    """
    ODE 유형에 따라 적절한 우변 함수 반환
    
    Args:
        ode_type (str): ODE 유형 이름
        
    Returns:
        callable: 해당 ODE 유형의 우변 함수
    """
    rhs_dict = {
        'lotka_volterra': lotka_volterra_rhs,
        'sir_model': sir_model_rhs,
        'lorenz_system': lorenz_system_rhs,
        'pendulum': pendulum_rhs,
        'harmonic_oscillator': harmonic_oscillator_rhs
    }
    
    if ode_type not in rhs_dict:
        raise ValueError(f"지원하지 않는 ODE 유형: {ode_type}")
    
    return rhs_dict[ode_type]


def get_parameter_dim(ode_type):
    """
    ODE 유형에 따른 파라미터 차원 반환
    
    Args:
        ode_type (str): ODE 유형 이름
        
    Returns:
        int: 해당 ODE 유형의 파라미터 차원
    """
    param_dims = {
        'lotka_volterra': 4,  # alpha, beta, gamma, delta
        'sir_model': 2,       # beta, gamma
        'lorenz_system': 3,   # sigma, rho, beta
        'pendulum': 2,        # g/l, b
        'harmonic_oscillator': 2  # k/m, b/m
    }
    
    if ode_type not in param_dims:
        raise ValueError(f"지원하지 않는 ODE 유형: {ode_type}")
    
    return param_dims[ode_type]


def get_state_dim(ode_type):
    """
    ODE 유형에 따른 상태 벡터 차원 반환
    
    Args:
        ode_type (str): ODE 유형 이름
        
    Returns:
        int: 해당 ODE 유형의 상태 벡터 차원
    """
    state_dims = {
        'lotka_volterra': 2,     # y1, y2
        'sir_model': 3,          # S, I, R
        'lorenz_system': 3,      # x, y, z
        'pendulum': 2,           # theta, omega
        'harmonic_oscillator': 2  # x, v
    }
    
    if ode_type not in state_dims:
        raise ValueError(f"지원하지 않는 ODE 유형: {ode_type}")
    
    return state_dims[ode_type] 