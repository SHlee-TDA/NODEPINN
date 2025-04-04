#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
다양한 파라미터 θ에 대한 ODE trajectory 생성 스크립트
"""

import os
import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import argparse
import yaml
import pickle
from pathlib import Path


def lotka_volterra(t, y, alpha, beta, gamma, delta):
    """
    Lotka-Volterra ODE 시스템
    dy1/dt = alpha*y1 - beta*y1*y2
    dy2/dt = -gamma*y2 + delta*y1*y2
    """
    y1, y2 = y
    dy1_dt = alpha * y1 - beta * y1 * y2
    dy2_dt = -gamma * y2 + delta * y1 * y2
    return [dy1_dt, dy2_dt]


def generate_trajectory(theta, y0, t_span, t_eval):
    """
    지정된 파라미터와 초기값으로 ODE trajectory 생성
    """
    alpha, beta, gamma, delta = theta
    sol = solve_ivp(
        lambda t, y: lotka_volterra(t, y, alpha, beta, gamma, delta),
        t_span=t_span,
        y0=y0,
        method='RK45',
        t_eval=t_eval
    )
    return sol.t, sol.y


def sample_parameters(n_samples, param_ranges):
    """
    파라미터 공간에서 무작위 샘플링
    """
    theta_samples = []
    for _ in range(n_samples):
        alpha = np.random.uniform(*param_ranges['alpha'])
        beta = np.random.uniform(*param_ranges['beta'])
        gamma = np.random.uniform(*param_ranges['gamma'])
        delta = np.random.uniform(*param_ranges['delta'])
        theta_samples.append((alpha, beta, gamma, delta))
    return theta_samples


def main(args):
    # 설정 파일 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 경로 생성
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # 파라미터 샘플링
    param_ranges = config['parameter_ranges']
    theta_samples = sample_parameters(config['n_samples'], param_ranges)
    
    # 데이터셋 생성
    dataset = []
    for i, theta in enumerate(theta_samples):
        # 초기값 샘플링
        y0 = [
            np.random.uniform(*config['initial_condition_ranges']['y1']),
            np.random.uniform(*config['initial_condition_ranges']['y2'])
        ]
        
        # trajectory 생성
        t_span = (0, config['t_max'])
        t_eval = np.linspace(0, config['t_max'], config['n_points'])
        t, y = generate_trajectory(theta, y0, t_span, t_eval)
        
        # 데이터셋에 추가
        dataset.append({
            'theta': theta,
            'y0': y0,
            't': t,
            'y': y
        })
        
        # 시각화 (샘플링된 몇 개 확인용)
        if i < config['n_vis_samples']:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(t, y[0], 'b-', label='y1(t)')
            plt.plot(t, y[1], 'r-', label='y2(t)')
            plt.xlabel('Time')
            plt.ylabel('Population')
            plt.legend()
            plt.title(f'Trajectory (α={theta[0]:.2f}, β={theta[1]:.2f}, γ={theta[2]:.2f}, δ={theta[3]:.2f})')
            
            plt.subplot(1, 2, 2)
            plt.plot(y[0], y[1], 'g-')
            plt.xlabel('y1')
            plt.ylabel('y2')
            plt.title('Phase Portrait')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'sample_{i}.png'))
            plt.close()
    
    # 데이터셋 저장
    with open(os.path.join(args.output_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f'생성된 데이터셋 크기: {len(dataset)} 샘플')
    print(f'데이터셋 저장 위치: {os.path.join(args.output_dir, "dataset.pkl")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ODE 데이터셋 생성')
    parser.add_argument('--config', type=str, default='../configs/data_config.yaml', 
                        help='설정 파일 경로')
    parser.add_argument('--output_dir', type=str, default='./generated', 
                        help='출력 디렉토리')
    args = parser.parse_args()
    
    main(args) 