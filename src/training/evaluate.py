#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
학습된 모델 평가 및 시각화 모듈
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import solve_ivp
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_parameter_estimation(theta_true, theta_pred, param_names=None):
    """
    파라미터 추정 성능 평가
    
    Args:
        theta_true (np.ndarray): 실제 파라미터 배열 (n_samples, param_dim)
        theta_pred (np.ndarray): 예측된 파라미터 배열 (n_samples, param_dim)
        param_names (list, optional): 파라미터 이름 목록
        
    Returns:
        dict: 평가 메트릭
        pandas.DataFrame: 파라미터별 메트릭
    """
    n_samples, param_dim = theta_true.shape
    
    # 기본 파라미터 이름
    if param_names is None:
        if param_dim == 4:  # Lotka-Volterra 가정
            param_names = ['alpha', 'beta', 'gamma', 'delta']
        else:
            param_names = [f'theta_{i+1}' for i in range(param_dim)]
    
    # 전체 MSE, MAE, R^2
    mse = mean_squared_error(theta_true, theta_pred)
    mae = mean_absolute_error(theta_true, theta_pred)
    r2 = r2_score(theta_true, theta_pred)  # 전체 R^2
    
    # 파라미터별 메트릭
    param_metrics = []
    for i in range(param_dim):
        param_mse = mean_squared_error(theta_true[:, i], theta_pred[:, i])
        param_mae = mean_absolute_error(theta_true[:, i], theta_pred[:, i])
        param_rmse = np.sqrt(param_mse)
        
        # 상대 오차 (MAPE)
        mape = np.mean(np.abs((theta_true[:, i] - theta_pred[:, i]) / (theta_true[:, i] + 1e-8))) * 100
        
        # 파라미터별 R^2
        param_r2 = r2_score(theta_true[:, i], theta_pred[:, i])
        
        # 상관 계수
        corr = np.corrcoef(theta_true[:, i], theta_pred[:, i])[0, 1]
        
        param_metrics.append({
            'parameter': param_names[i],
            'mse': param_mse,
            'rmse': param_rmse,
            'mae': param_mae,
            'mape': mape,
            'r2': param_r2,
            'corr': corr
        })
    
    # 데이터프레임으로 변환
    param_df = pd.DataFrame(param_metrics)
    
    # 전체 메트릭
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2
    }
    
    return metrics, param_df


def plot_parameter_estimation(theta_true, theta_pred, param_names=None, save_path=None):
    """
    파라미터 추정 결과 시각화
    
    Args:
        theta_true (np.ndarray): 실제 파라미터 배열 (n_samples, param_dim)
        theta_pred (np.ndarray): 예측된 파라미터 배열 (n_samples, param_dim)
        param_names (list, optional): 파라미터 이름 목록
        save_path (str, optional): 플롯 저장 경로
    """
    n_samples, param_dim = theta_true.shape
    
    # 기본 파라미터 이름
    if param_names is None:
        if param_dim == 4:  # Lotka-Volterra 가정
            param_names = ['alpha', 'beta', 'gamma', 'delta']
        else:
            param_names = [f'theta_{i+1}' for i in range(param_dim)]
    
    # 서브플롯 그리드 설정
    fig, axes = plt.subplots(2, param_dim, figsize=(4*param_dim, 8))
    
    # 최소/최대값 계산 (전체 파라미터에 대해)
    all_values = np.concatenate([theta_true.flatten(), theta_pred.flatten()])
    global_min, global_max = np.min(all_values), np.max(all_values)
    margin = (global_max - global_min) * 0.1
    plot_min, plot_max = global_min - margin, global_max + margin
    
    for i in range(param_dim):
        # 상단 행: 산점도
        ax = axes[0, i]
        ax.scatter(theta_true[:, i], theta_pred[:, i], alpha=0.6, s=30)
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--')  # 완벽한 예측 대각선
        ax.set_xlabel(f'True {param_names[i]}')
        ax.set_ylabel(f'Predicted {param_names[i]}')
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        
        # 파라미터별 R^2 및 RMSE 계산
        r2 = r2_score(theta_true[:, i], theta_pred[:, i])
        rmse = np.sqrt(mean_squared_error(theta_true[:, i], theta_pred[:, i]))
        ax.set_title(f'{param_names[i]}: R² = {r2:.3f}, RMSE = {rmse:.3f}')
        ax.grid(True, alpha=0.3)
        
        # 하단 행: 히스토그램 (오차 분포)
        ax = axes[1, i]
        errors = theta_pred[:, i] - theta_true[:, i]
        ax.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel(f'{param_names[i]} Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution: μ = {np.mean(errors):.3f}, σ = {np.std(errors):.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"파라미터 추정 플롯 저장됨: {save_path}")
    
    plt.show()


def compare_trajectories(ode_fn, theta_true, theta_pred, y0, t_span, t_eval, save_path=None):
    """
    실제 및 예측 파라미터로 생성된 ODE 궤적 비교
    
    Args:
        ode_fn (callable): ODE 우변 함수 (scipy 형식)
        theta_true (np.ndarray): 실제 파라미터 (샘플 1개)
        theta_pred (np.ndarray): 예측된 파라미터 (샘플 1개)
        y0 (np.ndarray): 초기 조건
        t_span (tuple): 시간 범위 (t_start, t_end)
        t_eval (np.ndarray): 평가 시간점
        save_path (str, optional): 플롯 저장 경로
    """
    # 실제 파라미터로 궤적 생성
    sol_true = solve_ivp(
        lambda t, y: ode_fn(t, y, *theta_true),
        t_span=t_span,
        y0=y0,
        method='RK45',
        t_eval=t_eval
    )
    
    # 예측 파라미터로 궤적 생성
    sol_pred = solve_ivp(
        lambda t, y: ode_fn(t, y, *theta_pred),
        t_span=t_span,
        y0=y0,
        method='RK45',
        t_eval=t_eval
    )
    
    # 궤적 차이 계산
    trajectory_mse = mean_squared_error(sol_true.y.T, sol_pred.y.T)
    
    # 시각화
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 첫 번째 플롯: 시간에 따른 상태 변수 (실제 vs 예측)
    for i in range(len(y0)):
        ax1.plot(sol_true.t, sol_true.y[i], 'b-', label=f'True y{i+1}(t)' if i == 0 else None)
        ax1.plot(sol_pred.t, sol_pred.y[i], 'r--', label=f'Pred y{i+1}(t)' if i == 0 else None)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('State')
    ax1.set_title(f'Trajectories (MSE = {trajectory_mse:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 두 번째 플롯: 상 공간 (Phase Space)
    if len(y0) >= 2:
        ax2.plot(sol_true.y[0], sol_true.y[1], 'b-', label='True')
        ax2.plot(sol_pred.y[0], sol_pred.y[1], 'r--', label='Predicted')
        ax2.set_xlabel('y1')
        ax2.set_ylabel('y2')
        ax2.set_title('Phase Portrait')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Phase portrait requires at least 2 dimensions',
                horizontalalignment='center', verticalalignment='center')
    
    # 세 번째 플롯: 상태별 오차
    for i in range(len(y0)):
        ax3.plot(sol_true.t, sol_pred.y[i] - sol_true.y[i], label=f'Error y{i+1}(t)')
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Error')
    ax3.set_title('Trajectory Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # 파라미터 정보 표시
    plt.figtext(0.5, 0.01, 
                f'True params: {theta_true} | Predicted params: {theta_pred}',
                ha='center', fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"궤적 비교 플롯 저장됨: {save_path}")
    
    plt.show()
    
    return trajectory_mse


def evaluate_multiple_trajectories(ode_fn, theta_true_list, theta_pred_list, y0_list, t_span, t_eval,
                                 save_dir=None, max_plots=5):
    """
    여러 파라미터 세트에 대한 궤적 비교 평가
    
    Args:
        ode_fn (callable): ODE 우변 함수 (scipy 형식)
        theta_true_list (list): 실제 파라미터 리스트
        theta_pred_list (list): 예측된 파라미터 리스트
        y0_list (list): 초기 조건 리스트
        t_span (tuple): 시간 범위 (t_start, t_end)
        t_eval (np.ndarray): 평가 시간점
        save_dir (str, optional): 플롯 저장 디렉토리
        max_plots (int): 생성할 최대 플롯 수
        
    Returns:
        dict: 평가 메트릭
    """
    n_samples = len(theta_true_list)
    trajectory_mses = []
    
    # 평가 디렉토리 생성
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 각 샘플에 대한 평가
    for i in range(min(n_samples, max_plots)):
        theta_true = theta_true_list[i]
        theta_pred = theta_pred_list[i]
        y0 = y0_list[i]
        
        # 저장 경로 설정
        save_path = os.path.join(save_dir, f'trajectory_comparison_{i+1}.png') if save_dir else None
        
        # 궤적 비교 및 MSE 계산
        mse = compare_trajectories(ode_fn, theta_true, theta_pred, y0, t_span, t_eval, save_path)
        trajectory_mses.append(mse)
    
    # 나머지 샘플에 대한 MSE만 계산 (플롯 없이)
    for i in range(max_plots, n_samples):
        theta_true = theta_true_list[i]
        theta_pred = theta_pred_list[i]
        y0 = y0_list[i]
        
        # 실제 파라미터로 궤적 생성
        sol_true = solve_ivp(
            lambda t, y: ode_fn(t, y, *theta_true),
            t_span=t_span,
            y0=y0,
            method='RK45',
            t_eval=t_eval
        )
        
        # 예측 파라미터로 궤적 생성
        sol_pred = solve_ivp(
            lambda t, y: ode_fn(t, y, *theta_pred),
            t_span=t_span,
            y0=y0,
            method='RK45',
            t_eval=t_eval
        )
        
        # MSE 계산
        mse = mean_squared_error(sol_true.y.T, sol_pred.y.T)
        trajectory_mses.append(mse)
    
    # 평가 메트릭 계산
    avg_mse = np.mean(trajectory_mses)
    std_mse = np.std(trajectory_mses)
    median_mse = np.median(trajectory_mses)
    
    metrics = {
        'avg_trajectory_mse': avg_mse,
        'std_trajectory_mse': std_mse,
        'median_trajectory_mse': median_mse,
        'min_trajectory_mse': np.min(trajectory_mses),
        'max_trajectory_mse': np.max(trajectory_mses),
        'n_samples': n_samples
    }
    
    # 히스토그램 플롯
    if save_dir:
        plt.figure(figsize=(10, 6))
        plt.hist(trajectory_mses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(avg_mse, color='r', linestyle='--', label=f'Mean MSE = {avg_mse:.4f}')
        plt.axvline(median_mse, color='g', linestyle='--', label=f'Median MSE = {median_mse:.4f}')
        plt.xlabel('Trajectory MSE')
        plt.ylabel('Frequency')
        plt.title('Distribution of Trajectory MSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'trajectory_mse_distribution.png'))
        plt.show()
    
    return metrics


def lotka_volterra_scipy(t, y, alpha, beta, gamma, delta):
    """
    Scipy 형식의 Lotka-Volterra 시스템 (evaluate_trajectory용)
    
    Args:
        t (float): 시간
        y (list): 상태 벡터 [y1, y2]
        alpha, beta, gamma, delta (float): 시스템 파라미터
        
    Returns:
        list: 미분 벡터 [dy1/dt, dy2/dt]
    """
    y1, y2 = y
    dy1_dt = alpha * y1 - beta * y1 * y2
    dy2_dt = -gamma * y2 + delta * y1 * y2
    return [dy1_dt, dy2_dt] 