#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NODEPINN 실험 실행 스크립트
"""

import os
import sys
import argparse
import time
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# 프로젝트 루트 디렉토리 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.vector_field import create_vector_field
from src.models.param_estimator import create_param_estimator
from src.models.ode_rhs import get_ode_rhs, get_state_dim, get_parameter_dim
from src.training.trainer import Trainer
from src.training.evaluate import (
    evaluate_parameter_estimation, 
    plot_parameter_estimation, 
    compare_trajectories,
    evaluate_multiple_trajectories,
    lotka_volterra_scipy
)
from src.utils.misc import (
    set_seed, 
    load_config, 
    prepare_device, 
    setup_logger, 
    get_model_summary
)


def prepare_data(config):
    """
    데이터 준비 및 로드
    
    Args:
        config (dict): 설정
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger.info("데이터 로드 중...")
    
    # 데이터셋 로드
    dataset_path = config['data']['dataset_path']
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    logger.info(f"데이터셋 로드 완료: {len(dataset)} 샘플")
    
    # 데이터셋 분할 비율
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    # 데이터 인덱스 분할
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 데이터 준비 함수
    def prepare_tensors(indices):
        traj_list = []
        time_list = []
        param_list = []
        init_list = []
        
        for idx in indices:
            sample = dataset[idx]
            traj = torch.tensor(sample['y'].T, dtype=torch.float32)  # (seq_len, state_dim)
            time = torch.tensor(sample['t'], dtype=torch.float32)    # (seq_len,)
            param = torch.tensor(sample['theta'], dtype=torch.float32)  # (param_dim,)
            y0 = torch.tensor(sample['y0'], dtype=torch.float32)     # (state_dim,)
            
            traj_list.append(traj)
            time_list.append(time)
            param_list.append(param)
            init_list.append(y0)
        
        # 배치 차원 추가
        trajectories = torch.stack(traj_list)        # (batch_size, seq_len, state_dim)
        time_points = torch.stack(time_list)         # (batch_size, seq_len)
        parameters = torch.stack(param_list)         # (batch_size, param_dim)
        initial_conditions = torch.stack(init_list)  # (batch_size, state_dim)
        
        return TensorDataset(trajectories, time_points, parameters, initial_conditions)
    
    # 데이터셋 생성
    train_dataset = prepare_tensors(train_indices)
    val_dataset = prepare_tensors(val_indices)
    test_dataset = prepare_tensors(test_indices)
    
    logger.info(f"데이터 분할 - 학습: {len(train_dataset)}, 검증: {len(val_dataset)}, 테스트: {len(test_dataset)}")
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=config['data']['shuffle'],
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, test_dataset


def build_models(config, device):
    """
    모델 생성
    
    Args:
        config (dict): 설정
        device (torch.device): 학습 장치
        
    Returns:
        tuple: (vector_field, param_estimator)
    """
    logger.info("모델 구축 중...")
    
    # ODE 시스템 차원 가져오기
    ode_type = 'lotka_volterra'  # TODO: config에서 가져오기
    state_dim = get_state_dim(ode_type)
    param_dim = get_parameter_dim(ode_type)
    seq_len = 200  # TODO: config에서 가져오기
    
    # 벡터 필드 생성
    vector_field = create_vector_field(
        config=config['model']['vector_field'],
        state_dim=state_dim
    )
    vector_field = vector_field.to(device)
    
    # 파라미터 추정기 생성
    param_estimator = create_param_estimator(
        config=config['model']['param_estimator'],
        input_dim=state_dim,
        seq_len=seq_len,
        output_dim=param_dim
    )
    param_estimator = param_estimator.to(device)
    
    # 모델 요약 정보
    logger.info(get_model_summary(vector_field))
    logger.info(get_model_summary(param_estimator))
    
    return vector_field, param_estimator


def train_model(config, vector_field, param_estimator, train_loader, val_loader, device):
    """
    모델 학습
    
    Args:
        config (dict): 설정
        vector_field (nn.Module): 벡터 필드 모델
        param_estimator (nn.Module): 파라미터 추정 모델
        train_loader (DataLoader): 학습 데이터 로더
        val_loader (DataLoader): 검증 데이터 로더
        device (torch.device): 학습 장치
        
    Returns:
        Trainer: 학습된 트레이너 인스턴스
    """
    logger.info("모델 학습 시작...")
    
    # 트레이너 초기화
    trainer = Trainer(
        vector_field=vector_field,
        param_estimator=param_estimator,
        config=config,
        device=device
    )
    
    # 체크포인트에서 학습 재개
    checkpoint_dir = os.path.join(config['experiment']['checkpoint_dir'], 
                                 config['experiment']['name'])
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    if args.resume and os.path.exists(checkpoint_path):
        logger.info(f"체크포인트에서 학습 재개: {checkpoint_path}")
        start_epoch = trainer.load_checkpoint(checkpoint_path)
        remaining_epochs = config['training']['n_epochs'] - start_epoch - 1
        if remaining_epochs <= 0:
            logger.info("모든 에폭이 완료되었습니다. 추가 학습 없음.")
            return trainer
    else:
        remaining_epochs = None  # 전체 에폭 학습
    
    # 모델 학습
    start_time = time.time()
    metrics = trainer.train(train_loader, val_loader, n_epochs=remaining_epochs)
    training_time = time.time() - start_time
    
    logger.info(f"학습 완료. 소요 시간: {training_time:.2f} 초")
    
    # 학습 곡선 플롯
    results_dir = os.path.join('results', config['experiment']['name'])
    os.makedirs(results_dir, exist_ok=True)
    trainer.plot_training_curves(save_path=os.path.join(results_dir, 'training_curves.png'))
    
    return trainer


def evaluate_model(config, trainer, test_loader, test_dataset, device):
    """
    학습된 모델 평가
    
    Args:
        config (dict): 설정
        trainer (Trainer): 학습된 트레이너 인스턴스
        test_loader (DataLoader): 테스트 데이터 로더
        test_dataset (TensorDataset): 테스트 데이터셋
        device (torch.device): 학습 장치
    """
    logger.info("모델 평가 중...")
    
    # 결과 디렉토리 설정
    results_dir = os.path.join('results', config['experiment']['name'])
    os.makedirs(results_dir, exist_ok=True)
    
    # 파라미터 추정 모델
    param_estimator = trainer.param_estimator
    param_estimator.eval()
    
    # 전체 예측 수행
    all_true_params = []
    all_pred_params = []
    all_y0 = []
    
    with torch.no_grad():
        for batch in test_loader:
            trajectories, time_points, parameters, initial_conditions = [b.to(device) for b in batch]
            
            # 파라미터 추정
            pred_params = param_estimator(trajectories)
            
            # CPU로 이동 및 NumPy 변환
            all_true_params.append(parameters.cpu().numpy())
            all_pred_params.append(pred_params.cpu().numpy())
            all_y0.append(initial_conditions.cpu().numpy())
    
    # 배열 연결
    true_params = np.vstack(all_true_params)
    pred_params = np.vstack(all_pred_params)
    initial_conditions = np.vstack(all_y0)
    
    # 평가 수행
    metrics, param_df = evaluate_parameter_estimation(true_params, pred_params)
    
    # 결과 저장
    logger.info("파라미터 추정 성능:")
    logger.info(f"전체 MSE: {metrics['mse']:.6f}")
    logger.info(f"전체 RMSE: {metrics['rmse']:.6f}")
    logger.info(f"전체 MAE: {metrics['mae']:.6f}")
    logger.info(f"전체 R^2: {metrics['r2']:.6f}")
    
    logger.info("\n파라미터별 성능:")
    logger.info(param_df.to_string())
    
    # 파라미터 추정 시각화
    plot_parameter_estimation(
        true_params, 
        pred_params,
        save_path=os.path.join(results_dir, 'parameter_estimation.png')
    )
    
    # 궤적 비교
    logger.info("궤적 비교 중...")
    
    # 시간 그리드 설정
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 200)
    
    # ODE 함수 설정
    ode_fn = lotka_volterra_scipy
    
    # 여러 궤적 비교
    metrics = evaluate_multiple_trajectories(
        ode_fn=ode_fn,
        theta_true_list=true_params[:10],  # 처음 10개 샘플만
        theta_pred_list=pred_params[:10],
        y0_list=initial_conditions[:10],
        t_span=t_span,
        t_eval=t_eval,
        save_dir=os.path.join(results_dir, 'trajectories'),
        max_plots=5
    )
    
    logger.info("궤적 평가 결과:")
    logger.info(f"평균 궤적 MSE: {metrics['avg_trajectory_mse']:.6f}")
    logger.info(f"중앙값 궤적 MSE: {metrics['median_trajectory_mse']:.6f}")
    logger.info(f"최소 궤적 MSE: {metrics['min_trajectory_mse']:.6f}")
    logger.info(f"최대 궤적 MSE: {metrics['max_trajectory_mse']:.6f}")
    logger.info(f"샘플 수: {metrics['n_samples']}")
    
    logger.info(f"평가 결과가 {results_dir}에 저장되었습니다.")


def main(args):
    """
    메인 실행 함수
    
    Args:
        args (argparse.Namespace): 명령줄 인수
    """
    # 설정 로드
    config = load_config(args.config)
    
    # 시드 설정
    set_seed(config['experiment']['seed'])
    
    # 장치 설정
    device = prepare_device(config['experiment']['device'])
    
    # 데이터 준비
    train_loader, val_loader, test_loader, test_dataset = prepare_data(config)
    
    # 모델 구축
    vector_field, param_estimator = build_models(config, device)
    
    if not args.eval_only:
        # 모델 학습
        trainer = train_model(config, vector_field, param_estimator, train_loader, val_loader, device)
    else:
        # 평가 모드: 모델 로드
        logger.info("평가 모드: 저장된 모델 로드")
        trainer = Trainer(
            vector_field=vector_field,
            param_estimator=param_estimator,
            config=config,
            device=device
        )
        
        checkpoint_dir = os.path.join(config['experiment']['checkpoint_dir'], 
                                     config['experiment']['name'])
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
        else:
            logger.error(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
            return
    
    # 모델 평가
    evaluate_model(config, trainer, test_loader, test_dataset, device)


if __name__ == '__main__':
    # 인수 파싱
    parser = argparse.ArgumentParser(description='NODEPINN 학습 및 평가')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='설정 파일 경로')
    parser.add_argument('--resume', action='store_true',
                        help='이전 체크포인트에서 학습 재개')
    parser.add_argument('--eval_only', action='store_true',
                        help='평가만 수행 (학습 없음)')
    args = parser.parse_args()
    
    # 로그 설정
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('nodepinn', log_dir, 'experiment.log')
    
    logger.info("=" * 50)
    logger.info(f"실험 시작: 설정 파일 {args.config}")
    logger.info("=" * 50)
    
    # 실행
    main(args)
    
    logger.info("=" * 50)
    logger.info("실험 완료")
    logger.info("=" * 50) 