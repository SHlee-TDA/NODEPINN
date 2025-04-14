import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .base import BaseModel

class LotkaVolterraModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 모델 하이퍼파라미터
        self.input_dim = config.get('input_dim', 2)  # 상태 변수 수
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()
            ) for _ in range(self.num_layers - 1)]
        )
        
        # 파라미터 예측기
        self.param_predictor = nn.Linear(self.hidden_dim, 4)  # alpha, beta, gamma, delta
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """모델의 forward pass를 정의합니다.
        
        Args:
            x: 입력 텐서 (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (예측된 파라미터, 예측된 궤적)
        """
        batch_size, seq_len, _ = x.shape
        
        # 인코딩
        encoded = self.encoder(x)
        
        # 파라미터 예측
        params = self.param_predictor(encoded[:, -1])  # 마지막 상태에서 파라미터 예측
        
        # 궤적 예측
        predicted_traj = self.decoder(encoded)
        
        return params, predicted_traj
    
    def compute_loss(self, 
                    pred_params: torch.Tensor,
                    pred_traj: torch.Tensor,
                    true_params: torch.Tensor,
                    true_traj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """손실 함수를 계산합니다.
        
        Args:
            pred_params: 예측된 파라미터
            pred_traj: 예측된 궤적
            true_params: 실제 파라미터
            true_traj: 실제 궤적
            
        Returns:
            Dict[str, torch.Tensor]: 각 손실 항목
        """
        # 파라미터 손실
        param_loss = nn.MSELoss()(pred_params, true_params)
        
        # 궤적 손실
        traj_loss = nn.MSELoss()(pred_traj, true_traj)
        
        # 전체 손실
        total_loss = param_loss + traj_loss
        
        return {
            'total_loss': total_loss,
            'param_loss': param_loss,
            'traj_loss': traj_loss
        } 