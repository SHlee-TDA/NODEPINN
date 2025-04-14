from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from pathlib import Path

class BaseModel(nn.Module, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """모델의 forward pass를 정의합니다.
        
        Args:
            x: 입력 텐서
            
        Returns:
            torch.Tensor: 모델 출력
        """
        pass
    
    def save(self, path: Union[str, Path]):
        """모델을 저장합니다.
        
        Args:
            path: 저장 경로
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseModel':
        """모델을 로드합니다.
        
        Args:
            path: 모델 파일 경로
            
        Returns:
            BaseModel: 로드된 모델 인스턴스
        """
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_num_params(self) -> int:
        """모델의 파라미터 수를 반환합니다.
        
        Returns:
            int: 파라미터 수
        """
        return sum(p.numel() for p in self.parameters())
    
    def to_device(self, x: torch.Tensor) -> torch.Tensor:
        """텐서를 모델의 디바이스로 이동시킵니다.
        
        Args:
            x: 입력 텐서
            
        Returns:
            torch.Tensor: 디바이스로 이동된 텐서
        """
        return x.to(self.device) 