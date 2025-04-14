import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List, Optional

class DataLoader:
    def __init__(self, data_dir: Union[str, Path] = "generated"):
        self.data_dir = Path(data_dir)
        
    def load_dataset(self, system_name: str, file_stem: Optional[str] = None) -> Dict:
        """시스템 데이터셋을 로드합니다.
        
        Args:
            system_name: 시스템 이름 (예: 'lotka_volterra')
            file_stem: 파일 이름 (기본값: system_name에서 '_'를 제거한 값)
            
        Returns:
            Dict: 로드된 데이터셋
        """
        if file_stem is None:
            file_stem = system_name.replace('_', '')
            
        path = self.data_dir / system_name / f'{file_stem}_dataset.npz'
        return np.load(path, allow_pickle=True)
    
    def convert_to_dataframe(self, npz_data: Dict) -> pd.DataFrame:
        """NPZ 데이터를 pandas DataFrame으로 변환합니다.
        
        Args:
            npz_data: np.load로 로드된 데이터
            
        Returns:
            pd.DataFrame: 변환된 데이터프레임
        """
        params = npz_data['params']
        x0 = npz_data['x0']
        samples = npz_data['samples']
        time_steps = npz_data['time_steps']
        sols = npz_data['sols']
        
        N, T, D = samples.shape
        records = []
        
        for i in range(N):
            try:
                true_traj = sols[i].y.T.tolist()
            except Exception:
                true_traj = None
            
            row = {
                'sample_idx': i,
                'x0': list(x0[i]),
                'params': list(params[i]),
                'times': list(time_steps[i]),
                'true_trajectory': true_traj
            }
            
            for d in range(D):
                row[f'state_{d}'] = list(samples[i, :, d])
                
            records.append(row)
            
        return pd.DataFrame(records)
    
    def get_train_val_test_split(self, df: pd.DataFrame, 
                                val_ratio: float = 0.1,
                                test_ratio: float = 0.1,
                                random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """데이터를 train/val/test 세트로 분할합니다.
        
        Args:
            df: 전체 데이터프레임
            val_ratio: 검증 세트 비율
            test_ratio: 테스트 세트 비율
            random_state: 랜덤 시드
            
        Returns:
            Dict[str, pd.DataFrame]: 분할된 데이터프레임 딕셔너리
        """
        from sklearn.model_selection import train_test_split
        
        # 먼저 train과 나머지로 분할
        train_df, temp_df = train_test_split(
            df, test_size=val_ratio + test_ratio, random_state=random_state
        )
        
        # 나머지를 val과 test로 분할
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=1-val_size, random_state=random_state
        )
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        } 