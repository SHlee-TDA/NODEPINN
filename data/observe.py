import numpy as np

class ObservationModel:
    """
    ODE 시스템의 관측 모델
    실제 궤적에 노이즈를 추가하여 관측 데이터를 생성
    """
    def __init__(self, noise_std=0.1):
        """
        Args:
            noise_std (float): 관측 노이즈의 표준편차
        """
        self.noise_std = noise_std
    
    def sample(self, sol, time_points=None):
        """
        ODE 해에서 노이즈가 있는 관측값을 샘플링합니다.
        
        Args:
            sol: ODE 솔루션 객체 (scipy.integrate.solve_ivp 의 반환값)
            time_points: 관측할 시간점들. None이면 sol.t의 모든 시간점 사용
        
        Returns:
            tuple: (obs_times, observations) - 관측 시간과 관측값
        """
        if time_points is None:
            # 모든 시간점에서 관측
            time_points = sol.t
            observations = sol.y.T  # shape: (time_points, state_dim)
        else:
            # 각 상태 변수를 time_points에 대해 선형 보간
            observations = np.transpose(np.array([
                np.interp(time_points, sol.t, sol.y[i, :])
                for i in range(sol.y.shape[0])
            ]))
        
        # 관측 노이즈 추가 (가우시안 노이즈)
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=observations.shape)
            observations = observations + noise
        
        return time_points, observations
