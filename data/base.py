import numpy as np
from scipy.integrate import solve_ivp

class BaseODESystem:
    """
    ODE 시스템의 기본 클래스
    """
    def __init__(self, state_dim, param_dim):
        """
        Args:
            state_dim (int): 상태 벡터의 차원
            param_dim (int): 파라미터 벡터의 차원
        """
        self.state_dim = state_dim
        self.param_dim = param_dim
    
    def f(self, t, x, params=None):
        """
        ODE 시스템의 우변 함수 (dy/dt = f(t, y, params))
        
        Args:
            t (float): 시간
            x (list/array): 상태 벡터
            params (list/array, optional): 파라미터 벡터
            
        Returns:
            list: 미분값 [dx1/dt, dx2/dt, ...]
        """
        raise NotImplementedError("파생 클래스에서 구현해야 합니다")
    
    def solve(self, time_span, t_eval=None, x0=None, params=None, method='RK45'):
        """
        ODE 시스템을 수치적으로 풉니다.
        
        Args:
            time_span (tuple): 시간 범위 (t_start, t_end)
            t_eval (array, optional): 해를 구할 시간점들
            x0 (array, optional): 초기 조건
            params (array, optional): 파라미터 벡터
            method (str): 수치해법 방법
            
        Returns:
            object: ODE 해 객체
        """
        if x0 is None:
            x0 = self.sample_initial_condition()
        
        # ODE 시스템을 푸는 람다 함수
        def rhs(t, x):
            return self.f(t, x, params)
        
        # 수치적 풀이
        sol = solve_ivp(
            rhs,
            time_span,
            x0,
            method=method,
            t_eval=t_eval
        )
        
        return sol
    
    def sample_initial_condition(self, init_cond=None):
        """
        초기 조건을 샘플링합니다.
        
        Args:
            init_cond (array, optional): 명시적 초기 조건
            
        Returns:
            array: 샘플링된 초기 조건
        """
        if init_cond is not None:
            return init_cond
        else:
            # 기본적으로 [0, 1] 범위에서 균등 샘플링
            return np.random.uniform(0, 1, self.state_dim)
    
    def sample_parameters(self, params=None):
        """
        파라미터를 샘플링합니다.
        
        Args:
            params (array, optional): 명시적 파라미터
            
        Returns:
            array: 샘플링된 파라미터
        """
        if params is not None:
            return params
        else:
            # 기본적으로 [0, 1] 범위에서 균등 샘플링
            return np.random.uniform(0, 1, self.param_dim)
