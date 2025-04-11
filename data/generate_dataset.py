import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import random
import torch
# 스크립트 직접 실행 시 상위 디렉토리를 path에 추가
if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 절대 경로 import 사용
from data.base import BaseODESystem
from data.observe import ObservationModel

def set_seed(seed):
    """
    재현성을 위한 시드 설정
    
    Args:
        seed (int): 시드 값
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"시드가 {seed}로 설정되었습니다.")

set_seed(42)


# 디렉토리 생성 함수 추가
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# matplotlib 설정
plt.switch_backend('agg')  # 그래픽 환경이 없는 경우를 위해

# 예시: Lotka-Volterra (LV) 방정식
class LotkaVolterra(BaseODESystem):
    def __init__(self, alpha=1.0, beta=0.1, delta=0.075, gamma=1.5):
        """
        Lotka-Volterra ODE 시스템.
        상태 변수: [prey, predator] → state_dim = 2
        파라미터: [alpha, beta, delta, gamma] → param_dim = 4
        """
        super().__init__(state_dim=2, param_dim=4)
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma

    def f(self, t, x, params):
        """
        Lotka-Volterra 방정식:
            dx/dt = alpha*x - beta*x*y
            dy/dt = delta*x*y - gamma*y
        params가 주어지면 그것을 사용 (순서: alpha, beta, delta, gamma)
        """
        if params is not None:
            alpha, beta, delta, gamma = params
        else:
            alpha, beta, delta, gamma = self.alpha, self.beta, self.delta, self.gamma
        dxdt = alpha * x[0] - beta * x[0] * x[1]
        dydt = delta * x[0] * x[1] - gamma * x[1]
        return [dxdt, dydt]

    def sample_initial_condition(self, init_cond=None):
        """
        Lotka-Volterra는 개체 수이므로 양의 값이어야 함.
        사용자가 주지 않으면 [0.5, 1.5] 구간에서 균등 샘플링.
        """
        if init_cond is not None:
            return init_cond
        x0 = np.random.uniform(8, 12)
        y0 = np.random.uniform(4, 6)
        return x0, y0

    def sample_parameters(self, params=None):
        """
        
        """
        if params is not None:
            return params
        else:
            alpha = np.random.uniform(0.8, 1.2)
            beta  = np.random.uniform(0.08, 0.12)
            gamma = np.random.uniform(1.2, 1.8)
            delta = np.random.uniform(0.06, 0.09)
        return alpha, beta, delta, gamma
    
    def generate_sample(self, time_span, t_eval):
        """
        하나의 trajectory 샘플을 생성합니다.
        - time_span: (start_time, end_time)
        - t_eval: 시간 포인트들 (numpy array)
        반환:
        - sol: solve_ivp 객체 (trajectory 포함)
        - x0: 초기 조건 (prey, predator)
        - params: 파라미터 (alpha, beta, gamma, delta)
        """
        x0 = self.sample_initial_condition()
        params = self.sample_parameters()
        sol = self.solve(time_span, t_eval=t_eval, x0=x0, params=params)
        return sol, x0, params

# 2차원 선형 시스템 (2D Linear System)
class LinearSystem2D(BaseODESystem):
    def __init__(self, A=None):
        """
        2차원 선형 시스템: dx/dt = Ax
        상태 변수: [x, y] → state_dim = 2
        파라미터: A의 원소들 (a11, a12, a21, a22) → param_dim = 4
        """
        super().__init__(state_dim=2, param_dim=4)
        # 기본 행렬 A 설정 (안정적인 나선 형태)
        if A is None:
            self.A = np.array([[-0.5, -0.5], [0.5, -0.5]])  # 나선 형태의 안정한 시스템
        else:
            self.A = A

    def f(self, t, x, params=None):
        """
        2차원 선형 시스템: dx/dt = Ax
        params가 주어지면 A 행렬을 재구성 (a11, a12, a21, a22)
        """
        if params is not None:
            a11, a12, a21, a22 = params
            A = np.array([[a11, a12], [a21, a22]])
        else:
            A = self.A
        
        return A @ x

    def sample_initial_condition(self, init_cond=None):
        """
        초기 조건 샘플링
        """
        if init_cond is not None:
            return init_cond
        x0 = np.random.uniform(-2, 2)
        y0 = np.random.uniform(-2, 2)
        return x0, y0

    def sample_parameters(self, params=None):
        """
        파라미터 (A 행렬 원소) 샘플링
        """
        if params is not None:
            return params
        
        # 랜덤 패턴 생성 - 안정적인 시스템 또는 불안정한 시스템
        pattern = np.random.choice(['stable_spiral', 'unstable_spiral', 'saddle', 'stable_node'])
        
        if pattern == 'stable_spiral':
            # 안정적인 나선
            a11 = np.random.uniform(-0.8, -0.2)
            a22 = np.random.uniform(-0.8, -0.2)
            a12 = np.random.uniform(-1.0, -0.2)
            a21 = np.random.uniform(0.2, 1.0)
        elif pattern == 'unstable_spiral':
            # 불안정한 나선
            a11 = np.random.uniform(0.2, 0.8)
            a22 = np.random.uniform(0.2, 0.8)
            a12 = np.random.uniform(-1.0, -0.2)
            a21 = np.random.uniform(0.2, 1.0)
        elif pattern == 'saddle':
            # 안장점
            a11 = np.random.uniform(0.2, 1.0)
            a22 = np.random.uniform(-1.0, -0.2)
            a12 = np.random.uniform(-0.5, 0.5)
            a21 = np.random.uniform(-0.5, 0.5)
        else:  # stable_node
            # 안정적인 노드
            a11 = np.random.uniform(-1.0, -0.2)
            a22 = np.random.uniform(-1.0, -0.2)
            a12 = np.random.uniform(-0.5, 0.5)
            a21 = np.random.uniform(-0.5, 0.5)
            
        return a11, a12, a21, a22
    
    def generate_sample(self, time_span, t_eval):
        """
        하나의 trajectory 샘플을 생성합니다.
        """
        x0 = self.sample_initial_condition()
        params = self.sample_parameters()
        sol = self.solve(time_span, t_eval=t_eval, x0=x0, params=params)
        return sol, x0, params

# 로렌츠 오실레이터 (Lorenz Attractor)
class LorenzSystem(BaseODESystem):
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        """
        로렌츠 시스템 (3차원 카오스 시스템)
        상태 변수: [x, y, z] → state_dim = 3
        파라미터: [sigma, rho, beta] → param_dim = 3
        """
        super().__init__(state_dim=3, param_dim=3)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def f(self, t, x, params=None):
        """
        로렌츠 방정식:
            dx/dt = sigma * (y - x)
            dy/dt = x * (rho - z) - y
            dz/dt = x * y - beta * z
        """
        if params is not None:
            sigma, rho, beta = params
        else:
            sigma, rho, beta = self.sigma, self.rho, self.beta
        
        dxdt = sigma * (x[1] - x[0])
        dydt = x[0] * (rho - x[2]) - x[1]
        dzdt = x[0] * x[1] - beta * x[2]
        
        return [dxdt, dydt, dzdt]

    def sample_initial_condition(self, init_cond=None):
        """
        초기 조건 샘플링
        """
        if init_cond is not None:
            return init_cond
        # 로렌츠 시스템은 초기값에 매우 민감, 작은 범위에서 샘플링
        x0 = np.random.uniform(-5, 5)
        y0 = np.random.uniform(-5, 5)
        z0 = np.random.uniform(0, 10)
        return x0, y0, z0

    def sample_parameters(self, params=None):
        """
        파라미터 샘플링
        """
        if params is not None:
            return params
        
        # 카오스 발생하는 범위 내에서 샘플링
        sigma = np.random.uniform(9.0, 11.0)
        rho = np.random.uniform(25.0, 30.0)
        beta = np.random.uniform(2.5, 3.0)
        
        return sigma, rho, beta
    
    def generate_sample(self, time_span, t_eval):
        """
        하나의 trajectory 샘플을 생성합니다.
        """
        x0 = self.sample_initial_condition()
        params = self.sample_parameters()
        sol = self.solve(time_span, t_eval=t_eval, x0=x0, params=params)
        return sol, x0, params

def generate_and_visualize(ode_system, 
                           num_samples=3, 
                           time_span=(0, 10), 
                           dt=0.1, 
                           num_obs_points=5, 
                           noise_std=0.2, 
                           output_dir='generated',
                           save_format='npz'):
    """
    ODE 시스템에 대한 데이터 생성 및 시각화 함수
    
    Args:
        ode_system: BaseODESystem 클래스의 인스턴스
        num_samples: 생성할 샘플 수
        time_span: 시간 범위 (시작, 끝)
        dt: 시간 간격
        num_obs_points: 관측할 시간점 개수
        noise_std: 관측 노이즈 표준편차
        output_dir: 결과 저장 디렉토리
    """
    # 출력 디렉토리 생성
    ensure_dir(output_dir)
    
    # 결과 저장 변수
    samples = []
    sols = []
    time_steps = []
    x0_list = []
    params_list = []
    t_eval = np.arange(time_span[0], time_span[1] + dt, dt)
    
    # 관측 모델 초기화
    obs_model = ObservationModel(noise_std=noise_std)
    
    # 뚜렷한 색상 팔레트
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 샘플 생성
    for i in range(num_samples):
        sol, x0, params = ode_system.generate_sample(time_span, t_eval)
        sols.append(sol)
        
        # 균등 간격으로 관측 시간점 선택
        obs_times = np.linspace(time_span[0], time_span[1], num_obs_points)
        
        # 해당 시간점에서 관측
        t_obs, obs = obs_model.sample(sol, time_points=obs_times)
        time_steps.append(t_obs)
        samples.append(obs)
        x0_list.append(x0)
        params_list.append(params)
    
    # ODE 시스템에 따른 다른 시각화 방법
    if ode_system.state_dim == 2:  # 2차원 시스템 (Lotka-Volterra, Linear System)
        visualize_2d_system(ode_system, sols, samples, time_steps, distinct_colors, output_dir)
    elif ode_system.state_dim == 3:  # 3차원 시스템 (Lorenz)
        visualize_3d_system(ode_system, sols, samples, time_steps, distinct_colors, output_dir)
    
    # 데이터셋 저장
    dataset = {
        'time_span': time_span,
        'dt': dt,
        'sols': sols,
        'time_steps': time_steps,
        'samples': samples,
        'system_type': ode_system.__class__.__name__,
        'params': params_list,
        'x0': x0_list
    }
    
    # 시스템 이름 가져오기
    system_name = ode_system.__class__.__name__.lower()
    
    # 저장 방식 분기
    if save_format == 'pkl':
        file_path = os.path.join(output_dir, f'{system_name}_dataset.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"피클 파일로 저장 완료: {file_path}")

    elif save_format == 'npz':
        npz_data = {
            'params': np.array(params_list),
            'x0': np.array(x0_list),
            'samples': np.array(samples),       # shape: (N, T, D)
            'time_steps': np.array(time_steps), # shape: (N, T)
            'sols': sols,
            'dt': dt,
            'system_type': ode_system.__class__.__name__,
        }
        file_path = os.path.join(output_dir, f'{system_name}_dataset.npz')
        np.savez(file_path, **npz_data)
        print(f"압축 npz 파일로 저장 완료: {file_path}")
    else:
        raise ValueError(f"지원하지 않는 저장 형식: {save_format}")
    
    return dataset

def visualize_2d_system(ode_system, sols, samples, time_steps, colors, output_dir):
    """2차원 ODE 시스템 시각화"""
    system_name = ode_system.__class__.__name__
    
    plt.figure(figsize=(16, 8))
    
    # 시간에 따른 상태 변수 변화
    plt.subplot(1, 2, 1)
    
    for i, sol in enumerate(sols):
        if i >= len(colors):
            break
            
        if system_name == 'LotkaVolterra':
            var_names = ['Prey', 'Predator']
        else:  # LinearSystem2D
            var_names = ['x', 'y']
            
        plt.plot(sol.t, sol.y[0, :], label=f'{var_names[0]} {i+1}', color=colors[i], linewidth=2)
        plt.plot(sol.t, sol.y[1, :], label=f'{var_names[1]} {i+1}', color=colors[i], linestyle='--', linewidth=2)
    
    for i, (t_obs, obs) in enumerate(zip(time_steps, samples)):
        if i >= len(colors):
            break
            
        plt.scatter(t_obs, obs[:, 0], label=f'{var_names[0]} Obs {i+1}', color=colors[i], marker='o', s=80, edgecolor='black', alpha=0.8)
        plt.scatter(t_obs, obs[:, 1], label=f'{var_names[1]} Obs {i+1}', color=colors[i], marker='s', s=80, edgecolor='black', alpha=0.8)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('State Variables', fontsize=12)
    plt.title(f'{system_name}: Time Series', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=10)
    
    # 위상 공간
    plt.subplot(1, 2, 2)
    
    for i, sol in enumerate(sols):
        if i >= len(colors):
            break
            
        plt.plot(sol.y[0, :], sol.y[1, :], label=f'True {i+1}', color=colors[i], linewidth=2)
    
    for i, obs in enumerate(samples):
        if i >= len(colors):
            break
            
        plt.scatter(obs[:, 0], obs[:, 1], label=f'Obs {i+1}', color=colors[i], marker='o', s=80, edgecolor='black', alpha=0.8)
    
    plt.xlabel(var_names[0], fontsize=12)
    plt.ylabel(var_names[1], fontsize=12)
    plt.title(f'{system_name}: Phase Space', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig(os.path.join(output_dir, f'{system_name.lower()}_visualization.png'), dpi=300)
    print(f"그래프가 {os.path.join(output_dir, f'{system_name.lower()}_visualization.png')}에 저장되었습니다.")

def visualize_3d_system(ode_system, sols, samples, time_steps, colors, output_dir):
    """3차원 ODE 시스템 시각화"""
    system_name = ode_system.__class__.__name__
    
    # 3D 시각화를 위한 설정
    plt.figure(figsize=(18, 12))
    
    # 시간에 따른 3개 변수 변화
    plt.subplot(2, 2, 1)
    for i, sol in enumerate(sols):
        if i >= len(colors):
            break
        plt.plot(sol.t, sol.y[0, :], label=f'x{i+1}', color=colors[i])
    
    for i, (t_obs, obs) in enumerate(zip(time_steps, samples)):
        if i >= len(colors):
            break
        plt.scatter(t_obs, obs[:, 0], color=colors[i], marker='o', s=50, edgecolor='black')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('x', fontsize=12)
    plt.title('Time Series - x', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for i, sol in enumerate(sols):
        if i >= len(colors):
            break
        plt.plot(sol.t, sol.y[1, :], label=f'y{i+1}', color=colors[i])
    
    for i, (t_obs, obs) in enumerate(zip(time_steps, samples)):
        if i >= len(colors):
            break
        plt.scatter(t_obs, obs[:, 1], color=colors[i], marker='o', s=50, edgecolor='black')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Time Series - y', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for i, sol in enumerate(sols):
        if i >= len(colors):
            break
        plt.plot(sol.t, sol.y[2, :], label=f'z{i+1}', color=colors[i])
    
    for i, (t_obs, obs) in enumerate(zip(time_steps, samples)):
        if i >= len(colors):
            break
        plt.scatter(t_obs, obs[:, 2], color=colors[i], marker='o', s=50, edgecolor='black')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('z', fontsize=12)
    plt.title('Time Series - z', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # 3D 위상 공간
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(2, 2, 4, projection='3d')
    
    for i, sol in enumerate(sols):
        if i >= len(colors):
            break
        ax.plot3D(sol.y[0, :], sol.y[1, :], sol.y[2, :], color=colors[i], label=f'True {i+1}', linewidth=1.5, alpha=0.7)
    
    for i, obs in enumerate(samples):
        if i >= len(colors):
            break
        ax.scatter3D(obs[:, 0], obs[:, 1], obs[:, 2], color=colors[i], s=50, edgecolor='black', label=f'Obs {i+1}')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    ax.set_title(f'{system_name}: 3D Phase Space', fontsize=14)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig(os.path.join(output_dir, f'{system_name.lower()}_visualization.png'), dpi=300)
    print(f"그래프가 {os.path.join(output_dir, f'{system_name.lower()}_visualization.png')}에 저장되었습니다.")

def main():
    """메인 함수: 다양한 ODE 시스템에 대한 데이터 생성 및 시각화"""
    # 각 ODE 시스템 별 설정
    systems = {
        'lotka_volterra': {
            'name': '로트카-볼테라 포식자-피식자 모델',
            'system': LotkaVolterra(),
            'time_span': (0, 10),
            'dt': 0.1,
            'noise_std': 0.2
        },
        'linear_2d': {
            'name': '2차원 선형 시스템',
            'system': LinearSystem2D(),
            'time_span': (0, 10),
            'dt': 0.1,
            'noise_std': 0.1
        },
        'lorenz': {
            'name': '로렌츠 오실레이터 (카오스 시스템)',
            'system': LorenzSystem(),
            'time_span': (0, 20),
            'dt': 0.05,
            'noise_std': 0.1
        }
    }
    
    # 처리할 시스템 선택 (주석 해제하여 선택)
    systems_to_process = [
        'lotka_volterra',
        'linear_2d',
        'lorenz'
    ]
    
    # 선택된 시스템 처리
    for system_key in systems_to_process:
        if system_key in systems:
            system_info = systems[system_key]
            print(f"\n--- {system_info['name']} 데이터 생성 및 시각화 ---")
            
            generate_and_visualize(
                ode_system=system_info['system'],
                num_samples=10000,
                time_span=system_info['time_span'],
                dt=system_info['dt'],
                num_obs_points=8,
                noise_std=system_info['noise_std'],
                output_dir=os.path.join('generated', system_key),
                save_format='npz'
            )
    
    print("\n모든 데이터 생성 및 시각화 완료!")

if __name__ == '__main__':
    main()