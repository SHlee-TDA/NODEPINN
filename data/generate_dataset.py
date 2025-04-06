import argparse
import numpy as np
import matplotlib.pyplot as plt
from base import BaseODESystem
from observe import ObservationModel
import os

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


def main():
    # 출력 디렉토리 생성
    output_dir = 'generated'
    ensure_dir(output_dir)
    
    num_samples = 3
    samples = []
    sols = []
    time_steps = []
    time_span = (0, 10)
    dt = 0.2
    t_eval = np.arange(time_span[0], time_span[1] + dt, dt)

    # 관측 모델 초기화 (노이즈 표준편차 설정)
    obs_model = ObservationModel(noise_std=0.2)  # 노이즈 감소
    
    # 더 구분하기 쉬운 색상 팔레트 정의
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 파랑, 주황, 초록
    
    for i in range(num_samples):
        lv_system = LotkaVolterra()
        sol, x0, params = lv_system.generate_sample(time_span, t_eval)
        sols.append(sol)

        # 관측할 시간점 선택 (5개만) - 다음 세 가지 방법 중 하나 선택
        
        # 방법 1: 균등 간격으로 5개 선택
        obs_times_uniform = np.linspace(time_span[0], time_span[1], 5)
        
        # 방법 2: 랜덤하게 5개 선택
        obs_times_random = np.sort(np.random.choice(sol.t, 5, replace=False))
        
        # 방법 3: 특정 구간에 집중된 5개 시간점 (예: 중간 부분에 집중)
        obs_times_focused = np.linspace(4, 6, 5)  # 시간 4-6 사이에 5개 점
        
        # 여기서 원하는 방법 선택 (주석 해제)
        selected_obs_times = obs_times_uniform  # 균등 간격
        # selected_obs_times = obs_times_random  # 랜덤 선택
        # selected_obs_times = obs_times_focused  # 특정 구간
        
        # 선택한 시간점에서만 관측 수행
        t_obs, obs = obs_model.sample(sol, time_points=selected_obs_times)
        time_steps.append(t_obs)
        samples.append(obs)

    # 결과 그래프 생성 - 2개의 subplot만 사용
    plt.figure(figsize=(16, 8))

    # (1) Time-series plot
    plt.subplot(1, 2, 1)
    for i, sol in enumerate(sols):
        plt.plot(sol.t, sol.y[0, :], label=f'Prey {i+1}', color=distinct_colors[i], linewidth=2)
        plt.plot(sol.t, sol.y[1, :], label=f'Predator {i+1}', color=distinct_colors[i], linestyle='--', linewidth=2)
    
    for i, (t_obs, obs) in enumerate(zip(time_steps, samples)):
        plt.scatter(t_obs, obs[:, 0], label=f'Prey Obs {i+1}', color=distinct_colors[i], marker='o', s=80, edgecolor='black', alpha=0.8)
        plt.scatter(t_obs, obs[:, 1], label=f'Predator Obs {i+1}', color=distinct_colors[i], marker='s', s=80, edgecolor='black', alpha=0.8)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Population', fontsize=12)
    plt.title('Lotka-Volterra: True Trajectories and Observations', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=10)

    # (2) Phase space plot
    plt.subplot(1, 2, 2)
    for i, sol in enumerate(sols):
        plt.plot(sol.y[0, :], sol.y[1, :], label=f'True {i+1}', color=distinct_colors[i], linewidth=2)
    
    for i, obs in enumerate(samples):
        plt.scatter(obs[:, 0], obs[:, 1], label=f'Obs {i+1}', color=distinct_colors[i], 
                   marker='o', s=80, edgecolor='black', alpha=0.8)
        
    plt.xlabel('Prey (x)', fontsize=12)
    plt.ylabel('Predator (y)', fontsize=12)
    plt.title('Phase Space: True Trajectories and Observations', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)

    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig(os.path.join(output_dir, 'lotka_volterra_samples.png'), dpi=300)
    print(f"그래프가 {os.path.join(output_dir, 'lotka_volterra_samples.png')}에 저장되었습니다.")
    
    # 생성된 데이터 저장
    dataset = {
        'time_span': time_span,
        'dt': dt,
        'sols': sols,
        'time_steps': time_steps,
        'samples': samples
    }
    
    # 데이터 저장 방법 (예시)
    import pickle
    with open(os.path.join(output_dir, 'lv_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    print(f"데이터셋이 {os.path.join(output_dir, 'lv_dataset.pkl')}에 저장되었습니다.")

if __name__ == '__main__':
    main()