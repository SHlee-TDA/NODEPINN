# NODEPINN: Neural ODE + PINN for Parameter Estimation

본 프로젝트는 Neural ODE와 Physics-Informed Neural Networks(PINN)을 결합하여 ODE 시스템의 파라미터를 추정하는 프레임워크를 구현합니다.

## 프로젝트 개요

NODEPINN은 다음과 같은 특징을 가집니다:

- Neural ODE를 활용한 미분방정식 시뮬레이션
- PINN 기반 물리법칙 제약조건 적용
- 데이터와 물리법칙을 모두 활용한 파라미터 추정
- Lotka-Volterra 등 다양한 ODE 시스템에 적용 가능

## 디렉토리 구조

```
📂 project_root/
│
├── data/
│   └── generate_dataset.py         ← 다양한 θ로 생성된 ODE trajectory 저장
│
├── src/
│   ├── models/
│   │   ├── vector_field.py         ← fθ: Neural ODE (dy/dt = f(y, t))
│   │   ├── param_estimator.py      ← ϕ: θ̂ = φ(y_seq)
│   │   └── ode_rhs.py              ← 실제 ODE 구조 (e.g. Lotka-Volterra)
│   │
│   ├── training/
│   │   ├── loss.py                 ← physics loss, data loss 등 정의
│   │   ├── trainer.py              ← 학습 루프, optimizer, logger 등
│   │   └── evaluate.py             ← test set에서 예측 정확도 평가
│   │
│   ├── utils/
│   │   └── misc.py                 ← gradient check, reproducibility 등
│
├── configs/
│   └── config.yaml                 ← 실험 설정 관리 (network, training, dataset 등)
│
├── experiments/
│   └── run_exp.py                  ← 실험 실행 스크립트
│
└── notebooks/
    └── analysis.ipynb             ← 시각화 및 결과 분석
```

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/username/NODEPINN.git
cd NODEPINN

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 사용 방법

1. 데이터셋 생성:
```bash
python data/generate_dataset.py
```

2. 모델 학습:
```bash
python experiments/run_exp.py --config configs/config.yaml
```

3. 결과 분석:
- notebooks/analysis.ipynb 참조

## 참고 문헌

- Neural Ordinary Differential Equations (Chen et al., 2018)
- Physics-Informed Neural Networks (Raissi et al., 2019) 