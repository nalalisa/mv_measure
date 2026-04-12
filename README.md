# 🫀 3D TEE Mitral Valve Quantification Prototype

본 프로젝트는 3D 경식도 심초음파(3D TEE) 딥러닝 Segmentation 복셀(Voxel) 데이터로부터 승모판막(Mitral Valve) 및 대동맥판막(Aortic Valve)의 3D 기하학적 랜드마크와 임상 지표를 자동 산출하는 **수학적/기하학적 정량화 엔진의 파이썬(Python) 프로토타입**입니다.

## 🎯 Architecture Philosophy (C++ 포팅을 위한 설계 철학)

이 파이썬 코드는 일반적인 데이터 사이언스 스크립트가 아닙니다. **가까운 미래에 C++ 3D 뷰어 엔진으로 1:1 포팅될 것을 전제로 엄격하게 설계되었습니다.**

1.  **Strict Typing & Data Classes:** 모든 데이터 모델은 `@dataclass`와 `numpy` 타입 힌트를 강제하여 작성되었습니다. 이는 C++의 `struct` 및 `Eigen::Vector3d` 구조와 정확히 일치합니다.
2.  **No Pythonic Magic:** 동적 타이핑이나 파이썬 특유의 축약 문법을 배제하고, 명시적인 행렬 연산(Matrix Operations)과 절차적 흐름을 유지했습니다.
3.  **Stateless Algorithms:** 알고리즘 모듈은 상태(State)를 가지지 않는 순수 함수(Pure Function)들로 구성되어 있어, C++의 `static` 함수나 `namespace` 내 독립 함수로 분리하기 쉽습니다.

---

## 📂 Project Structure

```text
mv_quantification_prototyper/
├── README.md                 # 현재 문서 (프로젝트 개요 및 포팅 가이드)
├── mv_engine/                # 핵심 정량화 엔진 (C++ namespace: mv_engine)
│   ├── core/
│   │   └── data_models.py    # [C++ struct] Plane3D, MitralLandmarks 등 순수 데이터 구조
│   ├── math/
│   │   ├── fourier.py        # [C++ Eigen] 푸리에 급수 OLS 피팅 (계수 산출 및 복원)
│   │   └── geometry.py       # [C++ Eigen] 좌표계 변환, 벡터 내적/외적, 평면 투영
│   ├── algo/
│   │   ├── preprocessing.py  # [C++ PCL VoxelGrid] 밀도 편향 제거용 다운샘플링
│   │   ├── plane_fitter.py   # Annulus 중심 평면 및 3D 닫힌 곡선 추출
│   │   ├── commissure.py     # [C++ nanoflann] 최단거리 기반 교차점 탐색
│   │   ├── landmarks.py      # 직교축 투영 기반 A, P 포인트 도출
│   │   └── tenting.py        # 접합부 Tenting Height, Area, Volume 연산
│   └── pipeline/
│       └── mv_analyzer.py    # [C++ Controller Class] 위 알고리즘들을 조립하는 메인 파이프라인
├── viz/
│   └── visualizer.py         # 파이썬 검증 전용 3D 시각화 (C++ 엔진과는 무관함)
└── tests/
    └── test_algo.py          # ★ C++ 포팅 시 무결성 교차 검증을 위한 하드코딩 단위 테스트
```

---

## 🧠 Core Algorithms & Justifications (왜 이렇게 짰는가?)

딥러닝 Segmentation 마스크는 필연적으로 **오탐지(Noise), 결손(Missing data), 두께 뭉침(Density Skew)**을 동반합니다. 본 엔진의 알고리즘들은 이러한 오류에 흔들리지 않도록 고안된 '강건한(Robust)' 기하학적 파이프라인입니다.

### 1. Voxel Downsampling (밀도 편향 통제)
* **How:** 모든 Voxel 마스크를 일정한 Grid 크기로 다운샘플링하여 1격자당 1개의 점만 남깁니다.
* **Why (The Problem):** 초음파 음영이나 석회화(MAC)로 인해 특정 부위에 픽셀이 비정상적으로 뭉쳐있으면, 단순 평균(PCA, Center of Mass) 알고리즘은 픽셀이 많은 쪽으로 끌려가 심각하게 왜곡됩니다.
* **Solution:** 다운샘플링을 통해 두께나 노이즈에 상관없이 공간상의 "점 밀도"를 강제로 균일하게 만들어 기하학적 형태의 순수성만 남깁니다.

### 2. Fourier Series Plane Fitting (평면 추출)
* **How:** 점들을 원통 좌표계로 변환한 뒤, 최소제곱법(OLS)을 사용해 Z축 높이를 각도 `theta`에 대한 푸리에 급수(2~3차)로 피팅합니다. 여기서 0차 항과 1차 항만 빼내어 최종 평면으로 사용합니다.
* **Why (The Problem):** 단순 평면 피팅(PCA, RANSAC)이나 Skeletonization 후 평면을 구하는 방식은 데이터의 일부가 끊겨 있거나, 판막 탈출증(Prolapse)으로 한쪽이 위로 솟아있으면 평면 전체가 시소처럼 틀어져 버립니다(시소 효과).
* **Solution:** 푸리에 급수는 승모판을 거시적인 '파동(Wave)'으로 이해합니다. 솟아오른 병변이나 노이즈는 고주파(3차 이상)로 격리하여 버리고, 끊긴 구간은 자연스럽게 보간하여 **"가장 완벽한 기본 기울기(1차 항)"**만을 평면으로 추출할 수 있습니다. (TomTec 등 임상 표준 방식)

### 3. Distance Sum Minimization (Commissure 추출)
* **How:** 정제된 Annulus 3D 곡선 위를 360도로 돌면서, 마스크 1(Anterior)과 마스크 2(Posterior)까지의 거리의 합(`D1 + D2`)이 가장 작아지는 두 개의 깊은 골짜기(Local Minima)를 찾아냅니다.
* **Why (The Problem):** 딥러닝 마스크 사이에 미세한 틈이 있거나 노이즈로 경계가 겹쳐 있으면 교차점을 특정하기 어렵습니다.
* **Solution:** "Commissure는 두 Leaflet이 링 위에서 동시에 가장 가까워지는 지점이다"라는 해부학적 정의를 수학적으로 가장 완벽하게 구현한 전역 탐색(Global Search) 방식입니다.

### 4. Orthogonal AP Axis Projection (A, P 랜드마크 추출)
* **How:** 추출된 Commissure Line에 직교(Orthogonal)하면서 Annulus 중심을 지나는 선을 그어 A점과 P점을 찾습니다. (Anterior 무게중심은 단지 '앞뒤 방향'을 결정하는 데만 사용합니다.)
* **Why (The Problem):** Anterior Leaflet의 무게중심을 직접 이용해 축을 잡으면, 판막 비대칭성 때문에 축이 틀어질 위험이 큽니다.
* **Solution:** 해부학적으로 가장 뚜렷하고 흔들리지 않는 랜드마크인 'Commissure 교차선'을 1차 기준으로 삼고 직교축을 내리는 것이 수학적으로 가장 안전합니다.

### 5. Mid-surface Triangulation (Leaflet 면적/길이 연산)
* **How:** 판막 복셀의 두께를 깎아내어(Skeletonization) 중앙 곡면(Mid-surface)으로 만든 뒤 3D 면적을 적분합니다.
* **Why (The Problem):** 복셀 개수를 그대로 세거나 외곽선을 쓰면, 딥러닝이 마스크를 두껍게 칠했을 때 면적이 2~3배 과대평가됩니다. 중앙 곡면 기반 연산만이 두께 오차를 완벽히 상쇄합니다.

---

## 🛠️ C++ Porting Guide (라이브러리 매핑표)

파이썬 환경에서 검증된 로직을 C++ 뷰어 엔진으로 포팅할 때, 다음의 C++ 라이브러리를 사용하십시오.

| 파이썬 패키지 (현재) | C++ 권장 라이브러리 (타겟) | 용도 및 포팅 전략 |
| :--- | :--- | :--- |
| `@dataclass` | `struct` | C++의 구조체(Struct) 또는 단순 데이터 클래스(POD)로 1:1 매핑 |
| `numpy (np.ndarray)` | `Eigen3` (Eigen::Vector3d) | 모든 3D 좌표 및 방향 벡터 보관 |
| `np.linalg.lstsq` | `Eigen::MatrixXd::bdcSvd` | 푸리에 피팅을 위한 최소제곱법 (의사역행렬 풀이) |
| `np.linalg.eigh` | `Eigen::SelfAdjointEigenSolver` | PCA를 위한 공분산 행렬의 고윳값/고유벡터 추출 |
| `scipy.spatial.cKDTree` | `nanoflann` | Commissure 및 Tenting 추출 시 초고속 최단거리 검색 (PCL KdTree보다 가벼워 엔진 탑재 유리) |
| `open3d.geometry.VoxelDownSample` | `PCL` (VoxelGrid filter) | Point Cloud 밀도 균일화 (간단한 형태이므로 C++ 직접 구현도 무방) |

## 🚀 How to Run & Verify

1. 파이썬 환경 설정: `pip install -r requirements.txt`
2. 샘플 데이터 연산 및 시각화: `python main_verify.py`
3. **[C++ 개발자 주의사항]:** C++ 포팅이 완료되면, 반드시 `tests/test_algo.py`에 명시된 하드코딩된 입력 점(Point) 데이터를 C++ 엔진에 넣고, 출력된 Normal Vector와 계산된 수치가 파이썬 결과와 소수점 4자리까지 일치하는지 교차 검증(Cross-validation)해야 합니다.
