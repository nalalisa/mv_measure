# 🫀 3D TEE Mitral Valve Quantification Prototype

본 프로젝트는 3D 경식도 심초음파(3D TEE)에서 생성된 **128 x 128 x 128 NRRD 라벨 볼륨**으로부터 승모판막(Mitral Valve)의 3D 기하학 랜드마크와 임상 정량 지표를 자동 산출하는 **C++ 포팅 우선 설계의 Python 프로토타입**입니다.

현재 구현은 **단일 frame(single-frame)** 기준의 측정값에 집중되어 있으며, 향후 C++ 3D 뷰어 엔진에서 동일한 연산 체인을 재현할 수 있도록 자료구조, 수학 연산, 모듈 경계를 명시적으로 유지합니다.

## 🎯 Architecture Philosophy

이 코드는 일반적인 Python 분석 스크립트가 아닙니다. **가까운 미래에 Eigen + PCL/nanoflann 기반 C++ 엔진으로 1:1 포팅될 것을 전제로 작성되었습니다.**

1. **Strict Typing & POD-like Data Models**
   `@dataclass`와 `numpy.ndarray` 타입 힌트를 사용해 C++의 `struct`, `Eigen::Vector3d`, `Eigen::MatrixXd`로 자연스럽게 대응되도록 구성했습니다.
2. **No Pythonic Magic**
   동적 타이핑, 과도한 메타프로그래밍, 과축약 리스트 내포를 지양하고, 행렬 연산과 절차적 흐름을 명시적으로 유지했습니다.
3. **Stateless Algorithms**
   주요 측정 모듈은 상태를 저장하지 않는 순수 함수 위주로 작성되어 C++의 `namespace` 함수 또는 `static` helper로 포팅하기 쉽습니다.
4. **Geometry-first Biomedical Modeling**
   딥러닝 segmentation의 오탐지, 누락, thickness bias를 그대로 신뢰하지 않고, annulus/leaflet/coaptation을 **강건한 기하학 문제**로 다시 모델링합니다.

---

## 📥 Input Contract

입력은 **3D NRRD 라벨 파일 1개**입니다.

### Supported volume assumptions

- Shape: `128 x 128 x 128`
- Data type: integer label volume
- Physical spacing: NRRD header의 `spacings` 또는 `space directions`에서 해석
- World coordinates: `space origin`이 있으면 반영, 없으면 `(0, 0, 0)` 사용

### Label mapping

| Label | Meaning |
| :--- | :--- |
| `0` | Background |
| `1` | Anterior leaflet |
| `2` | Posterior leaflet |
| `3` | Mitral annulus |
| `4` | Aortic annulus |

### Current ingestion path

- `mv_engine/io/nrrd_reader.py`
- `load_labeled_volume_from_nrrd(path)`
- `build_patient_valve_data_from_label_volume(label_volume)`
- `MitralValveAnalyzer.run_quantification_from_nrrd(path)`

---

## 📂 Project Structure

```text
mv_quantification_prototyper/
├── README.md
├── main_verify.py
├── requirements.txt
├── docs/
│   └── porting_guide.md
├── mv_engine/
│   ├── core/
│   │   ├── data_models.py      # POD-like dataclasses and grouped result models
│   │   └── types.py            # ndarray aliases for points, clouds, matrices
│   ├── io/
│   │   └── nrrd_reader.py      # NRRD loading, header parsing, label extraction
│   ├── math/
│   │   ├── fourier.py          # Fourier OLS fitting and reconstruction
│   │   └── geometry.py         # Plane/basis/area/angle/polyline helpers
│   ├── algo/
│   │   ├── preprocessing.py    # Voxel grid downsampling
│   │   ├── plane_fitter.py     # Fourier-based annulus plane and smooth closed curve
│   │   ├── commissure.py       # Distance-sum minima along annulus curve
│   │   ├── landmarks.py        # AP axis, A/P points
│   │   ├── annulus_metrics.py  # Annulus diameters, area, perimeter, trigone-based metrics
│   │   ├── coaptation.py       # Contact/open gap analysis and closure lines
│   │   ├── leaflet_metrics.py  # Leaflet area, profile angle, profile length
│   │   ├── tenting.py          # Tenting height, area, volume
│   │   └── measurements.py     # Common geometric measurement helpers
│   ├── pipeline/
│   │   └── mv_analyzer.py      # Main orchestrator / future C++ controller
│   └── utils/
│       └── logger.py
├── tests/
│   ├── test_algo.py
│   └── test_math.py
└── viz/
    └── visualizer.py
```

---

## 🔄 End-to-End Data Flow

이 프로젝트의 데이터 플로우는 아래 순서로 고정되어 있습니다.

### 1. NRRD loading

- `pynrrd`로 volume과 header를 읽습니다.
- `space directions`, `spacings`, `space origin`을 사용해 voxel index를 mm 단위 world coordinate로 변환합니다.

### 2. Label-to-point-cloud conversion

- `np.argwhere()`로 각 label의 voxel index를 추출합니다.
- index matrix에 direction matrix를 곱하고 origin을 더해 physical point cloud를 생성합니다.
- 결과:
  - `mask_anterior`
  - `mask_posterior`
  - `mask_annulus`
  - `mask_aortic_annulus`

### 3. Preprocessing

- 각 point cloud는 필요 시 voxel-grid downsampling으로 밀도 편향을 줄입니다.
- 이 단계는 PCL `VoxelGrid`로 직접 포팅 가능한 형태로 작성되어 있습니다.

### 4. Mitral annulus modeling

- MV annulus point cloud를 PCA로 초기 정렬합니다.
- 회전 좌표계에서 `(theta, z)`에 대해 Fourier OLS fitting을 수행합니다.
- 0차와 1차 항으로 annulus plane을 복원하고, 360-sample smooth annulus curve를 생성합니다.

### 5. Aortic annulus modeling

- Aortic annulus 역시 동일한 point-cloud 기반 plane/curve fitting 경로를 탑니다.
- 현재는 Ao-MV angle, intertrigonal logic, anterior curtain geometry 계산에 사용됩니다.

### 6. Landmark extraction

- `find_commissures()`가 MV annulus curve를 따라 돌며 anterior/posterior leaflet까지의 거리 합이 최소인 두 지점을 찾습니다.
- `extract_ap_points()`가 commissure line에 직교하는 AP axis를 만들고 A/P points를 찾습니다.
- `find_trigones()`가 aortic annulus에 가장 가까운 anterior annulus sector 두 점을 좌/우 trigone으로 정의합니다.

### 7. Measurement computation

- `annulus_metrics.py`가 annulus geometry 기반 metrics를 계산합니다.
- `coaptation.py`가 closure line, open gap, prolapse, gap area를 계산합니다.
- `tenting.py`가 tenting height/area/volume을 계산합니다.
- `leaflet_metrics.py`가 leaflet surface surrogate area와 AP slice profile 길이/각도를 계산합니다.

### 8. Result packaging

- 결과는 category별 dataclass로 묶입니다.
- 최종 반환 타입: `MitralValveQuantificationResult`
  - `landmarks`
  - `annulus_metrics`
  - `shape_metrics`
  - `coaptation_metrics`
  - `leaflet_metrics`
  - `misc_metrics`

---

## 🧠 Why These Models Were Chosen

딥러닝 segmentation 마스크는 본질적으로 다음 문제를 가집니다.

- **Noise**: 초음파 artifact로 인한 잘못된 voxel
- **Missing data**: leaflet edge/annulus 일부 단절
- **Thickness bias**: leaflet가 과도하게 두껍게 segmentation되는 현상
- **Density skew**: 특정 부위 voxel density가 과하게 높아지는 현상

따라서 본 프로젝트는 raw voxel count 대신, **의미 있는 해부학 구조를 다시 기하학적으로 복원하는 방식**을 사용합니다.

### Fourier annulus modeling

단순 PCA plane 또는 RANSAC plane은 prolapse/noise에 쉽게 끌립니다. Fourier 기반 saddle reconstruction은:

- 끊긴 고리를 부드럽게 보간하고
- 고주파 noise를 저차/고차 성분으로 분리하며
- 기본 기울기만 annulus plane으로 보존할 수 있어

임상적 saddle annulus를 더 잘 반영합니다.

### Distance-sum commissure search

Commissure는 "두 leaflet에 동시에 가장 가까운 annulus sector"라는 해부학 정의를 가집니다. 이를 직접 최소화 문제로 만든 것이 현재 구현입니다.

### Central-band coaptation pairing

Leaflet 전체를 무작정 최근접점으로 연결하면 annulus insertion이나 외측 leaflet body까지 gap으로 오인될 수 있습니다. 현재 구현은:

- AP axis를 기준으로 중앙 band만 추출하고
- posterior leaflet과의 nearest-neighbor pair를 만들고
- contact threshold / search radius를 분리하여

closure line과 open gap을 동시에 계산합니다.

### Local PCA + Delaunay leaflet surface surrogate

현재 단일-frame prototype은 아직 full skeletonization이나 explicit mesh reconstruction을 쓰지 않습니다. 대신:

- local PCA plane으로 leaflet point cloud를 2D에 내리고
- Delaunay triangulation을 수행한 뒤
- 비정상적으로 긴 triangle을 제거해 area를 적분합니다.

이는 두꺼운 voxel mask를 직접 세는 것보다 훨씬 안정적이며, 향후 C++ mesh/surface pipeline으로 교체하기 쉬운 중간 단계입니다.

---

## 📏 Implemented Single-Frame Measurements

아래 항목들은 **현재 코드에 구현되어 있으며**, dynamic measurement는 intentionally 제외했습니다.

### Annulus results

| Metric | Current implementation |
| :--- | :--- |
| `AP Diameter` | A-P annulus landmarks의 **plane-projected chord length** |
| `AL-PM Diameter` | AL/PM commissure points 간 **3D chord length** |
| `Sphericity Index` | `AP Diameter / Commissural Diameter` |
| `Intertrigonal Distance` | aortic annulus proximity로 찾은 좌/우 trigone 사이의 3D 거리 |
| `Commissural Diameter` | commissure pair의 **plane-projected chord length** |
| `Saddle-Shaped Annulus Area (3D)` | smooth annulus curve에 대한 fan triangulation area |
| `Saddle-Shaped Annulus Perimeter (3D)` | 3D annulus curve polyline perimeter |
| `D-Shaped Annulus Area (2D)` | posterior arc + intertrigonal chord polygon의 2D area |
| `D-Shaped Annulus Perimeter` | posterior arc 2D length + trigone chord length |

### Shape results

| Metric | Current implementation |
| :--- | :--- |
| `Annulus Height` | annulus curve의 plane signed distance range (`max - min`) |
| `Non-planar Angle` | highest / lowest annulus points를 중심에서 본 vector angle |
| `Tenting Volume` | plane 아래 leaflet points의 depth 적분 x voxel column area, `ml` 단위 변환 |
| `Coaptation Depth` | coaptation contact midline 중 가장 깊은 점의 plane distance |
| `Tenting Area` | A-P 중심 단면에서 leaflet lower envelope와 annulus line 사이의 2D 적분 |
| `Ao-MV Angle` | MV plane normal과 Ao plane normal 사이의 acute angle |

### Coaptation results

| Metric | Current implementation |
| :--- | :--- |
| `Maximum Prolapse Height` | leaflet points 중 annulus plane의 atrial side로 가장 높이 솟은 signed distance |
| `Maximal Open Coaptation Gap` | central band에서 anterior-posterior nearest-neighbor gap의 최대값 |
| `Maximal Open Coaptation Width` | open-gap pair들의 commissure-axis extent |
| `Total Open Coaptation Area (3D)` | open-gap pair midline segment에 대해 `gap x segment length` 누적 적분 |

### Leaflets results

| Metric | Current implementation |
| :--- | :--- |
| `Anterior Leaflet Area` | local PCA projection + filtered Delaunay triangulation 3D area |
| `Posterior Leaflet Area` | local PCA projection + filtered Delaunay triangulation 3D area |
| `Distal Anterior Leaflet Angle` | AP slice profile distal half chord의 plane angle |
| `Posterior Leaflet Angle` | posterior AP slice profile 전체 chord의 plane angle |
| `Anterior Leaflet Length` | anterior AP slice lower-envelope polyline length |
| `Posterior Leaflet Length` | posterior AP slice lower-envelope polyline length |

### Miscellaneous results

| Metric | Current implementation |
| :--- | :--- |
| `Annulus Area (2D)` | full annulus curve의 plane-projected polygon area |
| `Anterior Closure Line Length (2D)` | contact anterior line의 plane-projected polyline length |
| `Anterior Closure Line Length (3D)` | contact anterior line의 3D polyline length |
| `Posterior Closure Line Length (2D)` | contact posterior line의 plane-projected polyline length |
| `Posterior Closure Line Length (3D)` | contact posterior line의 3D polyline length |
| `C-Shaped Annulus` | anterior arc length가 intertrigonal chord보다 충분히 길면 `True` |

### Not included in current release

- Dynamic measurement results
  - `Annular Displacement (max)`
  - `Annular Displacement Velocity (max)`
  - `Tenting Volume Fraction`
  - `Annulus Area Fraction (2D)`

이 항목들은 반드시 multi-frame temporal tracking이 필요하므로 현재 단일-frame 프로토타입의 범위 밖입니다.

---

## 🛠️ C++ Porting Guide

| Python (prototype) | C++ target | Porting note |
| :--- | :--- | :--- |
| `@dataclass` | `struct` / POD class | grouped result structs로 직접 대응 |
| `numpy.ndarray (3,)` | `Eigen::Vector3d` | point / normal / direction vector |
| `numpy.ndarray (N, 3)` | `Eigen::MatrixXd` or `std::vector<Eigen::Vector3d>` | point cloud storage |
| `np.linalg.lstsq` | `Eigen::MatrixXd::bdcSvd()` | Fourier least squares |
| `np.linalg.eigh` | `Eigen::SelfAdjointEigenSolver` | PCA plane fitting |
| `scipy.spatial.cKDTree` | `nanoflann` | nearest-neighbor pairing and distance fields |
| `scipy.spatial.Delaunay` | `CGAL` or custom triangulation | leaflet surface surrogate triangulation |
| voxel-grid downsample | `PCL::VoxelGrid` or custom hash-grid | density normalization |
| NRRD parsing (`pynrrd`) | ITK / Teem / in-house NRRD reader | header spacing/origin/direction parsing |

### Porting notes by subsystem

- `mv_engine/io/nrrd_reader.py`
  - NRRD header parsing만 C++ 전용 I/O 계층으로 교체하면 나머지 point-cloud pipeline은 동일하게 유지할 수 있습니다.
- `mv_engine/algo/annulus_metrics.py`
  - circular index traversal, trigone selection, polygon/perimeter logic은 STL + Eigen만으로 그대로 옮길 수 있습니다.
- `mv_engine/algo/coaptation.py`
  - KDTree query 부분만 `nanoflann`으로 바꾸면 pairing 로직은 동일합니다.
- `mv_engine/algo/leaflet_metrics.py`
  - Delaunay 부분은 향후 더 정밀한 mesh pipeline으로 바꾸기 쉬운 임시 surrogate입니다.

---

## 🚀 How To Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the synthetic end-to-end verification case

```bash
python main_verify.py
```

### 3. Run a real NRRD case

```bash
python main_verify.py path/to/case.nrrd
```

### 4. Run tests

```bash
python -m pytest -q
```

---

## ✅ Verification Status

현재 테스트는 다음을 포함합니다.

- Fourier / PCA math unit tests
- synthetic 128³ label volume -> point cloud conversion test
- full single-frame quantification pipeline test
- synthetic `.nrrd` round-trip load and quantification test

즉, **point-cloud demo 수준이 아니라 실제 NRRD ingestion path까지 포함한 최소 검증 루프가 이미 연결된 상태**입니다.

---

## ⚠️ Current Limitations

1. 일부 metric은 아직 clinical workstation의 proprietary definition과 완전히 동일하지 않고, **C++ 엔진 탑재 전 biomedical validation**이 필요합니다.
2. leaflet area는 explicit mid-surface skeleton이 아닌 **PCA + triangulation surrogate**입니다.
3. coaptation/gap 계열은 segmentation thickness와 free-edge labeling quality에 영향을 받습니다.
4. `C-Shaped Annulus`는 현재 boolean heuristic이며, 추후 더 엄밀한 morphologic classifier로 교체할 수 있습니다.

그럼에도 불구하고 현재 구조는 **입력 계약, 데이터 플로우, 핵심 측정군, C++ 포팅 경로**가 모두 분리되어 있어, 다음 단계인 C++ 엔진 통합과 cross-validation에 바로 사용 가능한 상태입니다.
