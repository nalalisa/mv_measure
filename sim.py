import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

def generate_saddle_data(num_points=300, radius=10, height=3, noise_level=0.0, gap_ratio=0.0):
    """3차원 안장 형태의 점군 데이터를 생성하고 노이즈와 결손을 적용합니다."""
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = height * np.cos(2 * theta)
    
    if gap_ratio > 0:
        gap_start = np.pi - (gap_ratio * np.pi)
        gap_end = np.pi + (gap_ratio * np.pi)
        mask = (theta < gap_start) | (theta > gap_end)
        x, y, z, theta = x[mask], y[mask], z[mask], theta[mask]
        
    if noise_level > 0:
        x += np.random.normal(0, noise_level, len(x))
        y += np.random.normal(0, noise_level, len(y))
        z += np.random.normal(0, noise_level, len(z))
        
    return x, y, z

def fit_circle_algebraic(x, y):
    """
    Kasa Method를 이용한 기하학적 원 피팅 (Robust Fitting).
    데이터가 호(Arc) 형태만 남아있어도 전체 원의 중심을 역산해냅니다.
    수식: x^2 + y^2 + ax + by + c = 0
    """
    A = np.column_stack([x, y, np.ones(len(x))])
    B = -(x**2 + y**2)
    # 최소제곱법으로 a, b, c 계수 도출
    coef, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    # 원의 중심점 좌표 계산: cx = -a/2, cy = -b/2
    cx = -coef[0] / 2
    cy = -coef[1] / 2
    return cx, cy

def fit_fourier_3d(x, y, z, order, center_method='mean'):
    """선택된 방식에 따라 중심점을 구하고 푸리에 피팅을 수행합니다."""
    # 1. 중심점 도출 방식 선택
    if center_method == 'Robust Fit':
        cx, cy = fit_circle_algebraic(x, y)
    else: # 'Simple Mean'
        cx, cy = np.mean(x), np.mean(y)
    
    # 2. 중심 이동 및 원통 좌표계 변환
    dx, dy = x - cx, y - cy
    theta_pts = np.arctan2(dy, dx)
    r_pts = np.sqrt(dx**2 + dy**2)
    
    # 3. 푸리에 피팅
    A = [np.ones(len(theta_pts))]
    for k in range(1, int(order) + 1):
        A.append(np.cos(k * theta_pts))
        A.append(np.sin(k * theta_pts))
    A = np.column_stack(A)
    
    c_r, _, _, _ = np.linalg.lstsq(A, r_pts, rcond=None)
    c_z, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    
    # 4. 곡선 재구성
    t_smooth = np.linspace(-np.pi, np.pi, 200)
    A_smooth = [np.ones(len(t_smooth))]
    for k in range(1, int(order) + 1):
        A_smooth.append(np.cos(k * t_smooth))
        A_smooth.append(np.sin(k * t_smooth))
    A_smooth = np.column_stack(A_smooth)
    
    r_fit = A_smooth @ c_r
    z_fit = A_smooth @ c_z
    
    x_fit = cx + r_fit * np.cos(t_smooth)
    y_fit = cy + r_fit * np.sin(t_smooth)
    
    return x_fit, y_fit, z_fit, cx, cy

# --- 시각화 UI 설정 ---
fig = plt.figure(figsize=(12, 8))
ax3d = fig.add_axes([0.05, 0.25, 0.7, 0.7], projection='3d') # 3D 뷰어 크기 조정

# 슬라이더
ax_order = fig.add_axes([0.15, 0.15, 0.5, 0.03])
ax_noise = fig.add_axes([0.15, 0.10, 0.5, 0.03])
ax_gap = fig.add_axes([0.15, 0.05, 0.5, 0.03])

s_order = Slider(ax_order, 'Fourier Order', 1, 8, valinit=3, valstep=1, color='lightblue')
s_noise = Slider(ax_noise, 'Noise Level', 0.0, 3.0, valinit=0.5, color='lightgreen')
s_gap = Slider(ax_gap, 'Gap Ratio', 0.0, 0.7, valinit=0.0, color='salmon')

# 라디오 버튼 (중심점 계산 방식 선택)
ax_radio = fig.add_axes([0.75, 0.05, 0.2, 0.15], facecolor='lightgray')
radio_center = RadioButtons(ax_radio, ('Simple Mean', 'Robust Fit'))

def update(val):
    order = int(s_order.val)
    noise = s_noise.val
    gap = s_gap.val
    center_method = radio_center.value_selected
    
    ax3d.cla()
    x, y, z = generate_saddle_data(noise_level=noise, gap_ratio=gap)
    
    x_fit, y_fit, z_fit, cx, cy = fit_fourier_3d(x, y, z, order, center_method)
    
    # 원본 데이터 시각화
    ax3d.scatter(x, y, z, c='gray', s=10, alpha=0.5, label='Voxel Data')
    
    # 푸리에 곡선
    line_color = 'red' if center_method == 'Simple Mean' else 'green'
    ax3d.plot(x_fit, y_fit, z_fit, c=line_color, linewidth=3, label=f'Fourier Fit')
    
    # 계산된 중심점 표시
    ax3d.scatter([cx], [cy], [0], c='blue', marker='X', s=150, label=f'Center ({center_method})')
    
    # 원래의 진정한 0,0,0 중심 표시 (참조용)
    ax3d.scatter([0], [0], [0], c='black', marker='+', s=100, label='True Origin')
    
    ax3d.set_xlim([-15, 15])
    ax3d.set_ylim([-15, 15])
    ax3d.set_zlim([-8, 8])
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title("Mitral Annulus Fitting: Mean vs Robust Center")
    ax3d.legend(loc='upper right')
    
    fig.canvas.draw_idle()

# 이벤트 연결
s_order.on_changed(update)
s_noise.on_changed(update)
s_gap.on_changed(update)
radio_center.on_clicked(update)

update(None)
plt.show()
