import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 수학적 코어 함수 모음 (실제 파이프라인 적용용)
# ==========================================

def generate_tilted_saddle(num_points=300, noise=0.0, gap=0.0, tilt_deg=30):
    """임의의 각도로 기울어지고 중심이 이동된 3D 안장 데이터 생성"""
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = 10 * np.cos(theta)
    y = 10 * np.sin(theta)
    z = 3 * np.cos(2 * theta)
    
    if gap > 0:
        gap_start = np.pi - (gap * np.pi)
        gap_end = np.pi + (gap * np.pi)
        mask = (theta < gap_start) | (theta > gap_end)
        x, y, z, theta = x[mask], y[mask], z[mask], theta[mask]
        
    pts = np.vstack((x, y, z))
    
    # 임의의 3D 회전 (Tilt) 적용
    rad = np.radians(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
    Ry = np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])
    rotated_pts = Ry @ Rx @ pts
    
    # 임의의 위치로 이동
    rotated_pts[0, :] += 5.0
    rotated_pts[1, :] -= 8.0
    rotated_pts[2, :] += 12.0
    
    if noise > 0:
        rotated_pts += np.random.normal(0, noise, rotated_pts.shape)
        
    return rotated_pts[0], rotated_pts[1], rotated_pts[2]

def apply_pca_alignment(x, y, z):
    """PCA를 이용해 3D 공간의 점군을 XY 평면으로 정렬 (Step 2)"""
    pts = np.vstack((x, y, z))
    mean = np.mean(pts, axis=1, keepdims=True)
    centered = pts - mean
    
    # 공분산 행렬 및 고유벡터 계산
    cov = np.cov(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 고윳값 정렬 (가장 작은 것이 법선 벡터 Z축이 됨)
    idx = eigenvalues.argsort()[::-1]   
    eigenvectors = eigenvectors[:,idx]
    
    # 방향 일관성을 위한 처리 (Z축이 항상 위를 향하도록)
    if eigenvectors[2, 2] < 0:
        eigenvectors[:, 2] *= -1
        
    # 정렬 (회전)
    aligned_pts = eigenvectors.T @ centered
    return aligned_pts[0], aligned_pts[1], aligned_pts[2], mean, eigenvectors

def robust_circle_fit_2d(x, y):
    """Kasa Method: 2D 평면에서 잘린 원의 기하학적 중심 찾기 (Step 2b)"""
    A = np.column_stack([x, y, np.ones(len(x))])
    B = -(x**2 + y**2)
    coef, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return -coef[0] / 2, -coef[1] / 2

def fit_fourier_1d(theta, values, order):
    """1D 푸리에 피팅 (Step 3)"""
    A = [np.ones(len(theta))]
    for k in range(1, int(order) + 1):
        A.append(np.cos(k * theta))
        A.append(np.sin(k * theta))
    A = np.column_stack(A)
    coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    return coeffs

def reconstruct_fourier_1d(coeffs, theta_new, order):
    A_new = [np.ones(len(theta_new))]
    for k in range(1, int(order) + 1):
        A_new.append(np.cos(k * theta_new))
        A_new.append(np.sin(k * theta_new))
    A_new = np.column_stack(A_new)
    return A_new @ coeffs

# ==========================================
# 2. 대시보드 UI 및 시각화 설정
# ==========================================

fig = plt.figure(figsize=(18, 10))
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.25, wspace=0.3, hspace=0.3)
fig.suptitle("3D Mitral Valve Annulus Fourier Reconstruction Pipeline", fontsize=18, fontweight='bold')

# 6개의 서브플롯 구성 (2x3 그리드)
ax1 = fig.add_subplot(231, projection='3d', title="Step 1. Raw Tilted Data")
ax2 = fig.add_subplot(232, projection='3d', title="Step 2. PCA Aligned & Robust Centered")
ax3 = fig.add_subplot(233, title="Step 3a. Cylindrical Fit (Radius r vs θ)")
ax4 = fig.add_subplot(234, projection='3d', title="Step 4. 3D Fit in Aligned Space")
ax5 = fig.add_subplot(235, projection='3d', title="Step 5. Inverse Transformed (Final)")
ax6 = fig.add_subplot(236, title="Step 3b. Cylindrical Fit (Height z vs θ)")

# 컨트롤 슬라이더 영역
ax_tilt = plt.axes([0.15, 0.15, 0.7, 0.02])
ax_gap = plt.axes([0.15, 0.11, 0.7, 0.02])
ax_noise = plt.axes([0.15, 0.07, 0.7, 0.02])
ax_order = plt.axes([0.15, 0.03, 0.7, 0.02])

s_tilt = Slider(ax_tilt, '1. 3D Tilt Angle', 0, 90, valinit=30, valstep=5)
s_gap = Slider(ax_gap, '2. Gap Ratio (Missing)', 0.0, 0.7, valinit=0.4, valstep=0.05)
s_noise = Slider(ax_noise, '3. Noise Level', 0.0, 3.0, valinit=0.8)
s_order = Slider(ax_order, '4. Fourier Order', 1, 6, valinit=3, valstep=1)

def set_3d_axes_equal(ax, limits=15):
    """3D 플롯의 축 스케일을 고정하여 왜곡 방지"""
    ax.set_xlim([-limits, limits])
    ax.set_ylim([-limits, limits])
    ax.set_zlim([-limits, limits])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def update(val=None):
    # 축 초기화
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.cla()
        
    order = int(s_order.val)
    gap = s_gap.val
    
    # ----------------------------------------------------
    # Step 1: Raw Data
    # ----------------------------------------------------
    x, y, z = generate_tilted_saddle(noise=s_noise.val, gap=gap, tilt_deg=s_tilt.val)
    ax1.scatter(x, y, z, c='gray', alpha=0.6, s=15)
    
    # 중심을 단순 평균으로 구했을 때의 오류 시각화
    naive_center = [np.mean(x), np.mean(y), np.mean(z)]
    ax1.scatter(*naive_center, c='red', s=100, marker='x', label="Naive Center (Wrong)")
    
    ax1.legend()
    set_3d_axes_equal(ax1, limits=np.max(x)+5)
    
    # ----------------------------------------------------
    # Step 2: PCA Alignment & Robust Centering
    # ----------------------------------------------------
    xa, ya, za, mean_vec, rot_mat = apply_pca_alignment(x, y, z)
    
    # 눕혀진 평면(XY)에서 Kasa Method로 정확한 중심 도출
    cx, cy = robust_circle_fit_2d(xa, ya)
    
    # 진짜 중심을 원점(0,0)으로 이동
    xa_c, ya_c, za_c = xa - cx, ya - cy, za
    
    ax2.scatter(xa_c, ya_c, za_c, c='dodgerblue', alpha=0.6, s=15)
    ax2.scatter(0, 0, 0, c='green', s=100, marker='*', label="Robust Center (Origin)")
    ax2.legend()
    set_3d_axes_equal(ax2, limits=15)

    # ----------------------------------------------------
    # Step 3: Cylindrical Conversion & 1D Fourier Fits
    # ----------------------------------------------------
    theta_pts = np.arctan2(ya_c, xa_c)
    r_pts = np.sqrt(xa_c**2 + ya_c**2)
    
    # 정렬 (Plotting을 위해 각도순 정렬)
    sort_idx = np.argsort(theta_pts)
    t_sort, r_sort, z_sort = theta_pts[sort_idx], r_pts[sort_idx], za_c[sort_idx]
    
    # 피팅
    c_r = fit_fourier_1d(t_sort, r_sort, order)
    c_z = fit_fourier_1d(t_sort, z_sort, order)
    
    # 예측 곡선 생성
    t_new = np.linspace(-np.pi, np.pi, 200)
    r_fit = reconstruct_fourier_1d(c_r, t_new, order)
    z_fit = reconstruct_fourier_1d(c_z, t_new, order)
    
    # 2D 플롯 (r vs theta)
    ax3.scatter(t_sort, r_sort, c='gray', s=10, alpha=0.5)
    ax3.plot(t_new, r_fit, c='orange', lw=3, label=f'r Fit (Order {order})')
    ax3.set_xlim([-np.pi, np.pi])
    ax3.set_ylim([0, 15])
    ax3.legend()
    
    # 2D 플롯 (z vs theta)
    ax6.scatter(t_sort, z_sort, c='gray', s=10, alpha=0.5)
    ax6.plot(t_new, z_fit, c='magenta', lw=3, label=f'z Fit (Order {order})')
    ax6.set_xlim([-np.pi, np.pi])
    ax6.set_ylim([-6, 6])
    ax6.legend()

    # ----------------------------------------------------
    # Step 4: Reconstruct 3D Fit in Aligned Space
    # ----------------------------------------------------
    x_fit_a = r_fit * np.cos(t_new)
    y_fit_a = r_fit * np.sin(t_new)
    z_fit_a = z_fit
    
    ax4.scatter(xa_c, ya_c, za_c, c='lightgray', alpha=0.3, s=10)
    ax4.plot(x_fit_a, y_fit_a, z_fit_a, c='blue', lw=3, label='Aligned 3D Fit')
    ax4.legend()
    set_3d_axes_equal(ax4, limits=15)

    # ----------------------------------------------------
    # Step 5: Inverse Transform to Original Space
    # ----------------------------------------------------
    # 1) Robust 중심 되돌리기 (cx, cy 더하기)
    pts_fit_centered = np.vstack((x_fit_a + cx, y_fit_a + cy, z_fit_a))
    
    # 2) PCA 회전 되돌리기 (Inverse Rotation = rot_mat * pts)
    pts_fit_rotated = rot_mat @ pts_fit_centered
    
    # 3) 초기 Mean 되돌리기
    pts_final = pts_fit_rotated + mean_vec
    
    x_final, y_final, z_final = pts_final[0], pts_final[1], pts_final[2]
    
    ax5.scatter(x, y, z, c='gray', alpha=0.4, s=15, label='Raw Data')
    ax5.plot(x_final, y_final, z_final, c='red', lw=4, label='Final Reconstructed Curve')
    ax5.legend()
    set_3d_axes_equal(ax5, limits=np.max(x)+5)

    fig.canvas.draw_idle()

# 이벤트 바인딩
s_tilt.on_changed(update)
s_gap.on_changed(update)
s_noise.on_changed(update)
s_order.on_changed(update)

# 초기화
update()
plt.show()
