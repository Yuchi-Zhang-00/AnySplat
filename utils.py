from pathlib import Path
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.misc.image_io import save_interpolated_video
from src.model.ply_export import export_ply
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image
import imageio
import numpy as np
from scipy.spatial import ConvexHull
import cv2


###############################################################################
#                            法线计算 svd/ransac                               #
###############################################################################


def fit_plane_ransac_safe(points, num_iters=500, dist_thresh=0.005, sample_N=20000):
    if points.shape[0] > sample_N:
        idx = np.random.choice(points.shape[0], sample_N, replace=False)
        pts = points[idx]
    else:
        idx = np.arange(points.shape[0])
        pts = points

    best_inliers = None
    best_count = 0
    best_normal = None
    best_center = None

    N = pts.shape[0]

    for _ in range(num_iters):
        ids = np.random.choice(N, 3, replace=False)
        p0, p1, p2 = pts[ids]

        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        d = -normal @ p0

        dist = np.abs(pts @ normal + d)
        inliers = dist < dist_thresh
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = normal
            best_center = pts[inliers].mean(axis=0)

    if best_normal is None:
        raise RuntimeError("RANSAC failed")
    
    print(f"num of points {points.shape[0]}   best count {best_count}")

    # refine normal
    pts_in = pts[best_inliers]
    pts_centered = pts_in - pts_in.mean(0)
    _, _, Vt = np.linalg.svd(pts_centered)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    
    n2 = np.array([0.0, normal[1], normal[2]])
    norm = np.linalg.norm(n2)
    if norm < 1e-6:
        raise ValueError("normal 几乎平行 X 轴，无法约束 nx=0")

    n2 /= norm

    # return normal, best_center, idx[best_inliers]
    return n2, best_center, idx[best_inliers]


def fit_plane_ransac_safe_2(points, num_iters=500, dist_thresh=0.005, sample_N=20000):
    if points.shape[0] > sample_N:
        idx = np.random.choice(points.shape[0], sample_N, replace=False)
        pts = points[idx]
    else:
        idx = np.arange(points.shape[0])
        pts = points

    best_inliers = None
    best_count = 0
    best_normal = None

    N = pts.shape[0]

    # ---------- RANSAC ----------
    for _ in range(num_iters):
        ids = np.random.choice(N, 3, replace=False)
        p0, p1, p2 = pts[ids]

        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        d = -normal @ p0

        dist = np.abs(pts @ normal + d)
        inliers = dist < dist_thresh
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = normal

    if best_normal is None:
        raise RuntimeError("RANSAC failed")

    print(f"num of points {points.shape[0]}   best count {best_count}")

    # ---------- refine normal ----------
    pts_in = pts[best_inliers]

    fit_center = pts_in.mean(axis=0)   # ⚠️ 只用于 SVD
    pts_centered = pts_in - fit_center

    _, _, Vt = np.linalg.svd(pts_centered)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal

    # ---------- 强制 nx = 0 ----------
    n2 = np.array([0.0, normal[1], normal[2]])
    norm = np.linalg.norm(n2)
    if norm < 1e-6:
        raise ValueError("normal 几乎平行 X 轴，无法约束 nx=0")
    n2 /= norm

    # ---------- 几何中心（平面内 AABB center） ----------
    # 1. 平面坐标系
    u, v = plane_coordinate_system(n2)

    # 2. 投影到平面
    proj = project_to_plane(pts_in, n2, fit_center)

    # 3. 转到 2D
    pts_2d = to_2d(proj, fit_center, u, v)

    # 4. 平面内 bbox center
    cx = (pts_2d[:, 0].min() + pts_2d[:, 0].max()) / 2
    cy = (pts_2d[:, 1].min() + pts_2d[:, 1].max()) / 2

    geometric_center = fit_center + cx * u + cy * v

    # ---------- 返回 ----------
    return n2, geometric_center, idx[best_inliers]


#  拟合桌面平面（拿法向量）  用 SVD / 最小二乘（比 RANSAC 简洁，mask 已知）
def fit_plane(points):
    if points.shape[0] < 50:
        raise ValueError("Too few points for plane fitting")
    
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    return normal, centroid


###############################################################################
#                               2D/3D 空间变换                                 #
###############################################################################

# mask内的像素投影到点云
def depth_to_points(depth, mask, fx, fy, cx, cy):

    v, u = np.where(mask > 0)
    z = depth[v, u]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    print(f"深度最大值 最小值 平均值  {z.min(), z.max(), z.mean()}")
    lower_bound = np.percentile(z, 10)
    upper_bound = np.percentile(z, 90)

    print(f"80% 的值分布在区间: [{lower_bound:.2f}, {upper_bound:.2f}]")
    # depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    # imageio.imwrite('./depth_visual2.png', depth_normalized)
    save_masked_depth_viz(depth, mask, z, "table_depth_only.png")
    return np.stack([x, y, z], axis=1)


# 把点投影到桌面平面
def project_to_plane(points, normal, point_on_plane):
    diff = points - point_on_plane
    dist = diff @ normal
    projected = points - np.outer(dist, normal)
    return projected

# 建立桌面上的 2D 坐标系 1 选两个正交轴：
def plane_coordinate_system(normal):
    # 找一个不平行的向量
    tmp = np.array([1,0,0]) if abs(normal[0]) < 0.9 else np.array([0,1,0])
    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v

# 将 3D 点 → 2D
def to_2d(points, origin, u, v):
    rel = points - origin
    x = rel @ u
    y = rel @ v
    return np.stack([x, y], axis=1)

def align_points_to_table(points, R_align, t_align):
    """
    points: (N,3) 世界坐标
    """
    return (R_align @ points.T).T + t_align
    # return (R_align @ points.T).T


###############################################################################
#                                对mask预处理                                  #
###############################################################################

# 计算最小內接矩形
def compute_inner_rect_mask(mask, erode_ksize=15, margin=10):
    kernel = np.ones((erode_ksize, erode_ksize), np.uint8)
    mask_eroded = cv2.erode(mask, kernel)

    ys, xs = np.where(mask_eroded > 0)
    if len(xs) == 0:
        raise ValueError("Eroded mask is empty")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    inner = np.zeros_like(mask, dtype=np.uint8)
    inner[
        y_min+margin : y_max-margin,
        x_min+margin : x_max-margin
    ] = 1
    return inner

def filter_points_by_inner_rect_3d(points_cam, normal, keep_ratio=0.6):
    """
    在平面内做 inner rectangle，返回筛选后的点
    """
    # 1. 平面坐标系
    u, v = plane_coordinate_system(normal)

    center = points_cam.mean(axis=0)
    rel = points_cam - center
    pts_2d = np.stack([rel @ u, rel @ v], axis=1)

    # 2. 用分位数定义 inner rectangle
    x, y = pts_2d[:, 0], pts_2d[:, 1]
    q = (1 - keep_ratio) / 2 * 100

    x_min, x_max = np.percentile(x, [q, 100 - q])
    y_min, y_max = np.percentile(y, [q, 100 - q])

    keep = (
        (x > x_min) & (x < x_max) &
        (y > y_min) & (y < y_max)
    )

    return points_cam[keep]

def shrink_mask_erode(mask, ratio=0.1):
    """
    mask: uint8, 0 / 1 或 0 / 255
    ratio: 收缩比例（相对于 mask 尺寸）
    """
    mask = (mask > 0).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask")

    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1

    k = int(min(h, w) * ratio)
    k = max(k, 3) | 1  # 保证奇数 & >=3

    kernel = np.ones((k, k), np.uint8)
    mask_eroded = cv2.erode(mask, kernel)

    return mask_eroded

###############################################################################
#                                深度信息渲染                                   #
###############################################################################
def render_depth_from_points(
    points_world,
    intrinsic,
    extrinsic,
    H, W
):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    R_cw = extrinsic[:3, :3]
    t_cw = extrinsic[:3, 3]

    # world -> camera
    pts_cam = (R_cw @ points_world.T).T + t_cw
    z = pts_cam[:, 2]

    valid = z > 0
    pts_cam = pts_cam[valid]
    z = z[valid]

    u = (fx * pts_cam[:, 0] / z + cx).astype(int)
    v = (fy * pts_cam[:, 1] / z + cy).astype(int)

    depth = np.full((H, W), np.inf)

    mask = (
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H)
    )

    u, v, z = u[mask], v[mask], z[mask]

    for ui, vi, zi in zip(u, v, z):
        if zi < depth[vi, ui]:
            depth[vi, ui] = zi

    depth[depth == np.inf] = 0
    return depth



###############################################################################
#                                  可视化工具                                   #
###############################################################################

def make_axis_points(
    center,
    direction,
    length=0.4,
    num_points=50
):
    """
    在 direction 方向生成一条点轴
    """
    t = np.linspace(0, length, num_points)
    pts = center[None, :] + t[:, None] * direction[None, :]
    return pts


def make_normal_points(
    center,
    normal,
    length=0.3,
    num_points=50
):
    """
    在 normal 方向上生成一串点
    """
    t = np.linspace(0, length, num_points)
    pts = center[None, :] + t[:, None] * normal[None, :]
    return pts


def export_pointcloud_with_color(path, points, colors):
    """
    points: (N,3)
    colors: (N,3) uint8 [0,255]
    """
    assert points.shape[0] == colors.shape[0]

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            f.write(
                f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n"
            )

def export_plane_with_normal(
    path,
    plane_points,
    center,
    normal,
    normal_length=0.3
):
    # 桌面点（白色）
    pts_plane = plane_points
    col_plane = np.full((pts_plane.shape[0], 3), 255, dtype=np.uint8)

    # 法线点（红色）
    pts_normal = make_normal_points(center, normal, normal_length)
    col_normal = np.zeros((pts_normal.shape[0], 3), dtype=np.uint8)
    col_normal[:, 0] = 255

    # 中心点（绿色）
    pts_center = center[None, :]
    col_center = np.array([[0, 255, 0]], dtype=np.uint8)

    # 合并
    pts_all = np.vstack([pts_plane, pts_normal, pts_center])
    col_all = np.vstack([col_plane, col_normal, col_center])

    export_pointcloud_with_color(path, pts_all, col_all)

def project_points_to_image(points_3d, intrinsic, image_size=None):
    """
    针对已转换为像素空间的内参进行投影
    points_3d: (N, 3) 相机坐标系
    intrinsic: (3,3) 像素级内参 (fx, fy, cx, cy 均以像素为单位)
    image_size: 已不再严格需要，除非用于边界裁剪
    """
    # 此时 intrinsic[0,0]=fx, [1,1]=fy, [0,2]=cx, [1,2]=cy 已经是像素单位
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    # 防止除以 0 或投影相机背后的点
    eps = 1e-6
    Z_safe = np.clip(Z, eps, None)

    # 标准投影公式：u = fx * (X/Z) + cx
    u = fx * (X / Z_safe) + cx
    v = fy * (Y / Z_safe) + cy

    return np.stack([u, v], axis=1)

def save_masked_depth_viz(depth, mask, z, filename="depth_masked_viz.png"):
    # 1. 创建一个全黑的底图 (H, W)
    vis_depth = np.zeros_like(depth, dtype=np.float32)
    
    # 2. 将 z 值填回 mask 区域
    v, u = np.where(mask > 0)
    vis_depth[v, u] = z

    # 3. 归一化到 0-255 (建议使用 80% 区间的边界来拉伸对比度，效果更好)
    z_min = np.percentile(z, 2) # 使用 2% 和 98% 剔除极端噪声，对比度更自然
    z_max = np.percentile(z, 98)
    
    # 线性拉伸
    depth_norm = np.clip(vis_depth, z_min, z_max)
    depth_norm = (depth_norm - z_min) / (z_max - z_min) * 255
    depth_norm = depth_norm.astype(np.uint8)
    
    # 4. 应用掩码（让非桌面区域保持黑色）
    depth_norm[mask == 0] = 0

    # 5. 应用伪彩色 (如 COLORMAP_JET 或 COLORMAP_VIRIDIS)
    # Jet 色卡：近处红色，远处蓝色（或者相反，取决于数值）
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    # 将背景（非 mask 区域）设为黑色
    depth_color[mask == 0] = 0

    # 6. 保存
    cv2.imwrite(filename, depth_color)
    print(f"Masked 深度可视化图已保存至: {filename}")


# 算桌面的“长 × 宽”（重点）
# ✅ 推荐方法：PCA / Oriented Bounding Box
# 方法原理 1 桌子可能不是和相机对齐的 2 PCA 能找到 主方向（长边） 3 在 PCA 坐标系中算 min/max
def length_width_from_2d(points_2d):
    mean = points_2d.mean(axis=0)
    centered = points_2d - mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    axes = Vt[:2]   # 主轴

    proj = centered @ axes.T

    length = proj[:,0].max() - proj[:,0].min()
    width  = proj[:,1].max() - proj[:,1].min()

    return length, width



def draw_table_on_image(image, corners_px):
    """
    image: (H, W, 3) uint8
    corners_px: (4,2) 像素坐标
    """
    img = image.copy()
    corners_px = corners_px.astype(int)

    # 画角点
    for i, (u, v) in enumerate(corners_px):
        cv2.circle(img, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(
            img, str(i),
            (u + 5, v - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 0, 0), 2
        )

    # 画边
    for i in range(4):
        p1 = tuple(corners_px[i])
        p2 = tuple(corners_px[(i + 1) % 4])
        cv2.line(img, p1, p2, (0, 255, 0), 2)

    return img

def export_plane_with_axes(
    path,
    plane_points,
    center,
    dir_x,
    dir_y,
    dir_z,
    axis_length=0.4
):
    """
    dir_x: X 轴（红）
    dir_y: Y 轴（绿）
    dir_z: Z 轴（蓝）
    """

    pts_all = []
    col_all = []

    # ---------- 桌面 ----------
    pts_all.append(plane_points)
    col_all.append(
        np.full((plane_points.shape[0], 3), 255, dtype=np.uint8)
    )

    # ---------- X 轴（红） ----------
    pts_x = make_axis_points(center, dir_x, axis_length)
    col_x = np.zeros((pts_x.shape[0], 3), dtype=np.uint8)
    col_x[:, 0] = 255
    pts_all.append(pts_x)
    col_all.append(col_x)

    # ---------- Y 轴（绿） ----------
    pts_y = make_axis_points(center, dir_y, axis_length)
    col_y = np.zeros((pts_y.shape[0], 3), dtype=np.uint8)
    col_y[:, 1] = 255
    pts_all.append(pts_y)
    col_all.append(col_y)

    # ---------- Z 轴（蓝） ----------
    pts_z = make_axis_points(center, dir_z, axis_length)
    col_z = np.zeros((pts_z.shape[0], 3), dtype=np.uint8)
    col_z[:, 2] = 255
    pts_all.append(pts_z)
    col_all.append(col_z)

    # ---------- 中心点（黄） ----------
    pts_c = center[None, :]
    col_c = np.array([[255, 255, 0]], dtype=np.uint8)
    pts_all.append(pts_c)
    col_all.append(col_c)

    pts_all = np.vstack(pts_all)
    col_all = np.vstack(col_all)

    export_pointcloud_with_color(path, pts_all, col_all)


def make_bi_axis_points(
    center,
    direction,
    length=0.4,
    num_points=50
):
    """
    生成一根轴的正负两个方向
    """
    t = np.linspace(0, length, num_points)

    pts_pos = center[None, :] + t[:, None] * direction[None, :]
    pts_neg = center[None, :] - t[:, None] * direction[None, :]

    return pts_pos, pts_neg


def export_plane_with_axes_bidirectional(
    path,
    plane_points,
    center,
    dir_x,
    dir_y,
    dir_z,
    axis_length=0.4,
    rotation=None,
    translation=None,
):
    pts_all = []
    col_all = []

    # ========== 桌面 ==========
    pts_all.append(plane_points)
    col_all.append(
        np.full((plane_points.shape[0], 3), 255, dtype=np.uint8)
    )

    # ========== X 轴 ==========
    x_pos, x_neg = make_bi_axis_points(center, dir_x, axis_length)
    col_x_pos = np.tile([255, 0, 0], (x_pos.shape[0], 1))      # 红
    col_x_neg = np.tile([120, 0, 0], (x_neg.shape[0], 1))      # 深红
    pts_all += [x_pos, x_neg]
    col_all += [col_x_pos, col_x_neg]

    # ========== Y 轴 ==========
    y_pos, y_neg = make_bi_axis_points(center, dir_y, axis_length)
    col_y_pos = np.tile([0, 255, 0], (y_pos.shape[0], 1))      # 绿
    col_y_neg = np.tile([0, 120, 0], (y_neg.shape[0], 1))      # 深绿
    pts_all += [y_pos, y_neg]
    col_all += [col_y_pos, col_y_neg]

    # ========== Z 轴 ==========
    z_pos, z_neg = make_bi_axis_points(center, dir_z, axis_length)
    col_z_pos = np.tile([0, 0, 255], (z_pos.shape[0], 1))      # 蓝
    col_z_neg = np.tile([0, 0, 120], (z_neg.shape[0], 1))      # 深蓝
    pts_all += [z_pos, z_neg]
    col_all += [col_z_pos, col_z_neg]

    # ========== 中心点 ==========
    pts_all.append(center[None, :])
    col_all.append(np.array([[255, 255, 0]], dtype=np.uint8))  # 黄

    pts_all = np.vstack(pts_all)
    col_all = np.vstack(col_all)

    if rotation is not None and translation is not None:
        pts_all = (rotation @ pts_all.T).T + translation
    

    export_pointcloud_with_color(path, pts_all, col_all)
