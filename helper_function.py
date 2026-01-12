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
# mask内的像素投影到点云
def depth_to_points(depth, mask, fx, fy, cx, cy):
    v, u = np.where(mask > 0)
    z = depth[v, u]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=1)

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

    # refine normal
    pts_in = pts[best_inliers]
    pts_centered = pts_in - pts_in.mean(0)
    _, _, Vt = np.linalg.svd(pts_centered)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal

    return normal, best_center, idx[best_inliers]

#  拟合桌面平面（拿法向量）  用 SVD / 最小二乘（比 RANSAC 简洁，mask 已知）
def fit_plane(points):
    if points.shape[0] < 50:
        raise ValueError("Too few points for plane fitting")
    
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    return normal, centroid

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


def align_points_to_table(points, R_align, t_align):
    """
    points: (N,3) 世界坐标
    """
    return (R_align @ points.T).T + t_align

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
