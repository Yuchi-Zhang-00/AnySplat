"""Geometry helpers used by the table-alignment / 3DGS pipeline.

Only the functions still called by `inference*.py` in this directory and by
`pipeline/background_reconstruction.py` are kept:

  - fit_plane_ransac_safe_2  : RANSAC + SVD plane fit (with `nx = 0` snap)
  - depth_to_points          : depth map -> camera-frame point cloud (mask gated)
  - project_to_plane / to_2d / plane_coordinate_system : plane-coord helpers
  - align_points_to_table    : rigid transform of a point cloud
  - shrink_mask_erode        : ratio-based mask erosion
  - render_depth_from_points : project a world-frame cloud to a depth image
  - save_masked_depth_viz    : optional colour-mapped depth debug PNG

A handful of unused visualisation / OBB helpers were dropped during cleanup;
recover them from git history if you need them.
"""

import os

import cv2
import numpy as np


# ===== Plane fitting =====================================================

def fit_plane_ransac_safe_2(points, num_iters=500, dist_thresh=0.005, sample_N=20000):
    """RANSAC plane fit followed by an SVD refine of the normal.

    After fitting, the normal is snapped to the y/z plane (`nx = 0`) — this
    reflects an assumption baked into the surrounding pipeline that the camera
    is roughly upright relative to the table. The returned `center` is the
    AABB centre in the fitted plane's local 2D frame (not the inlier mean).
    """
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

    # --- RANSAC ---
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

    # --- SVD refine of the normal on inliers ---
    pts_in = pts[best_inliers]
    fit_center = pts_in.mean(axis=0)  # only used for SVD centring
    pts_centered = pts_in - fit_center

    _, _, Vt = np.linalg.svd(pts_centered)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal

    # --- Snap normal to the y/z plane (nx = 0). ---
    n2 = np.array([0.0, normal[1], normal[2]])
    norm = np.linalg.norm(n2)
    if norm < 1e-6:
        raise ValueError("normal is nearly parallel to the X axis; cannot enforce nx = 0")
    n2 /= norm

    # --- Geometric centre = AABB centre in the plane's 2D frame ---
    u, v = plane_coordinate_system(n2)
    proj = project_to_plane(pts_in, n2, fit_center)
    pts_2d = to_2d(proj, fit_center, u, v)
    cx = (pts_2d[:, 0].min() + pts_2d[:, 0].max()) / 2
    cy = (pts_2d[:, 1].min() + pts_2d[:, 1].max()) / 2
    geometric_center = fit_center + cx * u + cy * v

    return n2, geometric_center, idx[best_inliers]


# ===== 2D / 3D plane-frame helpers =======================================

def depth_to_points(depth, mask, fx, fy, cx, cy):
    """Back-project the mask-covered pixels of a depth map into camera frame.

    Returns (N, 3) points in camera coordinates. Optionally writes a colour
    depth visualisation if the `ANYSPLAT_DEBUG_VIZ` env var is set.
    """
    v, u = np.where(mask > 0)
    z = depth[v, u]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    print(f"depth min/max/mean: {z.min(), z.max(), z.mean()}")
    lower_bound = np.percentile(z, 10)
    upper_bound = np.percentile(z, 90)
    print(f"80% of values fall in: [{lower_bound:.2f}, {upper_bound:.2f}]")

    if os.environ.get("ANYSPLAT_DEBUG_VIZ"):
        save_masked_depth_viz(depth, mask, z, "table_depth_only.png")

    return np.stack([x, y, z], axis=1)


def project_to_plane(points, normal, point_on_plane):
    """Project 3D points onto the plane defined by `normal` and `point_on_plane`."""
    diff = points - point_on_plane
    dist = diff @ normal
    return points - np.outer(dist, normal)


def plane_coordinate_system(normal):
    """Build two orthonormal in-plane axes (u, v) for a given normal."""
    tmp = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def to_2d(points, origin, u, v):
    """Express 3D points as 2D coordinates in the (u, v) plane frame."""
    rel = points - origin
    x = rel @ u
    y = rel @ v
    return np.stack([x, y], axis=1)


def align_points_to_table(points, R_align, t_align):
    """Apply a rigid transform to an (N, 3) world-frame point cloud."""
    return (R_align @ points.T).T + t_align


# ===== Mask preprocessing ================================================

def shrink_mask_erode(mask, ratio=0.1):
    """Erode a binary mask by a kernel whose size is `ratio` * the mask bbox.

    Accepts uint8 input in either {0, 1} or {0, 255} form.
    """
    mask = (mask > 0).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask")

    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1

    k = int(min(h, w) * ratio)
    k = max(k, 3) | 1  # force odd and >= 3

    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(mask, kernel)


# ===== Depth rendering ===================================================

def render_depth_from_points(points_world, intrinsic, extrinsic, H, W):
    """Render a depth image by projecting a world-frame cloud through the
    given camera. Uses a vectorised z-buffer: points are sorted by z
    descending so closer points overwrite farther ones during the final
    fancy-index assignment.
    """
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
    print(f"size of valid {np.sum(valid)} / {H * W}")
    pts_cam = pts_cam[valid]
    z = z[valid]

    u = (fx * pts_cam[:, 0] / z + cx).astype(int)
    v = (fy * pts_cam[:, 1] / z + cy).astype(int)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inside], v[inside], z[inside]

    # Sort descending: farther points written first, closer ones overwrite.
    order = np.argsort(z)[::-1]
    u, v, z = u[order], v[order], z[order]

    depth = np.zeros((H, W), dtype=np.float32)
    depth[v, u] = z
    return depth


# ===== Debug visualisation ===============================================

def save_masked_depth_viz(depth, mask, z, filename="depth_masked_viz.png"):
    """Save a colour-mapped depth PNG, restricted to the mask region.

    Stretches contrast using the 2%-98% percentile of `z` so a few outliers
    don't squash the dynamic range. Pixels outside the mask are black.
    """
    vis_depth = np.zeros_like(depth, dtype=np.float32)
    v, u = np.where(mask > 0)
    vis_depth[v, u] = z

    z_min = np.percentile(z, 2)
    z_max = np.percentile(z, 98)

    depth_norm = np.clip(vis_depth, z_min, z_max)
    depth_norm = (depth_norm - z_min) / (z_max - z_min) * 255
    depth_norm = depth_norm.astype(np.uint8)
    depth_norm[mask == 0] = 0

    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    depth_color[mask == 0] = 0

    cv2.imwrite(filename, depth_color)
    print(f"Masked depth visualisation saved to: {filename}")
