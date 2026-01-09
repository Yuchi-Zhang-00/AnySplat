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


def project_points_to_image(points_3d, intrinsic, image_size):
    """
    points_3d: (N, 3) 相机坐标系
    intrinsic: (3,3) NeRF/NDC 内参
    image_size: (H, W)
    """
    H, W = image_size

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    # 防止除 0
    eps = 1e-6
    Z = np.clip(Z, eps, None)

    x_ndc = fx * (X / Z) + cx
    y_ndc = fy * (Y / Z) + cy

    u = x_ndc * W
    v = y_ndc * H

    return np.stack([u, v], axis=1)
# mask内的像素投影到点云
def depth_to_points(depth, mask, fx, fy, cx, cy):
    v, u = np.where(mask > 0)
    z = depth[v, u]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=1)
def fit_plane_ransac_safe(points, num_iters=300, dist_thresh=0.005, sample_N=8000):
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

#  完整用例
# points = depth_to_points(depth, mask, K)

# normal, center = fit_plane(points)

# proj_3d = project_to_plane(points, normal, center)

# u, v = plane_coordinate_system(normal)

# points_2d = to_2d(proj_3d, center, u, v)

# length, width = length_width_from_2d(points_2d)

# print(f"桌面尺寸：{length:.3f} m × {width:.3f} m")


def compute_table_geometry_ransac(depth, mask, intrinsic, extrinsic):
    """
    使用 RANSAC 平面 + inner PCA
    构造 world -> table-aligned 变换
    """

    H, W = depth.shape

    # ===== 1. intrinsic =====
    fx = intrinsic[0, 0] * W / 2
    fy = intrinsic[1, 1] * H / 2
    cx = intrinsic[0, 2] * W
    cy = intrinsic[1, 2] * H

    # ===== 2. depth -> camera points =====
    points_cam = depth_to_points(depth, mask, fx, fy, cx, cy)
    print("points_cam:", points_cam.shape)

    # ===== 3. RANSAC plane =====
    normal_cam, center_cam, inlier_idx = fit_plane_ransac_safe(
        points_cam,
        num_iters=300,
        dist_thresh=0.015  # 桌面通常很平
    )

    pts_plane = points_cam[inlier_idx]

    # ===== 4. plane coordinate system =====
    u, v = plane_coordinate_system(normal_cam)

    rel = pts_plane - center_cam
    pts_2d = np.stack([rel @ u, rel @ v], axis=1)

    # ===== 5. inner rectangle =====
    x, y = pts_2d[:, 0], pts_2d[:, 1]
    x_min, x_max = np.percentile(x, [15, 85])
    y_min, y_max = np.percentile(y, [15, 85])

    inner = (
        (x > x_min) & (x < x_max) &
        (y > y_min) & (y < y_max)
    )
    pts_inner = pts_2d[inner]

    if pts_inner.shape[0] < 50:
        raise RuntimeError("Too few inner RANSAC points")

    # ===== 6. PCA on inner =====
    mean_2d = pts_inner.mean(axis=0)
    centered = pts_inner - mean_2d
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    dir_long_2d = Vt[0]

    # ===== 7. 2D -> 3D =====
    dir_long_cam = dir_long_2d[0] * u + dir_long_2d[1] * v
    dir_long_cam /= np.linalg.norm(dir_long_cam)

    dir_short_cam = np.cross(normal_cam, dir_long_cam)
    dir_short_cam /= np.linalg.norm(dir_short_cam)

    normal_cam = np.cross(dir_long_cam, dir_short_cam)

    # ===== 8. 世界一致性（防翻转） =====
    R_cw = extrinsic[:3, :3]
    if (R_cw @ dir_long_cam)[0] < 0:
        dir_long_cam = -dir_long_cam
        dir_short_cam = -dir_short_cam

    # ===== 9. OBB 尺寸 =====
    proj = centered @ Vt[:2].T
    min_xy, max_xy = proj.min(0), proj.max(0)

    length = max_xy[0] - min_xy[0]
    width  = max_xy[1] - min_xy[1]

    center_plane_cam = (
        center_cam
        + mean_2d[0] * u
        + mean_2d[1] * v
    )

    corners_3d = (
        center_plane_cam
        + np.array([
            [min_xy[0], min_xy[1]],
            [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]],
            [min_xy[0], max_xy[1]],
        ])[:, 0, None] * dir_long_cam
        + np.array([
            [min_xy[0], min_xy[1]],
            [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]],
            [min_xy[0], max_xy[1]],
        ])[:, 1, None] * dir_short_cam
    )

    # ===== 10. alignment =====
    R_table_cam = np.stack(
        [dir_long_cam, dir_short_cam, normal_cam],
        axis=1
    )

    R_align_cam = R_table_cam.T
    t_align_cam = -R_align_cam @ center_plane_cam

    R_align_world = R_align_cam @ R_cw
    t_align_world = R_align_cam @ extrinsic[:3, 3] + t_align_cam

    print("RANSAC inlier ratio:", len(inlier_idx) / points_cam.shape[0])



    return {
        "corners_3d": corners_3d,
        "length": float(length),
        "width": float(width),
        "normal": normal_cam,
        "dir_long": dir_long_cam,
        "dir_short": dir_short_cam,
        "R_align_cam": R_align_cam,
        "t_align_cam": t_align_cam,
        "R_align_world": R_align_world,
        "t_align_world": t_align_world,
    }


def main():
    # Load the model from Hugging Face
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Load Images
    # image_folder = "examples/test"
    image_folder = "examples/new-desk"
    images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # images = ['./test.jpg']
    images = [process_image(img_path) for img_path in images]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape
    
    # Run Inference
    gaussians, pred_context_pose, depth_dict = model.inference((images+1)*0.5)
    print(f"type {type(depth_dict['depth'])}, keys {depth_dict.keys()}")
    print(f"depth map shape {depth_dict['depth'].shape}")
    # print(depth_dict['depth'])
    depth_map = depth_dict['depth'][0][0].squeeze().cpu().numpy()
    np.save(Path(image_folder) /'depth.npy', depth_map)
    print(f"type {type(depth_map)}, shape {depth_map.shape}, maximum value {depth_map.max()}")
    # imageio.imwrite('depth.png', depth_map)
    # 保存可视化版本
    depth_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
    # imageio.imwrite(Path(image_folder) /'depth_visual.png', depth_normalized)

    # Save the results
    pred_all_extrinsic = pred_context_pose['extrinsic']
    pred_all_intrinsic = pred_context_pose['intrinsic']
    print(f'pred_all_extrinsic,{pred_all_extrinsic}, shape {pred_all_extrinsic.shape}')
    np.save(Path(image_folder) /'extrinsic.npy', pred_all_extrinsic[0][0].cpu().numpy())
    print(f'pred_all_intrinsic, {pred_all_intrinsic}, shape {pred_all_intrinsic.shape}')
    np.save(Path(image_folder) /'intrinsic.npy', pred_all_intrinsic[0][0].cpu().numpy())
    # save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, image_folder, model.decoder)
    export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "gaussians.ply")
    export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "centralized_gaussians.ply",shift_and_scale=True)
    
        # ================= 桌面几何 =================

    # 你已有的数据
    depth = depth_map                       # (448, 448)
    intrinsic = pred_all_intrinsic[0][0].cpu().numpy()
    extrinsic = pred_all_extrinsic[0][0].cpu().numpy()

    # TODO: 换成你的桌面 mask
    mask = cv2.imread(Path(image_folder) / "../table_mask.png", cv2.IMREAD_GRAYSCALE).astype(np.uint8)  # 0/1
    # mask = compute_inner_rect_mask(mask)
    result = compute_table_geometry_ransac(
        depth=depth,
        mask=mask,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    print("\n====== 桌面几何结果 ======")
    print("长度 (m):", result["length"])
    print("宽度 (m):", result["width"])
    print("法向:", result["normal"])
    print("长边方向:", result["dir_long"])
    print("宽边方向:", result["dir_short"])
    print("四个角点:\n", result["corners_3d"])



        # ========== 投影桌面四角并画回图像 ==========

    # 读取原图（注意 process_image 做过 resize，这里要同样尺寸）
    # img_path = sorted([
    #     os.path.join(image_folder, f)
    #     for f in os.listdir(image_folder)
    #     if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    # ])[0]

    # image = cv2.imread(img_path)
    # image = cv2.resize(image, (448, 448))  # 和 depth 一致
    # image = images[0].cpu().numpy()
    image = images[0][0].permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)


    corners_3d = result["corners_3d"]

    corners_px = project_points_to_image(
        corners_3d,
        intrinsic=intrinsic,
        image_size=(448, 448)
    )

    vis = draw_table_on_image(image, corners_px)

    cv2.imwrite(
        str(Path(image_folder) / "table_corners_debug.png"),
        vis
    )

    print("已保存桌面四角可视化：table_corners_debug.png")

    points_table = align_points_to_table(
        gaussians.means[0].cpu().numpy(),   # 桌面点云
        result["R_align_cam"],
        result["t_align_cam"]
        )

    print("桌面点云 z 范围：", points_table[:,2].min(), points_table[:,2].max())
    export_ply(points_table, gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "aligned_cam_gaussians_ransac.ply",R_align=result["R_align_cam"])
    
    points_table = align_points_to_table(
        gaussians.means[0].cpu().numpy(),   # 桌面点云
        result["R_align_world"],
        result["t_align_world"]
        )
    export_ply(points_table, gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "aligned_world_gaussians_ransac.ply",R_align=result["R_align_world"])
    
    print(points_table[:,2].min(), points_table[:,2].max())

if __name__ == "__main__":
    main()