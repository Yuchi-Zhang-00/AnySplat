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
from helper_function import *

def compute_table_geometry_ransac(depth, mask, intrinsic, extrinsic):
    """
    使用 RANSAC 平面 + inner PCA
    构造 world -> table-aligned 变换
    """

    H, W = depth.shape

    # ===== 1. intrinsic =====
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # ===== 2. depth -> camera points =====
    points_cam = depth_to_points(depth, mask, fx, fy, cx, cy)
    print("points_cam:", points_cam.shape)

    # # ===== 2.5 粗 RANSAC（只为 normal）=====
    # normal_coarse, _, _ = fit_plane_ransac_safe(
    #     points_cam,
    #     num_iters=200,
    #     dist_thresh=0.01
    # )
    # # ===== 2.6 3D inner rectangle 过滤 =====
    # points_inner = filter_points_by_inner_rect_3d(
    #     points_cam,
    #     normal_coarse,
    #     keep_ratio=0.6   # 保留中间 60%
    # )
    # ===== 3. RANSAC plane =====
    normal_cam, center_cam, inlier_idx = fit_plane_ransac_safe(
        # points_inner,
        points_cam,
        num_iters=600,
        dist_thresh=0.0015  # 桌面通常很平
    )

    print(f' ransan 得到的 normal : {normal_cam}')

    pts_plane = points_cam[inlier_idx]

    # ===== 4. plane coordinate system =====
    u, v = plane_coordinate_system(normal_cam)

    rel = pts_plane - center_cam
    pts_2d = np.stack([rel @ u, rel @ v], axis=1)

    # ===== 5. inner rectangle =====
    x, y = pts_2d[:, 0], pts_2d[:, 1]
    x_min, x_max = np.percentile(x, [20, 80])
    y_min, y_max = np.percentile(y, [20, 80])

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

    # normal_cam = np.cross(dir_long_cam, dir_short_cam)

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



def compute_table_geometry_ransac_obb(depth, mask, intrinsic, extrinsic):
    """
    RANSAC 平面 + ConvexHull + 最小面积 OBB
    构造 world -> table-aligned 变换
    """

    H, W = depth.shape

    # ===== 1. intrinsic =====
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # ===== 2. depth -> camera points =====
    points_cam = depth_to_points(depth, mask, fx, fy, cx, cy)
    if points_cam.shape[0] < 100:
        raise RuntimeError("Too few depth points")

    # ===== 3. RANSAC plane =====
    normal_cam, center_cam, inlier_idx = fit_plane_ransac_safe(
        points_cam,
        num_iters=600,
        dist_thresh=0.002
    )

    pts_plane = points_cam[inlier_idx]

    # ===== 4. plane coordinate system =====
    u, v = plane_coordinate_system(normal_cam)

    # 投影到平面 2D
    rel = pts_plane - center_cam
    pts_2d = np.stack([rel @ u, rel @ v], axis=1)

    # ===== 5. Convex hull =====
    hull = ConvexHull(pts_2d)
    hull_pts = pts_2d[hull.vertices]

    # ===== 6. 最小面积 OBB =====
    rect = cv2.minAreaRect(hull_pts.astype(np.float32))
    (cx2d, cy2d), (w, h), angle_deg = rect

    # OpenCV angle: [-90, 0)
    theta = np.deg2rad(angle_deg)
    dir_long_2d = np.array([np.cos(theta), np.sin(theta)])

    # 保证 dir_long 是长边
    if w < h:
        w, h = h, w
        dir_long_2d = np.array([-dir_long_2d[1], dir_long_2d[0]])

    dir_short_2d = np.array([-dir_long_2d[1], dir_long_2d[0]])

    # ===== 7. 2D → 3D =====
    dir_long_cam = dir_long_2d[0] * u + dir_long_2d[1] * v
    dir_long_cam /= np.linalg.norm(dir_long_cam)

    dir_short_cam = np.cross(normal_cam, dir_long_cam)
    dir_short_cam /= np.linalg.norm(dir_short_cam)

    normal_cam = np.cross(dir_long_cam, dir_short_cam)
    normal_cam /= np.linalg.norm(normal_cam)

    # ===== 8. 防翻转（世界一致性） =====
    R_cw = extrinsic[:3, :3]
    if (R_cw @ dir_long_cam)[0] < 0:
        dir_long_cam  = -dir_long_cam
        dir_short_cam = -dir_short_cam

    # ===== 9. 桌面中心（3D） =====
    center_plane_cam = (
        center_cam
        + cx2d * u
        + cy2d * v
    )

    # ===== 10. OBB 四角（可视化用） =====
    corners_2d = np.array([
        [-w/2, -h/2],
        [ w/2, -h/2],
        [ w/2,  h/2],
        [-w/2,  h/2],
    ])

    corners_3d = (
        center_plane_cam
        + corners_2d[:, 0, None] * dir_long_cam
        + corners_2d[:, 1, None] * dir_short_cam
    )

    # ===== 11. alignment transform =====
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
        "length": float(w),
        "width": float(h),
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
    b, v, _, H, W = images.shape
    
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
    imageio.imwrite(Path(image_folder) /'../depth_visual1.png', depth_normalized)

    # Save the results
    pred_all_extrinsic = pred_context_pose['extrinsic'][0][0].cpu().numpy()
    pred_all_intrinsic = pred_context_pose['intrinsic'][0][0].cpu().numpy()
    pred_all_intrinsic[0, 0] = pred_all_intrinsic[0, 0] * W / 2
    pred_all_intrinsic[1, 1] = pred_all_intrinsic[1, 1] * H / 2
    pred_all_intrinsic[0, 2] = pred_all_intrinsic[0, 2] * W
    pred_all_intrinsic[1, 2] = pred_all_intrinsic[1, 2] * H
    print(f'pred_all_extrinsic,{pred_all_extrinsic}, shape {pred_all_extrinsic.shape}')
    np.save(Path(image_folder) /'extrinsic.npy', pred_all_extrinsic)
    print(f'pred_all_intrinsic, {pred_all_intrinsic}, shape {pred_all_intrinsic.shape}')
    np.save(Path(image_folder) /'intrinsic.npy', pred_all_intrinsic)
    # save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, image_folder, model.decoder)
    export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "gaussians.ply")
    export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "centralized_gaussians.ply",shift_and_scale=True)
    
        # ================= 桌面几何 =================

    # 你已有的数据
    (H, W) = (448, 448)                        
    
    intrinsic = pred_all_intrinsic
    extrinsic = pred_all_extrinsic
    # depth =  render_depth_from_points(gaussians.means[0].detach().cpu().numpy(), intrinsic, extrinsic, H, W)
    # epth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    # imageio.imwrite(Path(image_folder) /'../depth_visual2.png', depth_normalized)
    # imageio.imwrite(Path(image_folder) /'../depth_diff.png', abs(depth - depth_map).astype(np.uint8))
    # print('depth_diff',depth - depth_map)
    # print('depth diff sum', np.sum(abs(depth - depth_map)))
    depth = depth_map  
    # TODO: 换成你的桌面 mask
    mask = cv2.imread(Path(image_folder) / "../table_mask.png", cv2.IMREAD_GRAYSCALE).astype(np.uint8)  # 0/1
    # mask = compute_inner_rect_mask(mask)
    # result = compute_table_geometry_ransac_obb(
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
        str(Path(image_folder) / "../table_corners_debug.png"),
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