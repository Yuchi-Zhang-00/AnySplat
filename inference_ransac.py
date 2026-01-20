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
from utils import *

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

    # ===== 3. RANSAC plane =====
    normal_cam, center_cam, inlier_idx = fit_plane_ransac_safe_2(
        # points_inner,
        points_cam,
        num_iters=600,
        dist_thresh=0.005,  # 桌面通常很平
        sample_N=40000
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

    # ===== 8. 世界一致性（防翻转） =====
    R_cw = extrinsic[:3, :3]
    if (R_cw @ dir_long_cam)[0] < 0:
        dir_long_cam = -dir_long_cam
        dir_short_cam = -dir_short_cam

    export_plane_with_axes_bidirectional(
    "table_ransac.ply",
    plane_points=points_cam,
    center=center_cam,
    dir_x=dir_long_cam,
    dir_y=dir_short_cam,
    dir_z=normal_cam,
    axis_length=1
)


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
    export_plane_with_axes_bidirectional(
    "table_transformed_ransac.ply",
    plane_points=points_cam,
    center=center_cam,
    dir_x=dir_long_cam,
    dir_y=dir_short_cam,
    dir_z=normal_cam,
    axis_length=1,
    rotation=R_align_world,
    translation=t_align_world
)   

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
    image_folder = "examples/new-desk"
    images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    images = [process_image(img_path) for img_path in images]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
    b, v, _, H, W = images.shape
    
    # Run Inference
    gaussians, pred_context_pose, depth_dict = model.inference((images+1)*0.5)

    # Save the results
    pred_all_extrinsic = pred_context_pose['extrinsic'][0][0].inverse().cpu().numpy()  # anysplat输出的extrinsic是 camere2world
    pred_all_intrinsic = pred_context_pose['intrinsic'][0][0].cpu().numpy()
    print("raw intrinsic from network:\n", pred_context_pose['intrinsic'][0][0])
    print("converted fx fy cx cy:",
        pred_all_intrinsic[0,0],
        pred_all_intrinsic[1,1],
        pred_all_intrinsic[0,2],
        pred_all_intrinsic[1,2])
    print("expected image center:", W/2, H/2)
    pred_all_intrinsic[0,:] = pred_all_intrinsic[0,:] * W
    pred_all_intrinsic[1,:] = pred_all_intrinsic[1,:] * H
    print(f'pred_all_extrinsic, \n{pred_all_extrinsic}, \n shape  {pred_all_extrinsic.shape}')
    np.save(Path(image_folder) /'extrinsic.npy', pred_all_extrinsic)
    print(f'pred_all_intrinsic, \n{pred_all_intrinsic}, \n shape  {pred_all_intrinsic.shape}')
    np.save(Path(image_folder) /'intrinsic.npy', pred_all_intrinsic)
    intrinsic = pred_all_intrinsic
    extrinsic = pred_all_extrinsic
    gaussian_xyz = gaussians.means[0].detach().cpu().numpy()
    # aynsplat直给的深度图不准确 ,投影到三维后的桌面跟3DGS的桌面不贴合。
    # 所以需要靠3DGS重新渲染得到准确的depth  
    depth =  render_depth_from_points(gaussian_xyz, intrinsic, extrinsic, H, W)
    np.save(Path(image_folder) /'depth.npy', depth)
    # 保存可视化版本
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    imageio.imwrite(Path(image_folder) /'../depth_visual.png', depth_normalized)
    print(f"type {type(depth)}, shape {depth.shape}, maximum value {depth.max()}")
    # save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, image_folder, model.decoder)
    export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "gaussians.ply")
        # ================= 桌面几何 =================                     

    mask = cv2.imread(str(Path(image_folder) / "../table_mask.png"), cv2.IMREAD_GRAYSCALE).astype(np.uint8)  # 0/1
    # mask = compute_inner_rect_mask(mask)
    mask = shrink_mask_erode(mask, ratio=0.12)
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
    
    points_table_world = align_points_to_table(
        gaussian_xyz,   # 桌面点云
        result["R_align_world"],
        result["t_align_world"]
        )
   
    export_ply(points_table_world, gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "aligned_world_gaussians_ransac.ply")

    x = points_table_world[:,0].copy()
    y = points_table_world[:,1].copy()
    points_table_world[:,0] = y
    points_table_world[:,1] = x
    points_table_world[:,2] *= -1
    points_table_world[:,2] += 0.56

    points_table_world[:,0] -= 0.3

    # R = np.array([[1,0,0],
    #               [0,1,0],
    #               [0,0,-1]])
    # t = np.array([-0.5,0,0.56])

    # R = np.array([[0, -1 ,0],
    #               [1, 0, 0],
    #               [0, 0, -1]])
    # t = np.array([0.5,0,-0.56])
    
    # points = align_points_to_table(points_table_world, R, t)
   
    export_ply(points_table_world, gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "bridge_desk.ply")
    
    print(points_table_world[:,2].min(), points_table_world[:,2].max())

if __name__ == "__main__":
    main()