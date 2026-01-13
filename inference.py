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


# def compute_table_geometry(depth, mask, intrinsic, extrinsic=None):
#     """
#     使用 cv2.minAreaRect (OBB) 优化后的桌面几何计算
#     """
#     H, W = depth.shape

#     # ========== 1. intrinsic（NDC → pixel 适配 AnySplat） ==========
#     # 注意：AnySplat 的 cx/cy 通常是 0.5 (图像中心)
#     fx = intrinsic[0, 0]
#     fy = intrinsic[1, 1]
#     cx = intrinsic[0, 2]
#     cy = intrinsic[1, 2]


#     # ========== 2. depth → 点云 ==========
#     points = depth_to_points(depth, mask, fx, fy, cx, cy)

#     # ========== 3. 拟合桌面平面 ==========
#     normal, center = fit_plane(points)

#     # 保证法向朝向相机 (Z轴负方向) 或根据需要设定
#     # 这里的逻辑是确保法向量是桌面的“向上”方向
#     if normal[2] > 0:
#         normal = -normal

#     # ========== 4. 建立平面局部坐标系 (u, v) ==========
#     u, v = plane_coordinate_system(normal)

#     # ========== 5. 投影到 2D 平面 ==========
#     rel = points - center
#     pts_2d = np.stack([rel @ u, rel @ v], axis=1).astype(np.float32)

#     # ========== 6. 使用 cv2.minAreaRect 获取稳健的外接矩形 ==========
#     # rect 返回: ((center_x, center_y), (width, height), angle)
#     rect = cv2.minAreaRect(pts_2d)
#     box_2d = cv2.boxPoints(rect) # 得到 4 个角点的 2D 坐标
    
#     (rect_cx, rect_cy), (w_val, h_val), angle = rect
    
#     # 确定长边和短边及其方向
#     # 这里的 w_val, h_val 是矩形的两个边长，不一定哪个是长
#     edge1 = box_2d[1] - box_2d[0]
#     edge2 = box_2d[2] - box_2d[1]
    
#     norm1 = np.linalg.norm(edge1)
#     norm2 = np.linalg.norm(edge2)
    
#     if norm1 > norm2:
#         length, width = norm1, norm2
#         dir_long_2d = edge1 / norm1
#         dir_short_2d = edge2 / norm2
#     else:
#         length, width = norm2, norm1
#         dir_long_2d = edge2 / norm2
#         dir_short_2d = edge1 / norm1

#     # ========== 7. 将 2D 方向转回 3D ==========
#     dir_long_3d  = dir_long_2d[0] * u + dir_long_2d[1] * v
#     dir_short_3d = dir_short_2d[0] * u + dir_short_2d[1] * v
    
#     dir_long_3d  /= np.linalg.norm(dir_long_3d)
#     dir_short_3d /= np.linalg.norm(dir_short_3d)

#     # 重新构建严格正交的右手系: X=长边, Y=宽边, Z=法向
#     axis_z = -normal # 习惯上让桌面向上为 +Z
#     axis_x = dir_long_3d
#     axis_y = np.cross(axis_z, axis_x)
#     axis_y /= np.linalg.norm(axis_y)
    
#     # 再次修正 axis_x 确保完全垂直
#     axis_x = np.cross(axis_y, axis_z)

#     # ========== 8. 计算 3D 角点 ==========
#     # 这里的 box_2d 是相对于 pts_2d 坐标系的，需要转回相机/世界 3D 空间
#     corners_3d = []
#     for pt in box_2d:
#         # pt[0] 是在 u 上的分量，pt[1] 是在 v 上的分量
#         c3d = center + pt[0] * u + pt[1] * v
#         corners_3d.append(c3d)
#     corners_3d = np.array(corners_3d)

#     # ========== 9. 构造对齐变换（World -> Aligned Table） ==========
#     # R_align 的行向量是新的基向量，这样变换后点云会沿轴对齐
#     R_align = np.stack([axis_x, axis_y, axis_z], axis=0) 
#     t_align = -R_align @ center
#      # camera -> world
#     R_cw = extrinsic[:3, :3]
#     t_cw = extrinsic[:3, 3]

#     R_wc = R_cw.T
#     t_wc = -R_wc @ t_cw

#     R_world_align = R_align @ R_cw
#     t_world_align = R_align @ t_cw + t_align


#     return {
#         "corners_3d": corners_3d,
#         "length": length,
#         "width": width,
#         "dir_long": dir_long_3d,
#         "dir_short": dir_short_3d,
#         "normal": normal,
#         "R_align_cam": R_align,
#         "t_align_cam": t_align,
#         "R_align_world": R_world_align,
#         "t_align_world": t_world_align,
#     }


def compute_table_geometry(depth, mask, intrinsic, extrinsic):
    """
    从 depth + mask + 内外参 计算桌面几何，并构造
    world -> table-aligned (X=长, Y=宽, Z=法向) 的刚体变换
    """

    import numpy as np

    H, W = depth.shape

    # ========== 1. intrinsic（NDC → pixel） ==========
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]


    # ========== 2. depth → 点云（camera 坐标） ==========
    points_cam = depth_to_points(depth, mask, fx, fy, cx, cy)

    # ========== 3. 拟合桌面平面 ==========
    normal_cam, center_cam = fit_plane(points_cam)

    # 法向统一指向 +Z（相机系）
    if normal_cam[2] < 0:
        normal_cam = -normal_cam

    # ========== 4. 平面局部坐标系 ==========
    u, v = plane_coordinate_system(normal_cam)

    rel = points_cam - center_cam
    pts_2d = np.stack([rel @ u, rel @ v], axis=1)

    # ========== 5. inner rectangle（鲁棒去边） ==========
    x, y = pts_2d[:, 0], pts_2d[:, 1]
    x_min, x_max = np.percentile(x, [10, 90])
    y_min, y_max = np.percentile(y, [10, 90])

    inner_mask = (
        (x > x_min) & (x < x_max) &
        (y > y_min) & (y < y_max)
    )

    pts_2d_inner = pts_2d[inner_mask]

    # ========== 6. PCA（只在 inner 区域） ==========
    mean_2d = pts_2d_inner.mean(axis=0)
    centered = pts_2d_inner - mean_2d

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    dir_long_2d  = Vt[0]
    dir_short_2d = Vt[1]

    # ========== 7. 2D → 3D ==========
    dir_long_cam = dir_long_2d[0] * u + dir_long_2d[1] * v
    dir_long_cam /= np.linalg.norm(dir_long_cam)

    dir_short_cam = dir_short_2d[0] * u + dir_short_2d[1] * v
    dir_short_cam /= np.linalg.norm(dir_short_cam)

    # 正交 & 右手系修正
    dir_short_cam = np.cross(normal_cam, dir_long_cam)
    dir_short_cam /= np.linalg.norm(dir_short_cam)
    normal_cam = np.cross(dir_long_cam, dir_short_cam)
    normal_cam /= np.linalg.norm(normal_cam)

    # ========== 8. 世界坐标一致性（可选但强烈建议） ==========
    R_cw = extrinsic[:3, :3]
    t_cw = extrinsic[:3, 3]

    dir_long_world = R_cw @ dir_long_cam
    if dir_long_world[0] < 0:
        dir_long_cam  = -dir_long_cam
        dir_short_cam = -dir_short_cam

    
    export_plane_with_axes_bidirectional(
    "table_svd.ply",
    plane_points=points_cam,
    center=center_cam,
    dir_x=dir_long_cam,
    dir_y=dir_short_cam,
    dir_z=normal_cam,
    axis_length=1
    )

    # ========== 9. OBB（在 PCA 坐标系） ==========
    proj_pca = centered @ Vt[:2].T
    min_xy = proj_pca.min(axis=0)
    max_xy = proj_pca.max(axis=0)

    length = max_xy[0] - min_xy[0]
    width  = max_xy[1] - min_xy[1]

    corners_2d = np.array([
        [min_xy[0], min_xy[1]],
        [max_xy[0], min_xy[1]],
        [max_xy[0], max_xy[1]],
        [min_xy[0], max_xy[1]],
    ])

    # ⭐ 关键：OBB reference 必须是 inner PCA 中心
    center_plane_cam = (
        center_cam
        + mean_2d[0] * u
        + mean_2d[1] * v
    )

    corners_3d_cam = (
        center_plane_cam
        + corners_2d[:, 0, None] * dir_long_cam
        + corners_2d[:, 1, None] * dir_short_cam
    )

    # ========== 10. 构造对齐变换（camera → table） ==========
    R_table_cam = np.stack(
        [dir_long_cam, dir_short_cam, normal_cam],
        axis=1
    )

    R_align_cam = R_table_cam.T
    t_align_cam = -R_align_cam @ center_plane_cam

    # ========== 11. world → table ==========
    R_align_world = R_align_cam @ R_cw
    t_align_world = R_align_cam @ t_cw + t_align_cam

    export_plane_with_axes_bidirectional(
    "table_transformed_svd.ply",
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
        "corners_3d": corners_3d_cam,
        "length": float(length),
        "width": float(width),
        "dir_long": dir_long_cam,
        "dir_short": dir_short_cam,
        "normal": normal_cam,
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
    # imageio.imwrite(Path(image_folder) /'depth_visual.png', depth_normalized)

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
    # export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "centralized_gaussians.ply",shift_and_scale=True)
    
        # ================= 桌面几何 =================

    # 你已有的数据
    gaussian_xyz = gaussians.means[0].detach().cpu().numpy()
    intrinsic = pred_all_intrinsic
    extrinsic = pred_all_extrinsic
    # depth = depth_map   aynsplat直给的深度图不准确  需要靠3DGS重新渲染depth   
    depth =  render_depth_from_points(gaussian_xyz, intrinsic, extrinsic, H, W)

    # TODO: 换成你的桌面 mask
    mask = cv2.imread(Path(image_folder) / "../table_mask.png", cv2.IMREAD_GRAYSCALE).astype(np.uint8)  # 0/1
    # mask = compute_inner_rect_mask(mask)
    result = compute_table_geometry(
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
    image = images[0].cpu().numpy()

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
        gaussian_xyz,   # 桌面点云
        result["R_align_world"],
        result["t_align_world"]
        )
    export_ply(points_table, gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "aligned_world_gaussians.ply")
    




if __name__ == "__main__":
    main()