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
import cv2
import argparse
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

#     export_plane_with_axes_bidirectional(
#     "table_ransac.ply",
#     plane_points=points_cam,
#     center=center_cam,
#     dir_x=dir_long_cam,
#     dir_y=dir_short_cam,
#     dir_z=normal_cam,
#     axis_length=1
# )


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

#     corners_3d = (
#         center_plane_cam
#         + np.array([
#             [min_xy[0], min_xy[1]],
#             [max_xy[0], min_xy[1]],
#             [max_xy[0], max_xy[1]],
#             [min_xy[0], max_xy[1]],
#         ])[:, 0, None] * dir_long_cam
#         + np.array([
#             [min_xy[0], min_xy[1]],
#             [max_xy[0], min_xy[1]],
#             [max_xy[0], max_xy[1]],
#             [min_xy[0], max_xy[1]],
#         ])[:, 1, None] * dir_short_cam
#     )

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
#     export_plane_with_axes_bidirectional(
#     "table_transformed_ransac.ply",
#     plane_points=points_cam,
#     center=center_cam,
#     dir_x=dir_long_cam,
#     dir_y=dir_short_cam,
#     dir_z=normal_cam,
#     axis_length=1,
#     rotation=R_align_world,
#     translation=t_align_world
# )   

    return {
        # "corners_3d": corners_3d,
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


import os
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
import imageio


def process_single_image(image_path, model, device):
    """处理单个图片的完整流程"""
    image_folder = os.path.dirname(image_path)
    image_ori_path = os.path.join(image_folder, 'input_image.png') 
    # Load Image
    image = process_image(image_path)
    image_ori = process_image(image_ori_path)
    images_ori = torch.stack([image_ori], dim=0).unsqueeze(0).to(device)
    images = torch.stack([image], dim=0).unsqueeze(0).to(device)  # [1, 1, 3, 448, 448]
    b, v, _, H, W = images.shape
    
    # Run Inference
    with torch.no_grad():
        gaussians, pred_context_pose, depth_dict = model.inference((images+1)*0.5)
        gaussians_ori, pred_context_pose_ori, depth_dict_ori = model.inference((images_ori+1)*0.5)
    depth_ori = depth_dict_ori['depth'][0][0].squeeze().cpu().numpy()
    
    # 保存深度图
    depth_path = Path(image_folder) / 'depth_ori.npy'
    np.save(depth_path, depth_ori)
    # 保存可视化深度图
    depth_ori_normalized = ((depth_ori - depth_ori.min()) / (depth_ori.max() - depth_ori.min()) * 255).astype(np.uint8)
    depth_ori_visual_path = Path(image_folder) / 'depth_ori_visual.png'
    imageio.imwrite(depth_ori_visual_path, depth_ori_normalized)
    # Save the results
    pred_all_extrinsic = pred_context_pose['extrinsic'][0][0].inverse().cpu().numpy()  # anysplat输出的extrinsic是 camere2world
    pred_all_intrinsic = pred_context_pose['intrinsic'][0][0].cpu().numpy()
    
    print(f"处理 {os.path.basename(image_folder)}: 转换后的fx fy cx cy:")
    print(f"  fx: {pred_all_intrinsic[0,0] * W:.2f}, fy: {pred_all_intrinsic[1,1] * H:.2f}")
    print(f"  cx: {pred_all_intrinsic[0,2] * W:.2f}, cy: {pred_all_intrinsic[1,2] * H:.2f}")
    
    # 缩放内参矩阵到实际像素坐标
    pred_all_intrinsic[0,:] = pred_all_intrinsic[0,:] * W
    pred_all_intrinsic[1,:] = pred_all_intrinsic[1,:] * H
    
    # 保存相机参数
    extrinsic_path = Path(image_folder) / 'extrinsic.npy'
    intrinsic_path = Path(image_folder) / 'intrinsic.npy'
    np.save(extrinsic_path, pred_all_extrinsic)
    np.save(intrinsic_path, pred_all_intrinsic)
    
    intrinsic = pred_all_intrinsic
    extrinsic = pred_all_extrinsic
    gaussian_xyz = gaussians.means[0].detach().cpu().numpy()
    depth = depth_dict['depth'][0][0].squeeze().cpu().numpy()
    
    # 保存深度图
    depth_path = Path(image_folder) / 'depth.npy'
    np.save(depth_path, depth)
    
    # 保存可视化深度图
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_visual_path = Path(image_folder) / 'depth_visual.png'
    imageio.imwrite(depth_visual_path, depth_normalized)
    
    # 创建3D资产文件夹
    assets_folder = os.path.join(image_folder, "3d_assets")
    os.makedirs(assets_folder, exist_ok=True)
    
    # 导出原始的3D高斯模型
    export_ply(
        gaussians.means[0], 
        gaussians.scales[0], 
        gaussians.rotations[0], 
        gaussians.harmonics[0], 
        gaussians.opacities[0], 
        Path(assets_folder) / "bg_bridge.ply"
    )
    
    # 从3DGS重新渲染深度图
    depth_point = render_depth_from_points(gaussian_xyz, intrinsic, extrinsic, H, W)
    
    # 加载背景掩码
    mask_path = Path(image_folder) / "bg_mask.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        mask = shrink_mask_erode(mask, ratio=0.12)
        
        # 计算桌面几何
        result = compute_table_geometry_ransac(
            depth=depth_point,
            mask=mask,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )
        
        print(f"\n{os.path.basename(image_folder)} 桌面几何结果:")
        print(f"  长度 (m): {result['length']:.3f}")
        print(f"  宽度 (m): {result['width']:.3f}")
        print(f"  法向: {result['normal']}")
        
        # 将点云对齐到桌面坐标系
        points_table_world = align_points_to_table(
            gaussian_xyz,
            result["R_align_world"],
            result["t_align_world"]
        )
        
        # 居中处理
        points_table_world = points_table_world - np.median(points_table_world, axis=0)
        
        # 使用分位数来确定范围，避免异常值影响
        abs_points = np.abs(points_table_world)
        ref_range = np.quantile(abs_points, 0.95)
        
        # 计算缩放因子，使95%的数据在[-0.6, 0.6]内
        scale_factor = ref_range / 0.6
        
        # 应用缩放
        points_table_world = points_table_world / scale_factor
        gaussians.scales[0] = gaussians.scales[0] / scale_factor
        
        # 保存缩放因子
        scale_path = Path(image_folder) / 'scale.npy'
        np.save(scale_path, scale_factor)
        print(f"  缩放因子: {scale_factor:.3f}")
        
        # 坐标系转换
        x = points_table_world[:,0].copy()
        y = points_table_world[:,1].copy()
        points_table_world[:,0] = y
        points_table_world[:,1] = x
        points_table_world[:,2] *= -1
        points_table_world[:,2] += 0.56
        points_table_world[:,0] -= 0.4
        
        # 导出对齐后的3D高斯模型
        export_ply(
            points_table_world, 
            gaussians.scales[0], 
            gaussians.rotations[0], 
            gaussians.harmonics[0], 
            gaussians.opacities[0], 
            Path(assets_folder) / "bg_mujoco.ply"
        )
        
        print(f"  最小Z值: {points_table_world[:,2].min():.3f}, 最大Z值: {points_table_world[:,2].max():.3f}")
    else:
        print(f"警告: 未找到bg_mask.png文件，跳过桌面几何计算: {mask_path}")
    
    print(f"处理完成，结果保存在: {image_folder}")


def main():
    parser = argparse.ArgumentParser(description="将一张图像生成为3D高斯模型并输出对应的相机内外参和深度图。")
    parser.add_argument("input_dir", type=str, help="输入的图片所在目录。")
    # 判断输入的是文件还是目录，如果是文件则取其所在目录
    args = parser.parse_args()
    if os.path.isfile(args.input_dir):
        input_dir = os.path.dirname(args.input_dir)
    else:
        input_dir = args.input_dir
    
    
    # Load the model from Hugging Face (只加载一次)
    print("正在加载模型...")
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("模型加载完成")
    
    # 查找所有clean_background.png文件
    clean_background_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower() == 'clean_background.png' or file.lower() == 'clean_background.jpg':
                clean_background_files.append(os.path.join(root, file))
    
    print(f"找到 {len(clean_background_files)} 个clean_background.png文件")
    
    # 对每个clean_background.png文件进行处理
    for idx, image_path in enumerate(clean_background_files, 1):
        print(f"\n处理第 {idx}/{len(clean_background_files)} 个文件: {image_path}")
        
        try:
            process_single_image(image_path, model, device)
            print(f"成功处理: {image_path}")
        except Exception as e:
            print(f"处理 {image_path} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()