from pathlib import Path
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.misc.image_io import save_interpolated_video
from src.model.ply_export import export_ply
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image
import numpy as np
import cv2
from utils import *

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
    image_folder = "examples/eggplant"
    images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # images = ['./test.jpg']
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
    # depth =  render_depth_from_points(gaussian_xyz, intrinsic, extrinsic, H, W)
    depth = depth_dict['depth'][0][0].squeeze().cpu().numpy()
    np.save(Path(image_folder) /'depth.npy', depth)
    # 保存可视化版本
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    imageio.imwrite(Path(image_folder) /'../depth_visual.png', depth_normalized)
    print(f"anyplat mean_depth_ori: {depth.mean():.4f}, min_depth_ori: {depth.min():.4f}, max_depth_ori: {depth.max():.4f}")
    print(depth)
    print(f"type {type(depth)}, shape {depth.shape}, maximum value {depth.max()}")
    # save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, image_folder, model.decoder)
    export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(image_folder) / "gaussians.ply")



if __name__ == "__main__":
    main()