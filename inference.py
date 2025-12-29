
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
    image_folder = "examples/resized-desk"
    images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # images = ['./test.jpg']
    images = [process_image(img_path) for img_path in images]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape
    
    # Run Inference
    gaussians, pred_context_pose, depth_dict = model.inference((images+1)*0.5)
    print(f"type {type(depth_dict['depth'])}, keys {depth_dict.keys()}")
    print(f"depth map shape {depth_dict['depth'].shape}")
    print(depth_dict['depth'])
    depth_map = depth_dict['depth'][0][0].squeeze().cpu().numpy()
    np.save(Path(image_folder) /'depth.npy', depth_map)
    print(f"type {type(depth_map)}, shape {depth_map.shape}, maximum value {depth_map.max()}")
    # imageio.imwrite('depth.png', depth_map)
    # 保存可视化版本
    depth_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
    imageio.imwrite(Path(image_folder) /'depth_visual.png', depth_normalized)

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
    
if __name__ == "__main__":
    main()