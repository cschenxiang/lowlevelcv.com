import os
import cv2
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm
import random
import argparse
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

random.seed(1234)
np.random.seed(1234)



# I = Jt + A(1-t)

class FogGenerator:
    def __init__(self,
                 beta_range: Tuple[float, float] = (0.15, 0.35),
                 a_range: Tuple[float, float] = (0.7, 0.9),
                 depth_model_path: str = "Weights/depth_anything_vitl14.pth",
                 device: str = 'cuda'):

        self.beta_range = beta_range
        self.a_range = a_range
        self.device = device

        model_config = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        self.depth_model = DepthAnything(model_config).to(self.device)
        self.depth_model.load_state_dict(torch.load(depth_model_path, map_location=self.device))
        self.depth_model.eval()

        self.transform = Compose([
            Resize(518, 518, False, True, 14, 'lower_bound', cv2.INTER_CUBIC),
            NormalizeImage([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def _get_depth_map(self, image: np.ndarray) -> np.ndarray:

        h_orig, w_orig = image.shape[:2]

        img_rgb = image.astype(np.float32) / 255.0

        sample = self.transform({'image': img_rgb})['image']
        tensor = torch.from_numpy(sample).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.depth_model(tensor)

        depth = F.interpolate(depth[None], (h_orig, w_orig), mode='bilinear', align_corners=False)[0, 0]

        depth_min, depth_max = depth.min(), depth.max()

        depth_normalized = (depth - depth_min) / (depth_max - depth_min)

        return depth_normalized.cpu().numpy().astype(np.float32)

    def apply_fog(self, image: np.ndarray,
                  beta: Optional[float] = None) -> np.ndarray:

        # I = Jt + A(1-t)

        beta = np.random.uniform(*self.beta_range) if beta is None else beta
        a = np.random.uniform(*self.a_range)

        depth = self._get_depth_map(image)

        transmission = np.exp(-beta * (1-depth))
        transmission = np.clip(transmission, 0.1, 1.0)
        transmission_3ch = transmission[..., np.newaxis]

        image = image.astype(np.float32) / 255.0

        fogged = image * transmission_3ch + a * (1 - transmission_3ch)

        return (np.clip(fogged, 0, 1) * 255).astype(np.uint8)

    def process_folder(self, input_dir: str, output_dir: str):

        img_files = os.listdir(input_dir)

        os.makedirs(output_dir, exist_ok=True)

        for img_p in tqdm(img_files, desc="Adding Fog"):

            img_bgr = cv2.imread(os.path.join(input_dir, img_p))

            fogged_img = self.apply_fog(img_bgr)

            cv2.imwrite(os.path.join(output_dir, img_p), fogged_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于真实深度图的雾化数据集生成')
    parser.add_argument('--input', '-i', type=str, default="HQ", help='输入图像文件夹路径')
    parser.add_argument('--output', '-o', type=str, default="LQ", help='输出图像文件夹路径')
    parser.add_argument('--beta-min', type=float, default=1, help='雾化浓度参数下限')
    parser.add_argument('--beta-max', type=float, default=3, help='雾化浓度参数上限')
    parser.add_argument('--a-min', type=float, default=0.6, help='大气光值下限')
    parser.add_argument('--a-max', type=float, default=1, help='大气光值上限')
    parser.add_argument('--depth-model', type=str, default="Weights/depth_anything_vitl14.pth", help='深度模型权重路径')
    parser.add_argument('--device', type=str, default="cuda", help='运行设备')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    generator = FogGenerator(
        beta_range=(args.beta_min, args.beta_max),
        a_range=(args.a_min, args.a_max),
        depth_model_path=args.depth_model,
        device=args.device,
    )

    generator.process_folder(input_dir=args.input, output_dir=args.output)