import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
from tqdm import tqdm
import random
import argparse
random.seed(1234)
np.random.seed(1234)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# I = Jt + A(1-t)

class FogGenerator:
    def __init__(self, beta_range: Tuple[float, float] = (0.15, 0.35),
                 a_range: Tuple[float, float] = (0.7, 0.9),
                 center_ratio: Tuple[float, float] = (0.5, 0.55),
                 depth_max=35):

        self.beta_range = beta_range
        self.a_range = a_range
        self.center_ratio = center_ratio
        self.depth_max = depth_max

    def _generate_depth_map(self, shape: Tuple[int, int],
                            center_ratio: Tuple[float, float],
                            depth_max: float) -> np.ndarray:
        h, w = shape

        cx, cy = int(w * center_ratio[0]), int(h * center_ratio[1])

        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        corners = np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]])
        max_dist = np.max(np.sqrt((corners[:, 0] - cx) ** 2 + (corners[:, 1] - cy) ** 2))

        norm_dist = distance / (max_dist + 1e-8)
        depth = depth_max * (1 - norm_dist)
        return np.clip(depth, 0, depth_max)

    def apply_fog(self, image: np.ndarray,
                  beta: Optional[float] = None,
                ) -> np.ndarray:
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        beta = np.random.uniform(*self.beta_range)
        a = np.random.uniform(*self.a_range)
        center = self.center_ratio

        depth = self._generate_depth_map((h, w), center, depth_max=self.depth_max)

        transmission = np.exp(-beta * depth)
        transmission = np.clip(transmission, 0.1, 1.0)
        transmission_3ch = transmission[..., np.newaxis]

        fogged = image * transmission_3ch + a * (1 - transmission_3ch)

        return np.clip(fogged, 0, 1)

    def process_folder(self, input_dir: str, output_dir: str):

        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        img_files = []
        for ext in extensions:
            img_files.extend(list(in_path.glob(ext)))

        if not img_files:
            logging.warning(f"在 {input_dir} 中未找到图像文件。")
            return

        logging.info(f"开始处理 {len(img_files)} 张图像...")

        for img_p in tqdm(img_files, desc="Adding Fog"):
            try:
                img = cv2.imread(str(img_p))
                if img is None:
                    continue

                fogged_img = self.apply_fog(img)

                save_path = out_path / img_p.name
                cv2.imwrite(str(save_path), (fogged_img * 255).astype(np.uint8))
            except Exception as e:
                logging.error(f"处理文件 {img_p.name} 时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fog_dataset')
    parser.add_argument('--input', '-i', type=str, default="HQ", help='输入图像文件夹路径')
    parser.add_argument('--output', '-o', type=str, default="LQ", help='输出图像文件夹路径')
    parser.add_argument('--beta-min', type=float, default=0.15, help='雾化浓度参数下限')
    parser.add_argument('--beta-max', type=float, default=0.35, help='雾化浓度参数上限')
    parser.add_argument('--a-min', type=float, default=0.7, help='大气光值下限')
    parser.add_argument('--a-max', type=float, default=0.9, help='大气光值上限')
    parser.add_argument('--center-x', type=float, default=0.5, help='雾化中心X坐标比例')
    parser.add_argument('--center-y', type=float, default=0.55, help='雾化中心Y坐标比例')
    parser.add_argument('--depth-max', type=float, default=35, help='最大深度值')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    args = parser.parse_args()

    generator = FogGenerator(
        beta_range=(args.beta_min, args.beta_max),
        a_range=(args.a_min, args.a_max),
        center_ratio=(args.center_x, args.center_y),
        depth_max=args.depth_max
    )
    generator.process_folder(input_dir=args.input, output_dir=args.output)
