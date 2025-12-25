import os
import cv2
import numpy as np


def bicubic_super_resolution(lr_dir, sr_dir, scale=4):
    """
    无需权重！基于bicubic插值的图像超分辨率
    :param lr_dir: 低分辨率（LR）图像文件夹路径（输入）
    :param sr_dir: 超分结果（SR）保存路径（输出）
    :param scale: 超分倍数（默认4倍）
    """
    # 创建输出文件夹
    if not os.path.exists(sr_dir):
        os.makedirs(sr_dir)

    # 遍历LR图像
    lr_img_names = [f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in lr_img_names:
        # 读取LR图像
        lr_path = os.path.join(lr_dir, img_name)
        lr_img = cv2.imread(lr_path)
        if lr_img is None:
            print(f"跳过无法读取的图像：{img_name}")
            continue

        # 获取LR尺寸，计算SR尺寸
        lr_h, lr_w = lr_img.shape[:2]
        sr_h, sr_w = lr_h * scale, lr_w * scale

        # Bicubic插值放大（核心步骤，无需权重）
        sr_img = cv2.resize(
            lr_img, (sr_w, sr_h),
            interpolation=cv2.INTER_CUBIC  # bicubic插值核心
        )

        # 保存结果
        sr_path = os.path.join(sr_dir, f"Bicubic_SR_{img_name}")
        cv2.imwrite(sr_path, sr_img)

        print(f"处理完成：{img_name} | LR尺寸：{lr_w}x{lr_h} -> SR尺寸：{sr_w}x{sr_h}")

    print(f"\n所有图像超分完成！结果保存在：{sr_dir}")


# -------------------------- 配置参数（修改为你的目录）--------------------------
LR_DIR = "./datasets/LR"  # LR图像文件夹（可由之前的hr2lr.py生成）
SR_DIR = "./datasets/Bicubic_SR"  # 超分结果保存路径
SCALE = 4  # 超分倍数
# -------------------------------------------------------------------------------

if __name__ == "__main__":
    bicubic_super_resolution(LR_DIR, SR_DIR, SCALE)