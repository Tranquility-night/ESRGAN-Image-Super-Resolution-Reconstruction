import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random


# -------------------------- 1. 增强版FSRCNN模型（提升特征提取能力）--------------------------
class FSRCNN(nn.Module):
    def __init__(self, scale=4, num_channels=3):
        super(FSRCNN, self).__init__()
        # 特征提取（增加卷积核数量，提升特征捕捉能力）
        self.feature_extraction = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        # 收缩层（保持通道数，避免参数过多）
        self.shrink = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        # 映射层（增加到5层，强化特征映射能力）
        self.map = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 扩张层（对应特征提取层的通道数）
        self.expand = nn.Conv2d(16, 64, kernel_size=1, padding=0)
        # 上采样模块（改用PixelShuffle，比反卷积更清晰，无棋盘格噪点）
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),  # 4倍超分：通道数→64*4*4，Shuffle后变为64通道，尺寸×4
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        )
        # 激活函数（LeakyReLU比ReLU更稳定，避免梯度消失）
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()  # 约束输出在[0,1]，确保色彩正常

    def forward(self, x):
        residual = x  # 残差连接，保留低频特征
        x = self.lrelu(self.feature_extraction(x))
        x = self.lrelu(self.shrink(x))
        x = self.map(x)
        x = self.lrelu(self.expand(x))
        x = self.upsample(x)
        x = self.sigmoid(x + residual)  # 残差融合，提升细节
        return x


# -------------------------- 2. 增强版数据集（修复负步长问题）--------------------------
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir=None, scale=4, patch_size=48, augment=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir  # 若提供LR文件夹，直接使用（确保与HR配对）
        self.scale = scale
        self.patch_size = patch_size  # 增大图像块，捕捉更多细节
        self.augment = augment  # 数据增强（提升模型泛化能力）

        # 加载HR图像路径（过滤无效图像）
        self.hr_img_paths = []
        for f in os.listdir(hr_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(hr_dir, f)
                if cv2.imread(img_path) is not None:
                    self.hr_img_paths.append(img_path)
                else:
                    print(f"跳过损坏HR图像：{f}")

        # 若提供LR文件夹，验证配对（文件名一致）
        self.lr_img_paths = []
        if self.lr_dir is not None:
            for hr_path in self.hr_img_paths:
                lr_name = os.path.basename(hr_path)
                lr_path = os.path.join(lr_dir, lr_name)
                if os.path.exists(lr_path) and cv2.imread(lr_path) is not None:
                    self.lr_img_paths.append(lr_path)
                else:
                    print(f"HR图像{os.path.basename(hr_path)}无对应LR图像，跳过")
            # 确保HR和LR数量一致
            self.hr_img_paths = self.hr_img_paths[:len(self.lr_img_paths)]
        else:
            self.lr_img_paths = None  # 自动生成LR

        if len(self.hr_img_paths) == 0:
            raise ValueError("无有效HR-LR配对图像！请检查文件夹")

    def __len__(self):
        return len(self.hr_img_paths) * 20  # 每张图生成20个块，增加训练数据量

    def augment_img(self, hr_patch, lr_patch):
        """数据增强：随机翻转、旋转（修复负步长问题）"""
        # 随机水平翻转
        if random.random() > 0.5:
            hr_patch = np.fliplr(hr_patch).copy()  # 新增.copy()，消除负步长
            lr_patch = np.fliplr(lr_patch).copy()
        # 随机垂直翻转
        if random.random() > 0.5:
            hr_patch = np.flipud(hr_patch).copy()  # 新增.copy()
            lr_patch = np.flipud(lr_patch).copy()
        # 随机旋转90度
        if random.random() > 0.5:
            hr_patch = np.rot90(hr_patch).copy()  # 新增.copy()
            lr_patch = np.rot90(lr_patch).copy()
        return hr_patch, lr_patch

    def __getitem__(self, idx):
        # 读取HR和LR图像
        img_idx = idx // 20  # 每20个块对应一张图像
        hr_path = self.hr_img_paths[img_idx]
        hr_img = cv2.imread(hr_path)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB) / 255.0  # [0,1]

        # 读取/生成LR图像
        if self.lr_img_paths is not None:
            lr_path = self.lr_img_paths[img_idx]
            lr_img = cv2.imread(lr_path)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB) / 255.0
        else:
            # 自动bicubic下采样生成LR
            lr_h, lr_w = hr_img.shape[0] // self.scale, hr_img.shape[1] // self.scale
            lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

        # 裁剪图像块（确保LR和HR块对应）
        lr_h, lr_w = lr_img.shape[:2]
        hr_h, hr_w = hr_img.shape[:2]

        # 随机裁剪LR块，对应HR块自动对齐（偏移量×scale）
        lr_top = random.randint(0, lr_h - self.patch_size)
        lr_left = random.randint(0, lr_w - self.patch_size)
        lr_patch = lr_img[lr_top:lr_top + self.patch_size, lr_left:lr_left + self.patch_size]

        hr_top = lr_top * self.scale
        hr_left = lr_left * self.scale
        hr_patch = hr_img[hr_top:hr_top + self.patch_size * self.scale,
                   hr_left:hr_left + self.patch_size * self.scale]

        # 数据增强
        if self.augment:
            hr_patch, lr_patch = self.augment_img(hr_patch, lr_patch)

        # 转为Tensor格式（C, H, W）—— 新增.copy()，彻底消除负步长
        hr_patch = torch.tensor(np.transpose(hr_patch.copy(), (2, 0, 1)), dtype=torch.float32)
        lr_patch = torch.tensor(np.transpose(lr_patch.copy(), (2, 0, 1)), dtype=torch.float32)

        return lr_patch, hr_patch


# -------------------------- 3. 训练函数（修复损失函数加权问题）--------------------------
def train_fsrcnn(hr_dir, lr_dir, model_save_path, scale=4, epochs=80, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备：{device} | 训练轮数：{epochs} | 批次大小：{batch_size}")

    # 构建数据集（使用HR和LR配对，数据增强开启）
    dataset = SRDataset(hr_dir, lr_dir, scale=scale, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 初始化模型、损失函数、优化器
    model = FSRCNN(scale=scale).to(device)
    # 分别定义MSE和L1损失函数（修复：不直接对损失函数做乘法）
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)  # AdamW更稳定
    # 学习率调度器（分阶段降低，确保收敛）
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # 训练监控：记录最佳损失（保存最优模型）
    best_loss = float('inf')
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # 前向传播
            sr_imgs = model(lr_imgs)
            # 修复：对损失值进行加权求和（而不是对损失函数加权）
            mse_loss = mse_criterion(sr_imgs, hr_imgs)
            l1_loss = l1_criterion(sr_imgs, hr_imgs)
            loss = mse_loss + 0.1 * l1_loss  # 混合损失：MSE（像素还原）+ L1（减少模糊）

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，避免爆炸
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} | 平均损失：{avg_loss:.6f} | 学习率：{optimizer.param_groups[0]['lr']:.8f}")

        # 保存最优模型（仅保存损失最低的模型）
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"更新最佳模型！当前最低损失：{best_loss:.6f}")

        # 学习率调度
        scheduler.step()

    print(f"\n训练完成！最佳模型保存路径：{model_save_path}")


# -------------------------- 4. 超分推理函数（精细化后处理，确保效果）--------------------------
def fsrcnn_super_resolution(lr_dir, sr_dir, model_path, scale=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载最优模型
    model = FSRCNN(scale=scale).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型加载成功！运行设备：{device}")

    # 创建输出文件夹
    if not os.path.exists(sr_dir):
        os.makedirs(sr_dir)

    # 遍历LR图像超分
    lr_img_names = [f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in lr_img_names:
        lr_path = os.path.join(lr_dir, img_name)
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        if lr_img is None:
            print(f"跳过无法读取的图像：{img_name}")
            continue

        # 预处理（严格匹配训练流程）
        lr_img_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB) / 255.0  # [0,1]
        lr_tensor = torch.tensor(np.transpose(lr_img_rgb.copy(), (2, 0, 1)), dtype=torch.float32)  # 新增.copy()
        lr_tensor = lr_tensor.unsqueeze(0).to(device)

        # 超分推理（禁用梯度，加速+避免内存泄漏）
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        # 后处理（精细化，确保色彩+细节）
        sr_img_rgb = sr_tensor.squeeze(0).cpu().numpy()  # (C, H, W) → (H, W, C)
        sr_img_rgb = np.transpose(sr_img_rgb, (1, 2, 0))
        sr_img_rgb = np.clip(sr_img_rgb, 0.0, 1.0)  # 双重保险，避免溢出
        sr_img = (sr_img_rgb * 255.0).astype(np.uint8)  # [0,255]
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)  # 适配OpenCV保存

        # 锐化增强（可选，进一步提升细节）
        sr_img = cv2.GaussianBlur(sr_img, (0, 0), 1.0)
        sr_img = cv2.addWeighted(sr_img, 1.5, sr_img, -0.5, 0)

        # 无损保存（PNG格式，避免压缩损失）
        sr_path = os.path.join(sr_dir, f"FSRCNN_Best_SR_{img_name}")
        cv2.imwrite(sr_path, sr_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # 输出信息
        lr_h, lr_w = lr_img.shape[:2]
        sr_h, sr_w = sr_img.shape[:2]
        print(f"处理完成：{img_name} | LR:{lr_w}x{lr_h} → SR:{sr_w}x{sr_h} | 保存路径：{sr_path}")

    print(f"\n所有图像超分完成！结果保存在：{sr_dir}")


# -------------------------- 5. 主函数（一键训练+超分）--------------------------
if __name__ == "__main__":
    # -------------------------- 配置参数（必看！根据你的目录修改）--------------------------
    HR_DIR = "./datasets/HR"  # 训练用HR图像文件夹（10~100张，≥512×512）
    LR_DIR = "./datasets/LR"  # 训练+超分用LR图像文件夹（与HR配对）
    SR_DIR = "./datasets/FSRCNN_Best_SR"  # 最终超分结果保存路径
    MODEL_SAVE_PATH = "./fsrcnn_best_model.pth"  # 最优模型保存路径
    SCALE = 4  # 超分倍数（必须与LR生成时一致）
    EPOCHS = 10  # 训练轮数（CPU约30分钟，GPU约8分钟）
    BATCH_SIZE = 8  # 批次大小（CPU=8，GPU=16）
    # -------------------------------------------------------------------------------

    # 第一步：训练最优FSRCNN模型（使用HR-LR配对数据）
    print("=" * 60)
    print("开始训练增强版FSRCNN模型...")
    train_fsrcnn(HR_DIR, LR_DIR, MODEL_SAVE_PATH, SCALE, EPOCHS, BATCH_SIZE)

    # 第二步：超分推理（使用训练好的最优模型）
    print("\n" + "=" * 60)
    print("开始超分推理（使用最优模型）...")
    fsrcnn_super_resolution(LR_DIR, SR_DIR, MODEL_SAVE_PATH, SCALE)