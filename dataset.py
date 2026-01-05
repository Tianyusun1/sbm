import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from PIL import Image
import chardet
import logging

class MultiModalDataset(Dataset):
    def __init__(self, image_dir, audio_dir, label_path, img_transform=None, target_sample_rate=16000, max_audio_len=128):
        """
        Chimera-KAN 2.0 增强版多模态数据集加载器
        新增：音频能量序列提取，用于驱动 AG-Mamba 的动态扫描步长。
        """
        self.image_dir = image_dir
        self.audio_dir = audio_dir
        self.img_transform = img_transform
        self.target_sample_rate = target_sample_rate
        self.max_audio_len = max_audio_len

        # 1. 自动检测 CSV 编码并读取
        with open(label_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
        
        self.data = pd.read_csv(label_path, encoding=encoding)
        
        # 2. 处理音频文件名推断
        if 'audio_name' not in self.data.columns:
            print("【提示】正在根据图片名自动推断对应的理毛/社交音频文件 (.wav)...")
            self.data['audio_name'] = self.data.iloc[:, 0].apply(lambda x: os.path.splitext(str(x))[0] + '.wav')

        # 3. 过滤无效样本
        valid_indices = []
        missing_count = 0
        for idx in range(len(self.data)):
            img_name = self.data.iloc[idx, 0]
            audio_name = self.data.iloc[idx]['audio_name']
            
            img_full_path = os.path.join(self.image_dir, str(img_name))
            audio_full_path = os.path.join(self.audio_dir, str(audio_name))
            
            if os.path.isfile(img_full_path) and os.path.isfile(audio_full_path):
                valid_indices.append(idx)
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"【警告】过滤掉了 {missing_count} 个缺失的理毛场景样本。")

        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        print(f"Dataset 2.0 初始化完成: 共 {len(self.data)} 个有效多模态样本。")

        # 4. 音频变换定义 (Mel Spectrogram)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.data)

    def _get_audio_energy(self, waveform, n_frames):
        """
        创新点 2 支撑：提取 RMS 能量序列作为 Mamba 的控制信号。
        """
        # 使用滑动窗口计算均方根能量 (RMS Energy)
        # 确保窗口大小与 Mel 频谱的 hop_length 一致，实现时域对齐
        hop_length = 512
        window_size = 1024
        
        # 简单的滑动窗口能量计算
        unfolded = waveform.unfold(1, window_size, hop_length)
        energy = torch.sqrt(torch.mean(unfolded**2, dim=-1)) # [1, n_frames]
        
        # 归一化到 [0, 1] 区间
        if energy.max() > 0:
            energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
            
        return energy

    def __getitem__(self, idx):
        # --- A. 获取路径和多任务标签 ---
        row = self.data.iloc[idx]
        img_name = row.iloc[0]
        audio_name = row['audio_name']
        
        emotion_label = int(row.iloc[1])
        individual_label = int(row.iloc[2]) 
        gender_label = int(row.iloc[3])

        # --- B. 处理图像 ---
        img_path = os.path.join(self.image_dir, str(img_name))
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)

        # --- C. 处理音频与能量控制信号 ---
        audio_path = os.path.join(self.audio_dir, str(audio_name))
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 1. 生成用于模型输入的 Mel 频谱图
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec)

        # 2. 提取用于驱动 Mamba Delta 的能量包络
        # 这里的能量序列将直接决定 Mamba 扫描视觉 Token 的“采样率”
        audio_energy = self._get_audio_energy(waveform, mel_spec.shape[2])

        # 3. 尺寸对齐 (Padding/Crop)
        target_time = self.max_audio_len
        current_time = mel_spec.shape[2]

        if current_time < target_time:
            pad_amt = target_time - current_time
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amt))
            audio_energy = torch.nn.functional.pad(audio_energy, (0, pad_amt))
        else:
            mel_spec = mel_spec[:, :, :target_time]
            audio_energy = audio_energy[:, :target_time]

        # 最终返回
        return {
            'pixel_values': image,
            'audio_values': mel_spec,    # 频谱输入 [1, 128, 128]
            'audio_energy': audio_energy, # 创新 2 控制信号 [1, 128]
            'labels': {
                'emotion': torch.tensor(emotion_label, dtype=torch.long),
                'individual': torch.tensor(individual_label, dtype=torch.long),
                'gender': torch.tensor(gender_label, dtype=torch.long)
            }
        }