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
        多模态数据集加载器

        Args:
            image_dir (str): 图片文件夹路径
            audio_dir (str): 音频文件夹路径 (存放 .wav 文件)
            label_path (str): CSV 标签文件路径
            img_transform (callable, optional): 图像预处理 transforms
            target_sample_rate (int): 音频重采样率 (默认 16000)
            max_audio_len (int): 频谱图的时间维度长度 (为了对齐模型输入, 默认 128 -> 输出 [1, 128, 128])
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
        
        # 2. 数据清洗与准备
        # 假设 CSV 前几列结构是: [image_name, emotion, individual, gender, ...]
        
        # 关键步骤：处理音频文件名
        # 如果 CSV 里没有 'audio_name' 这一列，代码会自动假设音频文件名 = 图片文件名 (后缀换成 .wav)
        if 'audio_name' not in self.data.columns:
            print("【提示】CSV中未找到 'audio_name' 列, 正在尝试根据图片名自动推断 (.wav)...")
            # 假设图片名是 'monkey1.jpg' -> 音频名 'monkey1.wav'
            self.data['audio_name'] = self.data.iloc[:, 0].apply(lambda x: os.path.splitext(str(x))[0] + '.wav')

        # 3. 过滤无效样本 (确保图片和音频文件都存在)
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
            print(f"【警告】过滤掉了 {missing_count} 个文件缺失的样本。")

        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        print(f"Dataset 初始化完成: 共 {len(self.data)} 个有效多模态样本。")

        # 4. 定义音频变换 (Mel Spectrogram)
        # 这会将音频波形转换为 [128, Time] 的图像形式
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_mels=128,      # 频域高度 (对应模型输入的 height)
            n_fft=1024,
            hop_length=512
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # --- A. 获取路径和标签 ---
        row = self.data.iloc[idx]
        img_name = row.iloc[0]
        audio_name = row['audio_name']
        
        # 读取标签 (假设 emotion=col 1, individual=col 2, gender=col 3)
        # 请根据你实际的 CSV 列索引进行微调
        emotion_label = int(row.iloc[1])
        individual_label = int(row.iloc[2]) 
        gender_label = int(row.iloc[3])

        # --- B. 处理图像 ---
        img_path = os.path.join(self.image_dir, str(img_name))
        image = Image.open(img_path).convert('RGB')
        
        if self.img_transform:
            image = self.img_transform(image)

        # --- C. 处理音频 ---
        audio_path = os.path.join(self.audio_dir, str(audio_name))
        waveform, sample_rate = torchaudio.load(audio_path)

        # 1. 重采样 (如果原始采样率不是 16k)
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # 2. 混合声道 (如果是立体声，转单声道)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. 生成 Mel 频谱图 -> [1, 128, Time]
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec)

        # 4. 尺寸对齐 (Padding 或 Cutting)
        # 目标是 [1, 128, 128] 以匹配 CNN/ViT 输入
        current_time = mel_spec.shape[2]
        target_time = self.max_audio_len

        if current_time < target_time:
            # 如果音频太短，进行填充 (Pad)
            pad_amt = target_time - current_time
            # pad 格式: (left, right, top, bottom) -> 这里只填右边
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amt))
        else:
            # 如果音频太长，进行截断 (Crop)
            mel_spec = mel_spec[:, :, :target_time]

        # 最终 output
        return {
            'pixel_values': image,       # Image Tensor [3, 224, 224]
            'audio_values': mel_spec,    # Audio Tensor [1, 128, 128]
            'labels': {
                'emotion': torch.tensor(emotion_label, dtype=torch.long),
                'individual': torch.tensor(individual_label, dtype=torch.long),
                'gender': torch.tensor(gender_label, dtype=torch.long)
            }
        }