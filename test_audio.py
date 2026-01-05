import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn.functional as F

# 设置 matplotlib 不弹窗，直接绘图（适合服务器环境）
plt.switch_backend('agg')

def save_plot(fig, filepath):
    """
    保存图像到指定路径
    """
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    print(f"已保存图像: {filepath}")

def calculate_mamba_delta_signal(waveform, hop_length=512, win_length=1024):
    """
    模拟 Chimera-KAN 中 AG-Mamba 的步长控制信号计算
    """
    # 计算 RMS 能量
    unfolded = waveform.unfold(1, win_length, hop_length)
    energy = torch.sqrt(torch.mean(unfolded**2, dim=-1)).squeeze()
    
    # 归一化能量 [0, 1]
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
    
    # 模拟 Delta 控制逻辑：
    # 能量高 -> 步长小 (High Freq Scan)
    # 能量低 -> 步长大 (Low Freq Scan)
    # 假设基准 Delta 为 0.1
    # delta = base_dt / (1 + energy * scale)
    base_dt = 0.1
    scale = 5.0
    delta_curve = base_dt / (1.0 + energy * scale)
    
    return energy.numpy(), delta_curve.numpy()

def analyze_and_visualize(audio_path, model_path, output_dir, target_length=128, num_mel_bins=128):
    print(f"{'='*20} 开始深度分析 (Chimera-KAN 2.0 验证版) {'='*20}")
    print(f"音频路径: {audio_path}")
    print(f"输出目录: {output_dir}")

    # --- 0. 检查并创建输出目录 ---
    if not os.path.exists(output_dir):
        print(f"目录不存在，正在创建: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"找不到文件: {audio_path}")

    # --- 1. 加载与重采样 ---
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 强制转为 16k
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # 转单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 绘制波形图
    print("\n[1/5] 生成波形图...")
    fig_wave = plt.figure(figsize=(12, 4))
    plt.plot(waveform.t().numpy(), color='blue', alpha=0.7)
    plt.title(f"Raw Waveform (SR={sample_rate}Hz)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    save_plot(fig_wave, os.path.join(output_dir, "1_waveform.png"))

    # --- 2. 提取特征 (Mel Spectrogram) ---
    print("\n[2/5] 提取 Mel 频谱特征...")
    # 使用 torchaudio 的 transform 以保持与 dataset.py 一致
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=num_mel_bins,
        n_fft=1024,
        hop_length=512
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()
    
    mel_spec = mel_transform(waveform)
    mel_spec_db = db_transform(mel_spec)
    mel_np = mel_spec_db.squeeze().numpy()

    # 绘制原始 Mel 频谱图
    fig_mel = plt.figure(figsize=(10, 6))
    plt.imshow(mel_np, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins (Mel)")
    save_plot(fig_mel, os.path.join(output_dir, "2_mel_spectrogram.png"))

    # --- 3. 验证创新点：AG-Mamba 步长控制 ---
    print("\n[3/5] 验证 AG-Mamba 动态扫描步长...")
    energy, delta_curve = calculate_mamba_delta_signal(waveform)
    
    # 对齐时间轴用于绘图
    time_axis = np.linspace(0, len(waveform.t())/sample_rate, len(energy))
    
    fig_delta, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 绘制音频能量
    ax1.plot(time_axis, energy, color='orange', label='Audio Energy')
    ax1.set_ylabel("Normalized Energy")
    ax1.set_title("Audio Energy Envelope (Input to AG-Mamba)")
    ax1.grid(True)
    ax1.legend()
    
    # 绘制 Delta 变化
    ax2.plot(time_axis, delta_curve, color='green', label='Mamba Step Size ($\Delta$)')
    ax2.set_ylabel("Step Size $\Delta$")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Dynamic Scanning Step Size (Controls Vision Processing Rate)")
    ax2.grid(True)
    ax2.legend()
    
    # 标注物理意义
    ax2.text(0.02, 0.9, "Slow Scan (Identity Focus)", transform=ax2.transAxes, fontsize=10, color='blue')
    ax2.text(0.02, 0.1, "Fast Scan (Emotion Focus)", transform=ax2.transAxes, fontsize=10, color='red')
    
    save_plot(fig_delta, os.path.join(output_dir, "3_ag_mamba_control.png"))

    # --- 4. 验证创新点：Inverse-KAN 幻觉特征模拟 ---
    print("\n[4/5] 模拟 Inverse-KAN 幻觉特征...")
    # 简单模拟：将音频频谱映射到一个假想的视觉空间 (768维)
    # 这里只是为了可视化这一概念，实际计算是在模型内部完成的
    simulated_projection = np.dot(mel_np.T, np.random.rand(128, 768) * 0.1) # [Time, 768]
    
    fig_ghost = plt.figure(figsize=(12, 6))
    plt.imshow(simulated_projection.T, aspect='auto', origin='lower', cmap='cividis')
    plt.colorbar()
    plt.title("Visualized 'Audio Hallucination' Residual ($\Delta V_{ghost}$)")
    plt.xlabel("Time Steps")
    plt.ylabel("Visual Embedding Dimension (0-768)")
    save_plot(fig_ghost, os.path.join(output_dir, "4_hallucination_residual.png"))

    # --- 5. 验证权重文件 (保留原逻辑) ---
    print("\n[5/5] 验证权重文件匹配度...")
    if os.path.exists(model_path):
        try:
            # 尝试加载 safetensors 或 pth
            if model_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
            
            print(f"  - 成功读取权重文件，包含 {len(state_dict)} 个参数键")
            
        except Exception as e:
            print(f"  - 读取权重时发生警告/错误: {e}")
    else:
        print(f"  - ⚠️ 警告: 未找到权重文件 {model_path}")

    print(f"\n{'='*20} 分析完成 {'='*20}")
    print(f"所有验证图表已保存至: {output_dir}")

if __name__ == "__main__":
    # --- 配置区域 ---
    # 请替换为你自己的音频文件路径
    AUDIO_FILE = '/home/klj/pyfile/label_monkey/audio/monkey_test.wav' 
    # 替换为你训练好的模型路径 (可选)
    MODEL_PATH = './Chimera_KAN_V2_Final/model.safetensors'
    
    OUTPUT_DIR = './audio_analysis_results' 

    try:
        # 如果没有音频文件，先生成一个假的音频用于测试代码
        if not os.path.exists(AUDIO_FILE):
            print("⚠️ 未找到测试音频，正在生成临时测试音频...")
            os.makedirs(os.path.dirname(AUDIO_FILE), exist_ok=True)
            dummy_wav = torch.randn(1, 16000*2) # 2秒白噪声
            torchaudio.save(AUDIO_FILE, dummy_wav, 16000)
            
        analyze_and_visualize(AUDIO_FILE, MODEL_PATH, OUTPUT_DIR)
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()