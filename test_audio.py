import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import numpy as np

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

def analyze_and_visualize(audio_path, model_path, output_dir, target_length=1024, num_mel_bins=128):
    print(f"{'='*20} 开始深度分析 {'='*20}")
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
    print("\n[1/4] 生成波形图...")
    fig_wave = plt.figure(figsize=(12, 4))
    plt.plot(waveform.t().numpy(), color='blue', alpha=0.7)
    plt.title(f"Raw Waveform (SR={sample_rate}Hz)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # 保存到指定目录
    save_plot(fig_wave, os.path.join(output_dir, "1_waveform.png"))

    # --- 2. 提取特征 (Mel Spectrogram) ---
    print("\n[2/4] 提取 Mel 频谱特征...")
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_shift=10
    )
    
    fbank_np = fbank.numpy()

    # 绘制原始 Mel 频谱图
    fig_mel = plt.figure(figsize=(10, 6))
    plt.imshow(fbank_np.T, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram (Before Normalization)")
    plt.xlabel("Time Frames (10ms)")
    plt.ylabel("Frequency Bins (Mel)")
    
    # 保存到指定目录
    save_plot(fig_mel, os.path.join(output_dir, "2_mel_spectrogram.png"))

    # --- 3. 预处理 (Padding & Normalization) ---
    print("\n[3/4] 执行模型预处理 (Padding & Norm)...")
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    
    # Padding
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank_padded = m(fbank)
    elif p < 0:
        fbank_padded = fbank[0:target_length, :]
    else:
        fbank_padded = fbank

    # Normalization (AST AudioSet specific)
    fbank_norm = (fbank_padded - (-4.2677393)) / (4.5689974 * 2)
    
    # 绘制模型真正“看到”的输入图
    fig_input = plt.figure(figsize=(10, 6))
    plt.imshow(fbank_norm.numpy().T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title("Model Input Tensor (Normalized & Padded)")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    
    # 保存到指定目录
    save_plot(fig_input, os.path.join(output_dir, "3_model_input_visualized.png"))

    # --- 4. 验证权重文件 ---
    print("\n[4/4] 验证权重文件匹配度...")
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # 简单的统计
            print(f"  - 成功读取权重文件，包含 {len(state_dict)} 个参数键")
            
        except Exception as e:
            print(f"  - 读取权重时发生警告/错误: {e}")
    else:
        print(f"  - ⚠️ 警告: 未找到权重文件 {model_path}")

    print(f"\n{'='*20} 分析完成 {'='*20}")
    print(f"所有图片已保存至: {output_dir}")

if __name__ == "__main__":
    # --- 配置区域 ---
    AUDIO_FILE = '/media/dl-d/Data/sty/audio/Affiliative/Monkey_Affiliative_0001.wav'
    MODEL_PATH = '/media/dl-d/Data/sty/huggingface/audio/audioset_0.4593.pth'
    
    #在此处指定你想要的输出路径
    OUTPUT_DIR = './audio_analysis_results' 
    # 或者绝对路径，例如: '/home/sty/pyfile/SBM/results/monkey_001'

    try:
        analyze_and_visualize(AUDIO_FILE, MODEL_PATH, OUTPUT_DIR)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()