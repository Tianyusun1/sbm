import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import logging
from transformers import ViTForImageClassification
from tqdm import tqdm
import chardet
import json
import shutil
import time
import safetensors.torch
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support

# ==============================================================================
# 关键修改：导入 Chimera-KAN 2.0 的核心模块
# (替代了原文件中定义的 MultiTaskDataset 和 ViTForMultiTaskLearningWithVIM 类)
# ==============================================================================
from dataset import MultiModalDataset
from model import MultiModalChimeraKAN

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# 屏蔽 warning 级别的日志输出
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addFilter(lambda record: record.levelno != logging.WARNING)

# 检测可用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


if __name__ == "__main__":

    # 定义变换参数
    resize_size = (224, 224)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    rescale_factor = 1.0 / 255.0  # 动态计算重缩放因子

    # 定义变换
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # ==============================================================================
    # 关键修改 1：使用支持音频的 MultiModalDataset
    # ==============================================================================
    # 假设你的音频文件存放在 label_monkey 目录下的 audio 子文件夹中
    # 如果路径不同，请修改 audio_dir
    dataset = MultiModalDataset(
        image_dir='/home/klj/pyfile/label_monkey',
        audio_dir='/home/klj/pyfile/label_monkey/audio',  # 新增音频路径
        label_path='/home/klj/pyfile/label_monkey/label.csv',
        img_transform=transform
    )

    # 获取数据的特征和分组信息
    # 注意：MultiModalDataset 的数据存储在 self.data 中，而不是 self.labels
    X = dataset.data
    groups = dataset.data.iloc[:, 4]  # 假设 group_id 是第 5 列（索引为 4）

    # 划分训练集和剩余数据（验证集+测试集）
    splitter = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=42)
    train_indices, temp_indices = next(splitter.split(X, groups=groups))

    # 从剩余数据中划分验证集和测试集
    splitter = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
    val_indices, test_indices = next(splitter.split(X.iloc[temp_indices], groups=groups.iloc[temp_indices]))

    # 创建数据集的子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model_path = '/home/klj/pyfile/premodel/vit-base-patch-224-in21k'  # 替换为你的本地预训练模型路径
    vit_model = ViTForImageClassification.from_pretrained(model_path, local_files_only=True)

    # ==============================================================================
    # 关键修改 2：初始化 Chimera-KAN 2.0 模型
    # ==============================================================================
    model = MultiModalChimeraKAN(
        vit_model=vit_model,
        individual_num_labels=59,  # 个体分类的标签数量
        gender_num_labels=2,       # 性别分类的标签数量
        emotion_num_labels=5,      # 情感分类的标签数量
        unified_dim=768,           # 对齐维度
        num_chimera_layers=2       # 融合层数
    )

    # 将模型移动到设备
    model.to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # 定义字典来保存每个 epoch 的结果
    all_results = []

    # 训练模型
    model.train()
    num_epochs = 30
    for epoch in range(num_epochs):
        # 记录每个 epoch 的开始时间
        epoch_start_time = time.time()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # 将数据移动到设备
            pixel_values = batch['pixel_values'].to(device)
            # 关键修改：获取音频数据
            audio_values = batch['audio_values'].to(device)
            
            emotion_labels = batch['labels']['emotion'].to(device)
            individual_labels = batch['labels']['individual'].to(device)
            gender_labels = batch['labels']['gender'].to(device)

            # 关键修改：前向传播传入音频
            outputs = model(
                pixel_values=pixel_values,
                audio_values=audio_values,
                labels=emotion_labels,
                individual_labels=individual_labels,
                gender_labels=gender_labels
            )
            loss = outputs['loss']
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 更新进度条信息
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch}, Avg Loss: {avg_loss}")

        # 记录每个 epoch 的训练运行时间（秒）
        train_runtime = time.time() - epoch_start_time

        # 计算训练样本每秒处理速度
        train_samples_per_second = len(train_loader.dataset) / train_runtime

        # 计算训练步骤每秒处理速度
        train_steps_per_second = len(train_loader) / train_runtime

        # 验证集评估
        model.eval()
        val_start_time = time.time()  # 记录验证开始时间
        with torch.no_grad():
            total_correct_emotion = 0
            total_correct_individual = 0
            total_correct_gender = 0
            total_samples = 0
            val_loss_total = 0.0  # 记录验证集的总损失

            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                audio_values = batch['audio_values'].to(device) # 验证集也需要音频
                
                emotion_labels = batch['labels']['emotion'].to(device)
                individual_labels = batch['labels']['individual'].to(device)
                gender_labels = batch['labels']['gender'].to(device)

                outputs = model(
                    pixel_values=pixel_values,
                    audio_values=audio_values,
                    labels=emotion_labels,
                    individual_labels=individual_labels,
                    gender_labels=gender_labels  # 计算验证集的损失
                )
                loss = outputs['loss']
                val_loss_total += loss.item()

                emotion_logits = outputs['emotion_logits']
                individual_logits = outputs['individual_logits']
                gender_logits = outputs['gender_logits']

                _, predicted_emotion = torch.max(emotion_logits, 1)
                _, predicted_individual = torch.max(individual_logits, 1)
                _, predicted_gender = torch.max(gender_logits, 1)

                total_correct_emotion += (predicted_emotion == emotion_labels).sum().item()
                total_correct_individual += (predicted_individual == individual_labels).sum().item()
                total_correct_gender += (predicted_gender == gender_labels).sum().item()
                total_samples += emotion_labels.size(0)

            # 计算验证集的平均损失
            avg_val_loss = val_loss_total / len(val_loader)

            emotion_acc = total_correct_emotion / total_samples
            individual_acc = total_correct_individual / total_samples
            gender_acc = total_correct_gender / total_samples

            logging.info(
                f"Validation Emotion Accuracy: {emotion_acc:.4f}, Validation Individual Accuracy: {individual_acc:.4f}, Validation Gender Accuracy: {gender_acc:.4f}, Validation Loss: {avg_val_loss:.4f}")


        val_runtime = time.time() - val_start_time
        val_samples_per_second = len(val_loader.dataset) / val_runtime
        val_steps_per_second = len(val_loader) / val_runtime
        all_results.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_runtime": train_runtime,
            "train_samples_per_second": train_samples_per_second,
            "train_steps_per_second": train_steps_per_second,
            "eval_accuracy_emotion": emotion_acc,
            "eval_accuracy_individual": individual_acc,
            "eval_accuracy_gender": gender_acc,
            "eval_loss": avg_val_loss,
            "eval_runtime": val_runtime,
            "eval_samples_per_second": val_samples_per_second,
            "eval_steps_per_second": val_steps_per_second,
            "total_flos": None  # 这里没有具体的 FLOPs 计算逻辑，暂时设为 None
        })

        model.train()

    # 创建保存目录
    save_dir = "Multi_Model_with_Chimera_V2" # 修改保存目录名以示区分
    os.makedirs(save_dir, exist_ok=True)
    model_weights_path = os.path.join(save_dir, "model.safetensors")
    safetensors.torch.save_file(model.state_dict(), model_weights_path)

    # 生成配置文件时包含 Chimera 的信息
    # 完整保留你要求的字典映射
    complete_config = {
        "architectures": [model.__class__.__name__],
        "attention_probs_dropout_prob": model.vit.config.attention_probs_dropout_prob,
        "encoder_stride": model.vit.config.encoder_stride,
        "finetuning_task": "multi-task-learning-chimera-v2",
        "hidden_act": model.vit.config.hidden_act,
        "hidden_dropout_prob": model.vit.config.hidden_dropout_prob,
        "hidden_size": model.vit.config.hidden_size,
        "image_size": model.vit.config.image_size,
        "initializer_range": model.vit.config.initializer_range,
        "intermediate_size": model.vit.config.intermediate_size,
        "layer_norm_eps": model.vit.config.layer_norm_eps,
        "model_type": model.vit.config.model_type,
        "num_attention_heads": model.vit.config.num_attention_heads,
        "num_channels": model.vit.config.num_channels,
        "num_hidden_layers": model.vit.config.num_hidden_layers,
        "patch_size": model.vit.config.patch_size,
        "problem_type": "multi_label_classification",
        "qkv_bias": model.vit.config.qkv_bias,
        "emotion_labels": 5,
        "emotion_id2label": {
            "0": "angry",
            "1": "curious",
            "2": "fear",
            "3": "happy",
            "4": "neutral"
        },
        "emotion_label2id": {
            "angry": "0",
            "curious": "1",
            "fear": "2",
            "happy": "3",
            "neutral": "4"
        },
        "individual_id2label": {
            "0": "XZ",
            "1": "BM",
            "2": "BT",
            "3": "BX",
            "4": "DD",
            "5": "DX",
            "6": "DYD",
            "7": "EX",
            "8": "FBZ",
            "9": "FB",
            "10": "FC",
            "11": "JJ",
            "12": "KD",
            "13": "KH",
            "14": "LS",
            "15": "PP",
            "16": "FY",
            "17": "None",
            "18": "XH",
            "19": "QB",
            "20": "TS",
            "21": "TT",
            "22": "XJ",
            "23": "XYD",
            "24": "YK",
            "25": "YQ",
            "26": "YD",
            "27": "KZL",
            "28": "HH",
            "29": "HT",
            "30": "LN",
            "31": "OP",
            "32": "Mb",
            "33": "XF",
            "34": "YC",
            "35": "YL",
            "36": "YY",
            "37": "ALA",
            "38": "QQ",
            "39": "SZ",
            "40": "FOP",
            "41": "DT",
            "42": "DW",
            "43": "ET",
            "44": "FXQ",
            "45": "JH",
            "46": "KK",
            "47": "MM",
            "48": "NN",
            "49": "QY",
            "50": "TBG",
            "51": "WX",
            "52": "XF2",
            "53": "YB",
            "54": "YD2",
            "55": "YG",
            "56": "YKY",
            "57": "MB",
            "58": "SX"
        },
        "individual_label2id": {
            "XZ": "0",
            "BM": "1",
            "BT": "2",
            "BX": "3",
            "DD": "4",
            "DX": "5",
            "DYD": "6",
            "EX": "7",
            "FBZ": "8",
            "FB": "9",
            "FC": "10",
            "JJ": "11",
            "KD": "12",
            "KH": "13",
            "LS": "14",
            "PP": "15",
            "FY": "16",
            "None": "17",
            "XH": "18",
            "QB": "19",
            "TS": "20",
            "TT": "21",
            "XJ": "22",
            "XYD": "23",
            "YK": "24",
            "YQ": "25",
            "YD": "26",
            "KZL": "27",
            "HH": "28",
            "HT": "29",
            "LN": "30",
            "OP": "31",
            "Mb": "32",
            "XF": "33",
            "YC": "34",
            "YL": "35",
            "YY": "36",
            "ALA": "37",
            "QQ": "38",
            "SZ": "39",
            "FOP": "40",
            "DT": "41",
            "DW": "42",
            "ET": "43",
            "FXQ": "44",
            "JH": "45",
            "KK": "46",
            "MM": "47",
            "NN": "48",
            "QY": "49",
            "TBG": "50",
            "WX": "51",
            "XF2": "52",
            "YB": "53",
            "YD2": "54",
            "YG": "55",
            "YKY": "56",
            "MB": "57",
            "SX": "58"
        },
        "gender_id2label": {
            "0": "Female",
            "1": "Male"
        },
        "gender_label2id": {
            "Female": "0",
            "Male": "1"
        },
        # 更新配置信息，体现创新点
        "innovations": [
            "Inverse-KAN Audio Hallucination",
            "Acoustic-Guided Mamba Scanning",
            "Orthogonal Subspace Projection"
        ]
    }

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(complete_config, f, ensure_ascii=False, indent=4)

    logging.info(f"Complete config saved to {config_path}")


    all_results_path = os.path.join(save_dir, "all_results.json")
    with open(all_results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    logging.info(f"All training results saved to {all_results_path}")

    # 生成 preprocessor_config.json
    preprocessor_config = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": mean,
        "image_processor_type": "ViTImageProcessor",
        "image_std": std,
        "resample": 2,
        "rescale_factor": rescale_factor,
        "size": {
            "height": resize_size[0],
            "width": resize_size[1]
        }
    }

    preprocessor_config_path = os.path.join(save_dir, "preprocessor_config.json")
    with open(preprocessor_config_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessor_config, f, ensure_ascii=False, indent=4)

    logging.info(f"preprocessor_config saved to {preprocessor_config_path}")

    # 最终测试集评估
    model.eval()
    with torch.no_grad():
        total_correct_emotion = 0
        total_correct_individual = 0
        total_correct_gender = 0
        total_samples = 0

        # 准备存储预测结果和真实标签
        all_emotion_preds = []
        all_emotion_labels = []
        all_individual_preds = []
        all_individual_labels = []
        all_gender_preds = []
        all_gender_labels = []

        for batch in test_loader:
            pixel_values = batch['pixel_values'].to(device)
            audio_values = batch['audio_values'].to(device) # 测试集加入音频

            emotion_labels = batch['labels']['emotion'].to(device)
            individual_labels = batch['labels']['individual'].to(device)
            gender_labels = batch['labels']['gender'].to(device)

            outputs = model(pixel_values=pixel_values, audio_values=audio_values)
            emotion_logits = outputs['emotion_logits']
            individual_logits = outputs['individual_logits']
            gender_logits = outputs['gender_logits']

            _, predicted_emotion = torch.max(emotion_logits, 1)
            _, predicted_individual = torch.max(individual_logits, 1)
            _, predicted_gender = torch.max(gender_logits, 1)

            total_correct_emotion += (predicted_emotion == emotion_labels).sum().item()
            total_correct_individual += (predicted_individual == individual_labels).sum().item()
            total_correct_gender += (predicted_gender == gender_labels).sum().item()
            total_samples += emotion_labels.size(0)

            # 收集预测结果和真实标签
            all_emotion_preds.extend(predicted_emotion.cpu().numpy())
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
            all_individual_preds.extend(predicted_individual.cpu().numpy())
            all_individual_labels.extend(individual_labels.cpu().numpy())
            all_gender_preds.extend(predicted_gender.cpu().numpy())
            all_gender_labels.extend(gender_labels.cpu().numpy())

        emotion_acc = total_correct_emotion / total_samples
        individual_acc = total_correct_individual / total_samples
        gender_acc = total_correct_gender / total_samples
        logging.info(
            f"Test Emotion Accuracy: {emotion_acc:.4f}, Test Individual Accuracy: {individual_acc:.4f}, Test Gender Accuracy: {gender_acc:.4f}")

        # 计算情绪识别的Precision, Recall, F1
        emotion_precision, emotion_recall, emotion_f1, _ = precision_recall_fscore_support(
            all_emotion_labels, all_emotion_preds, average='macro')
        logging.info(f"Test Emotion Precision: {emotion_precision:.4f}, Recall: {emotion_recall:.4f}, F1: {emotion_f1:.4f}")

        # 计算个体识别的Precision, Recall, F1
        individual_precision, individual_recall, individual_f1, _ = precision_recall_fscore_support(
            all_individual_labels, all_individual_preds, average='macro')
        logging.info(f"Test Individual Precision: {individual_precision:.4f}, Recall: {individual_recall:.4f}, F1: {individual_f1:.4f}")

        # 计算性别识别的Precision, Recall, F1
        gender_precision, gender_recall, gender_f1, _ = precision_recall_fscore_support(
            all_gender_labels, all_gender_preds, average='macro')
        logging.info(f"Test Gender Precision: {gender_precision:.4f}, Recall: {gender_recall:.4f}, F1: {gender_f1:.4f}")

        # 记录测试结果
        test_results = {
            "emotion_accuracy": emotion_acc,
            "emotion_precision": emotion_precision,
            "emotion_recall": emotion_recall,
            "emotion_f1": emotion_f1,
            "individual_accuracy": individual_acc,
            "individual_precision": individual_precision,
            "individual_recall": individual_recall,
            "individual_f1": individual_f1,
            "gender_accuracy": gender_acc,
            "gender_precision": gender_precision,
            "gender_recall": gender_recall,
            "gender_f1": gender_f1
        }

        # 保存测试结果到文件
        test_results_path = os.path.join(save_dir, "test_results.json")
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)

        logging.info(f"Test results saved to {test_results_path}")

    # 保存训练日志
    shutil.copy("training.log", os.path.join(save_dir, "training.log"))

    logging.info(f"All training-related files saved to {save_dir}")