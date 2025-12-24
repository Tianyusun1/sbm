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
from fasterkan import FasterKANLayer
import shutil
import time
import safetensors.torch
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from vim import MambaConfig, VMamba
from vim_mamba_blocks import GlobalMambaBlockFromMamba, EnhancedLocalMambaBlock
from dataset import MultiModalDataset

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

# 定义多任务学习模型
class ViTForMultiTaskLearningWithVIM(torch.nn.Module):
    def __init__(self, vit_model, individual_num_labels, gender_num_labels, emotion_num_labels, mamba_config=None):
        super().__init__()
        self.vit = vit_model
        hidden_size = vit_model.config.hidden_size

        # 添加 Vision Mamba 模块
        if mamba_config is None:
            mamba_config = MambaConfig(
                d_model=vit_model.config.hidden_size,
                n_layers=4,
                dt_rank='auto',
                d_state=16,
                expand_factor=2,
                d_conv=4,
                dt_min=0.001,
                dt_max=0.1,
                dt_init="random",
                dt_scale=1.0,
                rms_norm_eps=1e-5,
                bias=False,
                conv_bias=True,
                inner_layernorms=False,
                pscan=True,
                use_cuda=torch.cuda.is_available(),
                bidirectional=True,
                divide_output=True
            )
        self.global_mamba = GlobalMambaBlockFromMamba(hidden_size)
        self.local_mamba = EnhancedLocalMambaBlock(mamba_config, num_layers=2)
        self.mamba_config = mamba_config
        fusion_dim = hidden_size * 2

        # 使用 FasterKANLayer 作为分类器
        self.emotion_classifier = FasterKANLayer(
            input_dim=fusion_dim,
            output_dim=emotion_num_labels,
            grid_min=-2.0,
            grid_max=2.0,
            num_grids=8,
            denominator=0.33,
            spline_weight_init_scale=0.1
        )

        self.individual_classifier = FasterKANLayer(
            input_dim=fusion_dim,
            output_dim=individual_num_labels,
            grid_min=-2.0,
            grid_max=2.0,
            num_grids=8,
            denominator=0.33,
            spline_weight_init_scale=0.1
        )

        self.gender_classifier = FasterKANLayer(
            input_dim=fusion_dim,
            output_dim=gender_num_labels,
            grid_min=-2.0,
            grid_max=2.0,
            num_grids=8,
            denominator=0.33,
            spline_weight_init_scale=0.1
        )

    def forward(self, pixel_values, labels=None, individual_labels=None, gender_labels=None):
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=True)
        sequence_output = outputs.hidden_states[-1]  # (B, N+1, D)

        cls_token = sequence_output[:, 0, :].unsqueeze(1)  # (B, 1, D)
        patch_tokens = sequence_output[:, 1:, :]  # (B, N, D)

        global_feat = self.global_mamba(cls_token).squeeze(1)  # (B, D)
        local_feat_seq = self.local_mamba(patch_tokens)  # (B, N, D)
        local_feat = local_feat_seq.mean(dim=1)  # (B, D)

        fused = torch.cat([global_feat, local_feat], dim=-1)  # (B, 2D)

        emotion_logits = self.emotion_classifier(fused)
        individual_logits = self.individual_classifier(fused)
        gender_logits = self.gender_classifier(fused)

        total_loss = None
        if labels is not None or individual_labels is not None or gender_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            emotion_loss = loss_fct(emotion_logits, labels) if labels is not None else 0
            individual_loss = loss_fct(individual_logits, individual_labels) if individual_labels is not None else 0
            gender_loss = loss_fct(gender_logits, gender_labels) if gender_labels is not None else 0
            total_loss = emotion_loss + individual_loss + gender_loss

        return {
            "loss": total_loss,
            "emotion_logits": emotion_logits,
            "individual_logits": individual_logits,
            "gender_logits": gender_logits
        }


# 定义数据集类
class MultiTaskDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        # 自动检测文件编码
        with open(label_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
        self.labels = pd.read_csv(label_path, encoding=encoding)

        # 检查并清理标签列中的非整数值和缺失值
        columns_to_check = [1, 2, 3]  # 假设标签列是第 2、3、4 列（索引从 0 开始）
        self.labels = self.labels.dropna(subset=self.labels.columns[columns_to_check])

        # 确保标签列中的值都是整数类型
        for col in columns_to_check:
            self.labels.iloc[:, col] = pd.to_numeric(self.labels.iloc[:, col], errors='coerce')

        # 删除包含无效整数的行
        self.labels = self.labels.dropna(subset=self.labels.columns[columns_to_check])

        # 确保标签值在合理范围内
        self.labels.iloc[:, 1] = self.labels.iloc[:, 1].apply(lambda x: min(max(x, 0), 4))  # 情绪分类的标签范围是 0 到 4
        self.labels.iloc[:, 2] = self.labels.iloc[:, 2].apply(lambda x: min(max(x, 0), 58))  # 个体分类的标签范围是 0 到 58
        self.labels.iloc[:, 3] = self.labels.iloc[:, 3].apply(lambda x: min(max(x, 0), 1))  # 性别分类的标签范围是 0 到 1

        # 检查图像文件是否存在
        valid_indices = []
        for idx in range(len(self.labels)):
            img_name = self.labels.iloc[idx, 0]
            img_path = os.path.join(self.image_dir, img_name)
            if os.path.isfile(img_path):
                valid_indices.append(idx)
            else:
                logging.warning(f"Image file not found: {img_path}")

        self.labels = self.labels.iloc[valid_indices].reset_index(drop=True)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # 确保图像转换为 RGB 模式
        if self.transform:
            image = self.transform(image)
        emotion_label = int(self.labels.iloc[idx, 1])
        individual_label = int(self.labels.iloc[idx, 2])
        gender_label = int(self.labels.iloc[idx, 3])
        return {
            'pixel_values': image,
            'labels': {
                'emotion': torch.tensor(emotion_label, dtype=torch.long),
                'individual': torch.tensor(individual_label, dtype=torch.long),
                'gender': torch.tensor(gender_label, dtype=torch.long)
            }
        }


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

    # 创建数据集
    dataset = MultiTaskDataset(
        image_dir='/home/klj/pyfile/label_monkey',
        label_path='/home/klj/pyfile/label_monkey/label.csv',
        transform=transform
    )

    # 获取数据的特征和分组信息
    X = dataset.labels
    groups = dataset.labels.iloc[:, 4]  # 假设 group_id 是第 5 列（索引为 4）

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

    # 初始化多任务学习模型（集成 Vision Mamba）
    model = ViTForMultiTaskLearningWithVIM(
        vit_model=vit_model,
        individual_num_labels=59,  # 个体分类的标签数量
        gender_num_labels=2,  # 性别分类的标签数量
        emotion_num_labels=5  # 情感分类的标签数量
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
            emotion_labels = batch['labels']['emotion'].to(device)
            individual_labels = batch['labels']['individual'].to(device)
            gender_labels = batch['labels']['gender'].to(device)

            outputs = model(
                pixel_values=pixel_values,
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
                emotion_labels = batch['labels']['emotion'].to(device)
                individual_labels = batch['labels']['individual'].to(device)
                gender_labels = batch['labels']['gender'].to(device)

                outputs = model(
                    pixel_values=pixel_values,
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
    save_dir = "Multi_Model_with_VIM"
    os.makedirs(save_dir, exist_ok=True)
    model_weights_path = os.path.join(save_dir, "model.safetensors")
    safetensors.torch.save_file(model.state_dict(), model_weights_path)

    # 生成配置文件时包含 Vision Mamba 的信息
    complete_config = {
        "architectures": [model.__class__.__name__],
        "attention_probs_dropout_prob": model.vit.config.attention_probs_dropout_prob,
        "encoder_stride": model.vit.config.encoder_stride,
        "finetuning_task": "multi-task-learning",
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
        # 添加 Vision Mamba 的配置信息
        "mamba_config": {
            "d_model": model.mamba_config.d_model,
            "n_layers": model.mamba_config.n_layers,
            "dt_rank": model.mamba_config.dt_rank,
            "d_state": model.mamba_config.d_state,
            "expand_factor": model.mamba_config.expand_factor,
            "d_conv": model.mamba_config.d_conv,
            "dt_min": model.mamba_config.dt_min,
            "dt_max": model.mamba_config.dt_max,
            "dt_init": model.mamba_config.dt_init,
            "dt_scale": model.mamba_config.dt_scale,
            "rms_norm_eps": model.mamba_config.rms_norm_eps,
            "bias": model.mamba_config.bias,
            "conv_bias": model.mamba_config.conv_bias,
            "inner_layernorms": model.mamba_config.inner_layernorms,
            "pscan": model.mamba_config.pscan,
            "use_cuda": model.mamba_config.use_cuda,
            "bidirectional": model.mamba_config.bidirectional,
            "divide_output": model.mamba_config.divide_output
        }
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
            emotion_labels = batch['labels']['emotion'].to(device)
            individual_labels = batch['labels']['individual'].to(device)
            gender_labels = batch['labels']['gender'].to(device)

            outputs = model(pixel_values=pixel_values)
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