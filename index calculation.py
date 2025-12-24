import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def calculate_metrics_for_task(df, true_col, pred_col, task_name):
    y_true = df[true_col]
    y_pred = df[pred_col]

    # 过滤掉 NaN（如果有）
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    print(f"\n=== {task_name} 分类报告 ===")
    print(classification_report(y_true, y_pred, zero_division=0))

    print(f"\n=== {task_name} 汇总指标 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"加权精确率: {precision:.4f}")
    print(f"加权召回率: {recall:.4f}")
    print(f"加权F1值: {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":
    # 替换为你的实际路径
    csv_path = "Swin2\\test_predictions.csv"
    df = pd.read_csv(csv_path)

    # 分别评估三个任务
    emotion_metrics = calculate_metrics_for_task(df, 'true_emotion', 'pred_emotion', '情绪')
    individual_metrics = calculate_metrics_for_task(df, 'true_individual', 'pred_individual', '个体')
    gender_metrics = calculate_metrics_for_task(df, 'true_gender', 'pred_gender', '性别')