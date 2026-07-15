# 혼돈행렬 만드는 코드
# th맞춰주면 그 값에 따라서 생성
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os
import numpy as np

# 비율 저장용 리스트
ratios = []
th = 0.5
# threshold = 4.1

BASE_SAVE_PATH = os.environ.get(
    "DAAD_CONFUSION_BASE_SAVE_PATH",
    os.environ.get(
        "SIMPLENET_CONFUSION_BASE_SAVE_PATH",
        "/26_0526 /ablation/8/run/models/0/savefile",
    ),
)

LINE_PLOT_PATH = os.path.join(BASE_SAVE_PATH, "line_plot")
EXCEL_PATH = os.path.join(BASE_SAVE_PATH, "excel")
HIGH_SCORE_PATH = os.path.join(BASE_SAVE_PATH, "high_score_plot")
LOW_SCORE_PATH = os.path.join(BASE_SAVE_PATH, "low_score_plot")


def plot_scores_from_excel(excel_path):
    scores_df = pd.read_excel(excel_path)
    scores = scores_df["Scores"].values

    greater_than_th = len([score for score in scores if score > th])
    total_data = len(scores)
    ratio = greater_than_th / total_data * 100

    title = excel_path.split('/')[-1].replace('_scores.xlsx', '')
    ratios.append({"File": title, "Ratio": ratio})




# D & N 파일 처리
score_files = [
    filename for filename in os.listdir(EXCEL_PATH)
    if filename.endswith("_scores.xlsx") and (filename.startswith("N_") or filename.startswith("D_"))
]

def score_sort_key(filename):
    title = filename.replace("_scores.xlsx", "")
    label, index = title.split("_", 1)
    return (0 if label == "N" else 1, int(index))

for score_file in sorted(score_files, key=score_sort_key):
    plot_scores_from_excel(os.path.join(EXCEL_PATH, score_file))

# 비율을 DataFrame으로 변환 및 저장
ratios_df = pd.DataFrame(ratios)

# # 혼돈 행렬 계산

d_ratios = ratios_df[ratios_df["File"].str.startswith("D")]["Ratio"]
n_ratios = ratios_df[ratios_df["File"].str.startswith("N")]["Ratio"]
#
# tp = sum(d_ratios > threshold)  # D 파일 중 abnormal(정답)으로 맞춘 수
# fn = sum(d_ratios <= threshold)  # D 파일 중 normal로 잘못 판단한 수
# fp = sum(n_ratios > threshold)  # N 파일 중 abnormal로 잘못 판단한 수
# tn = sum(n_ratios <= threshold)  # N 파일 중 normal(정답)으로 맞춘 수
#
# accuracy = (tp + tn) / (tp + tn + fp + fn)
# precision = tp / (tp + fp) if (tp + fp) != 0 else 0
# recall = tp / (tp + fn) if (tp + fn) != 0 else 0
# f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
#
#
# # 혼돈 행렬 출력
# print("Confusion Matrix:")
# print(f"TP (True Positive, D에서 {threshold} 초과): {tp}")
# print(f"FN (False Negative, D에서 {threshold} 이하): {fn}")
# print(f"FP (False Positive, N에서 {threshold} 초과): {fp}")
# print(f"TN (True Negative, N에서 {threshold} 이하): {tn}")
# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-Score: {f1_score:.2f}")

# AUROC 커브 계산 및 출력
# 라벨: D는 양성(1), N은 음성(0)
y_true = [1] * len(d_ratios) + [0] * len(n_ratios)
y_scores = list(d_ratios) + list(n_ratios)

# ROC 커브 계산
fpr, tpr, _ = roc_curve(y_true, y_scores)
auroc = auc(fpr, tpr)

# AP
ap_score = average_precision_score(y_true, y_scores)

# PR 커브 계산
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
auprc = auc(recall_curve, precision_curve)




total_images = int(os.environ.get("DAAD_TEST_TOTAL_IMAGES", os.environ.get("SIMPLENET_TEST_TOTAL_IMAGES", "0")))
total_seconds = float(os.environ.get("DAAD_TEST_TOTAL_SECONDS", os.environ.get("SIMPLENET_TEST_TOTAL_SECONDS", "0")))
num_datasets = int(os.environ.get("DAAD_TEST_NUM_DATASETS", os.environ.get("SIMPLENET_TEST_NUM_DATASETS", "0")))
seconds_per_image = total_seconds / total_images if total_images else 0
images_per_second = total_images / total_seconds if total_seconds else 0
seconds_per_dataset = total_seconds / num_datasets if num_datasets else 0

print(f"AUROC: {auroc:.3f}")
print(f"AUPRC: {auprc:.3f}")
print(f"Average Precision (AP): {ap_score:.3f}")
print(f"Test total images: {total_images}")
print(f"Test total time (sec): {total_seconds:.6f}")
print(f"Seconds per image: {seconds_per_image:.8f}")
print(f"Images per second: {images_per_second:.3f}")
print(f"Seconds per patient/dataset: {seconds_per_dataset:.6f}")

# 평가 결과를 txt 파일로 저장
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(BASE_SAVE_PATH)))
evaluation_results_path = os.path.join(PARENT_PATH, "evaluation_results.txt")

# evaluation_results_path = os.path.join(BASE_SAVE_PATH, "evaluation_results.txt")



# 통계 + 시각화

df = ratios_df.copy()

# 정상/비정상 구분
df["Label"] = df["File"].apply(lambda x: "Disease" if x.startswith("D") else "Normal")

# 그룹별 데이터 추출
normal_ratios = df[df["Label"] == "Normal"]["Ratio"]
disease_ratios = df[df["Label"] == "Disease"]["Ratio"]

# 그룹별 통계 출력
group_stats = df.groupby("Label")["Ratio"].agg(["mean", "std", "min", "max", "count"])
print("=== Group-wise Ratio Statistics ===")
print(group_stats)

with open(evaluation_results_path, "w") as file:
    file.write("Evaluation Results\n")
    file.write("==================\n")
    file.write(f"AUROC: {auroc:.3f}\n")
    file.write(f"AUPRC: {auprc:.3f}\n")
    file.write(f"Average Precision (AP): {ap_score:.3f}\n")
    file.write("\n\nTest Speed\n")
    file.write("==========\n")
    file.write(f"Total test images: {total_images}\n")
    file.write(f"Total test time (sec): {total_seconds:.6f}\n")
    file.write(f"Seconds per image: {seconds_per_image:.8f}\n")
    file.write(f"Images per second: {images_per_second:.3f}\n")
    file.write(f"Seconds per patient/dataset: {seconds_per_dataset:.6f}\n")
    file.write("\n\nGroup-wise Abnormal Ratio Statistics\n")
    file.write("====================================\n")
    file.write(group_stats.to_string())
    file.write("\n")

print(f"Evaluation results saved at: {evaluation_results_path}")
