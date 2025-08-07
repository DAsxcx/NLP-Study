from datasets import load_dataset
from transformers import pipeline, AutoConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch

def evaluate_model_on_imdb_test(model_path, batch_size=32, model_name=None):
    dataset = load_dataset("imdb", split="test")
    texts = dataset["text"]
    y_true = dataset["label"]

    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=model_path,
        device=device
    )

    config = AutoConfig.from_pretrained(model_path)
    label2id = config.label2id

    # 选择正类标签
    positive_label = None
    for k, v in label2id.items():
        if "POS" in k.upper() or v == 1:
            positive_label = k
            break
    if positive_label is None:
        positive_label = sorted(label2id.keys())[1]

    y_pred = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        results = classifier(batch_texts, truncation=True, max_length=512)
        for r in results:
            label = 1 if r["label"] == positive_label else 0
            y_pred.append(label)

    # 计算评估指标
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    name = model_name or model_path
    print(f"\n模型评估结果: {name}")
    print(f"{'='*50}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"{'='*50}\n")

    return {
        "model": name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

TEACHER_MODEL_PATH = "./imdb_teacher_bert_full"
STUDENT_MODEL_PATH = "./imdb_student_bert_distilled"

#教师模型
teacher_results = evaluate_model_on_imdb_test(
    model_path=TEACHER_MODEL_PATH,
    model_name="Teacher Model (BERT-base)"
)

#学生模型
student_results = evaluate_model_on_imdb_test(
    model_path=STUDENT_MODEL_PATH,
    model_name="Student Model (BERT-mini)"
)

print(f"Teacher Accuracy: {teacher_results['accuracy']:.4f}")
print(f"Student Accuracy: {student_results['accuracy']:.4f}")
