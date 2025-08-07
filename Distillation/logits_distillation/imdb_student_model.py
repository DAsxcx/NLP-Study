from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from torch.utils.data import default_collate

teacher_logits_path = "./imdb_all_logits.pt"
student_model_id = "prajjwal1/bert-mini"
tokenizer = BertTokenizer.from_pretrained(student_model_id)
accuracy = evaluate.load("accuracy")

# 会随机打乱所以需要加索引
dataset = load_dataset("imdb")["train"]
dataset = dataset.map(lambda example,idx:{"index":idx},with_indices=True)
dataset = dataset.train_test_split(test_size=0.1,seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]



teacher_logits = torch.load(teacher_logits_path)  # shape: [num_train, 2]
teacher_logits_list = teacher_logits.tolist()

# 添加教师 logits 到训练集
def add_teacher_logits(example):
    idx = int(example["index"])
    example["teacher_logits"] = teacher_logits_list[idx]
    return example
#只需给训练集添加logits即可
train_dataset = train_dataset.map(add_teacher_logits)


# 分词预处理 
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(preprocess, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")

eval_dataset = eval_dataset.map(preprocess, batched=True)
eval_dataset = eval_dataset.rename_column("label", "labels")

# 教师 logits 转 tensor（仅训练集）
train_dataset = train_dataset.map(lambda e: {"teacher_logits": torch.tensor(e["teacher_logits"]).float()})
train_dataset.set_format(type="torch",columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch",columns=["input_ids", "attention_mask", "labels"])

student_model = BertForSequenceClassification.from_pretrained(student_model_id, num_labels=2)

class DistillationTrainer(Trainer):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        teacher_logits = inputs.pop("teacher_logits", None)  # 验证集不会传入
        outputs = model(**inputs)
        student_logits = outputs.logits

        #训练集
        if teacher_logits is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
                reduction="batchmean"
            ) * (self.temperature ** 2)
            loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        else:
        #验证集
            loss = F.cross_entropy(student_logits, labels)

        return (loss, outputs) if return_outputs else loss

# collate_fn
def custom_collate(features):
    batch = default_collate(features)
    if "teacher_logits" in features[0]:
        batch["teacher_logits"] = torch.stack([f["teacher_logits"] for f in features])
    return batch

training_args = TrainingArguments(
    output_dir="./distill_output",
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    learning_rate=2e-5,
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=20,
    fp16=True,
    report_to=None,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# 这一步return_outputs=True，训练通常为False（减小占用显存容量）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    alpha=0.5,
    temperature=2.0,
    data_collator=custom_collate,
)

trainer.train()

trainer.save_model("./imdb_student_bert_distilled")
tokenizer.save_pretrained("./imdb_student_bert_distilled")
