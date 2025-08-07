from datasets import load_dataset
import numpy as np
import evaluate
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,EarlyStoppingCallback,BertConfig

model_id = "bert-large-uncased"
dataset = load_dataset("imdb", download_mode="force_redownload")
print(dataset)
#划分验证集
train_test = dataset["train"].train_test_split(test_size=0.1,seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]
tokenizer = BertTokenizer.from_pretrained(model_id)

#数据预处理
def Function_f(example):
    return tokenizer(example["text"], truncation=True, padding=True, max_length=512)

train_dataset = train_dataset.map(Function_f, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

eval_dataset = eval_dataset.map(Function_f, batched=True)
eval_dataset = eval_dataset.rename_column("label", "labels")
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

config = BertConfig.from_pretrained(
    model_id,
    num_labels=2,
    hidden_dropout_prob=0.3,                    
    attention_probs_dropout_prob=0.3
)

#全量微调
model = BertForSequenceClassification.from_pretrained(model_id,config=config)


training_args = TrainingArguments(
    output_dir="./checkpoints_full_finetune",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    eval_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="accuracy",
    fp16=True,
    report_to=None,
    weight_decay=0.1,
    lr_scheduler_type="cosine",               
    warmup_ratio=0.1,  
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

#训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    #早停
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
model.save_pretrained("./imdb_teacher_bert_full")
tokenizer.save_pretrained("./imdb_teacher_bert_full")