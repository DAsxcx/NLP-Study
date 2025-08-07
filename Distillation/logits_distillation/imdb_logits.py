from transformers import BertForSequenceClassification, BertTokenizer,DataCollatorWithPadding
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("./imdb_teacher_bert_full").to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained("./imdb_teacher_bert_full")

# 加载数据集
dataset = load_dataset("imdb")["train"]
print(dataset)
def function_f(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
dataset = dataset.map(function_f, batched=True)
dataset = dataset.remove_columns(["text"])
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")

#dataloader中的shuffle默认为false
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
#pin_memory-->使用DMA交换数据更快（无需经过CPU）
dataloader = DataLoader(dataset, batch_size=32, collate_fn=data_collator, pin_memory=True)

# 存储 logits 和 labels
all_logits = []
all_labels = []

#推理
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # BERT 分类模型的输出

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

# 拼接保,存
all_logits = torch.cat(all_logits, dim=0)
all_labels = torch.cat(all_labels, dim=0)

torch.save(all_logits, "./imdb_all_logits.pt")
torch.save(all_labels, "./imdb_all_labels.pt")
