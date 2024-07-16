# Log in to your W&B account
import wandb
wandb.login()

import wandb
wandb.init(project='multilingual_e5_large_fit_wikipedia_datasets')

import pickle
from datasets import load_dataset

# データセットの読み込み（ストリーミング）
lang = "simple"
batch_size = 1000  # 一度に処理するバッチサイズ
docs_stream_random = load_dataset(f"Cohere/wikipedia-2023-11-embed-multilingual-v3", lang, split="train", streaming=True)

# データセットのシャッフル
shuffled_dataset = docs_stream_random.shuffle(seed=42)

all_docs = []
unique_titles = set()

for doc in shuffled_dataset:
    title = doc['title']
    if title not in unique_titles and doc['_id'].endswith('_0'):
        unique_titles.add(title)
        all_docs.append(doc)
    if len(all_docs) % batch_size == 0:
        print(f"Processed {len(all_docs)} documents...")

# all_docsとunique_titlesを保存する
with open('all_docs.pkl', 'wb') as f:
    pickle.dump(all_docs, f)

with open('unique_titles.pkl', 'wb') as f:
    pickle.dump(unique_titles, f)

print("Data saved successfully.")

#データをロードする場合に実行する
import pickle

# all_docsとunique_titlesをロードする
with open('all_docs.pkl', 'rb') as f:
    all_docs = pickle.load(f)

with open('unique_titles.pkl', 'rb') as f:
    unique_titles = pickle.load(f)

print("Data loaded successfully.")
print(f"Number of documents: {len(all_docs)}")
print(f"Number of unique titles: {len(unique_titles)}")

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')

class WikipediaDataset(Dataset):
    def __init__(self, docs, tokenizer):
        self.docs = docs
        self.tokenizer = tokenizer
        self.tokenized_data = self.tokenize_docs(docs)

    def tokenize_docs(self, docs):
        titles = [doc['title'] for doc in docs]
        texts = [doc['text'] for doc in docs]
        inputs = self.tokenizer(titles, texts, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
        return inputs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        item = self.docs[idx]
        emb = item['emb']
        inputs = {key: val[idx] for key, val in self.tokenized_data.items()}
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'emb': torch.tensor(emb)
        }

# データセットの作成
split_index = int(len(all_docs) * 0.9)  # 90:10の比率で分割
train_docs = all_docs[:split_index]
eval_docs = all_docs[split_index:]

train_dataset = WikipediaDataset(train_docs, tokenizer)
eval_dataset = WikipediaDataset(eval_docs, tokenizer)

# データローダーの作成
train_dataloader = DataLoader(train_dataset, batch_size=8)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

class MultilingualE5Regressor(pl.LightningModule):
    def __init__(self):
        super(MultilingualE5Regressor, self).__init__()
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
        self.regressor = torch.nn.Linear(self.model.config.hidden_size, 1024)  # embの次元に合わせる

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]トークンの出力
        emb_output = self.regressor(cls_output)
        return emb_output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        emb = batch['emb']

        outputs = self(input_ids, attention_mask)
        loss = torch.nn.functional.mse_loss(outputs, emb)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        emb = batch['emb']

        outputs = self(input_ids, attention_mask)
        loss = torch.nn.functional.mse_loss(outputs, emb)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

# W&Bのロガーを設定
wandb_logger = WandbLogger(project='multilingual_e5_large_fit_wikipedia_datasets')

# モデルの初期化
model = MultilingualE5Regressor()

# コールバックの設定
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='my_model/',
    filename='multilingual-e5-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

# トレーナーの設定
trainer = Trainer(
    max_epochs=1,
    devices=1,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    callbacks=[checkpoint_callback],
    logger=wandb_logger  # W&Bロガーを追加
)

# 学習の実行
trainer.fit(model, train_dataloader, val_dataloaders=eval_dataloader)
