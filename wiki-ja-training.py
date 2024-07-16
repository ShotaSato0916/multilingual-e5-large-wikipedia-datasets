# データセットの準備

import wandb
wandb.init(project='multilingual_e5_large_fit_wikipedia_datasets')

import pickle
from datasets import load_dataset
import gc

# データセットの読み込み（ストリーミング）
lang_ja = "ja"
docs_stream_random_ja = load_dataset(f"Cohere/wikipedia-2023-11-embed-multilingual-v3", lang_ja, split="train", streaming=True)

# データセットのシャッフル
shuffled_dataset_ja = docs_stream_random_ja.shuffle(seed=42)

all_docs_ja = shuffled_dataset_ja.filter(lambda x: x["_id"].endswith('_0'))

# データセットを一時保存
from datasets import Dataset, DatasetDict

ds = Dataset.from_generator(lambda: (yield from all_docs_ja), features=all_docs_ja.features)
ds.to_parquet("docs_ja.parquet")

print("Data saved successfully.")

# データセットのロード
from datasets import load_dataset

ds = load_dataset('parquet', data_files='docs_ja.parquet', split='train')

# 訓練データと評価データに分割
all_docs_ja = ds.to_iterable_dataset()

# ストリーミングデータセットの分割
split_ratio = 0.9
train_docs_ja = all_docs_ja.take(int(len(ds) * split_ratio))
eval_docs_ja = all_docs_ja.skip(int(len(ds) * split_ratio))

print("data split completed")

# `IterableDataset`を通常の`Dataset`に変換
train_dataset_ja = Dataset.from_generator(lambda: (yield from train_docs_ja), features=train_docs_ja.features)
eval_dataset_ja = Dataset.from_generator(lambda: (yield from eval_docs_ja), features=eval_docs_ja.features)

print("IterableDataset converted to Dataset")

# これらのデータの一時保存

train_dataset_ja.to_parquet("train_dataset_ja.parquet")
eval_dataset_ja.to_parquet("eval_dataset_ja.parquet")

# データローダーの準備
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# トークナイザーの準備
tokenizer_ja = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')

# トークナイズ関数の定義
def tokenize_function(examples):
    return tokenizer_ja(examples['title'], examples['text'], truncation=True, padding='max_length', max_length=256)

# データセットのトークナイズ
train_dataset_ja = train_dataset_ja.map(tokenize_function, batched=True)
eval_dataset_ja = eval_dataset_ja.map(tokenize_function, batched=True)

# 必要なカラムのみを保持
train_dataset_ja.set_format(type='torch', columns=['input_ids', 'attention_mask', 'emb'])
eval_dataset_ja.set_format(type='torch', columns=['input_ids', 'attention_mask', 'emb'])

# データローダーの作成
train_dataloader_ja = DataLoader(train_dataset_ja, batch_size=8, num_workers=4, shuffle=True)
eval_dataloader_ja = DataLoader(eval_dataset_ja, batch_size=8, num_workers=4, shuffle=False)

print("DataLoader created")

# 学習の準備
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import pytorch_lightning as pla
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

class MultilingualE5Regressor_ja(pl.LightningModule):
    def __init__(self):
        super(MultilingualE5Regressor_ja, self).__init__()
        self.model_ja = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

    def forward(self, input_ids_ja, attention_mask_ja):
        outputs_ja = self.model_ja(input_ids=input_ids_ja, attention_mask=attention_mask_ja)
        return outputs_ja.last_hidden_state.mean(axis=1)  # [CLS]トークンの出力

    def training_step(self, batch_ja, batch_idx_ja):
        input_ids_ja = batch_ja['input_ids']
        attention_mask_ja = batch_ja['attention_mask']
        emb_ja = batch_ja['emb']

        outputs_ja = self(input_ids_ja, attention_mask_ja)
        loss_ja = torch.nn.functional.mse_loss(outputs_ja, emb_ja)
        self.log('train_loss', loss_ja)
        return loss_ja

    def validation_step(self, batch_ja, batch_idx_ja):
        input_ids_ja = batch_ja['input_ids']
        attention_mask_ja = batch_ja['attention_mask']
        emb_ja = batch_ja['emb']

        outputs_ja = self(input_ids_ja, attention_mask_ja)
        loss_ja = torch.nn.functional.mse_loss(outputs_ja, emb_ja)
        self.log('val_loss', loss_ja)
        return loss_ja

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

# W&Bのロガーを設定
wandb_logger_ja = WandbLogger(project='multilingual_e5_large_fit_wikipedia_datasets')

# モデルの初期化
model_ja = MultilingualE5Regressor_ja()

# コールバックの設定
checkpoint_callback_ja = ModelCheckpoint(
    monitor='val_loss',
    dirpath='my_model_ja/',
    filename='multilingual-e5-{epoch:02d}-{val_loss:.2f}-ja',
    save_top_k=3,
    mode='min',
)

# トレーナーの設定
trainer_ja = Trainer(
    max_epochs=1,
    devices=1,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    callbacks=[checkpoint_callback_ja],
    logger=wandb_logger_ja  # W&Bロガーを追加
)

# 学習の実行
trainer_ja.fit(model_ja, train_dataloader_ja, val_dataloaders=eval_dataloader_ja)
