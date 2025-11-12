#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_train_model.py — Fine-tuning CLIP с GeM pooling и Triplet Loss

Обучает модель различать локации вашего района через metric learning

Использование:
python scripts/05_train_model.py \
    --crops-meta meta/crops_with_ocr.csv \
    --output-dir models/clip_gem \
    --epochs 50 \
    --batch-size 64
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip

# ========================= GeM Pooling =========================

class GeM(nn.Module):
    """Generalized Mean Pooling для place recognition"""
    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if learn_p else p
        self.eps = eps
        
    def forward(self, x):
        # x: [B, C, H, W] -> [B, C]
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)

class CLIPGeM(nn.Module):
    """CLIP + GeM pooling для геолокализации"""
    def __init__(self, clip_model, gem_p=3.0, freeze_layers=True):
        super().__init__()
        self.visual = clip_model.visual
        self.gem = GeM(p=gem_p, learn_p=True)
        
        # Замораживаем первые слои (оставляем последние для fine-tuning)
        if freeze_layers:
            for name, param in self.visual.named_parameters():
                # Размораживаем только последний блок трансформера
                if 'transformer.resblocks.23' not in name and \
                   'transformer.resblocks.22' not in name and \
                   'transformer.resblocks.21' not in name:
                    param.requires_grad = False
        
    def forward(self, x):
        # Extract features перед pooling
        # Для ViT нужно обойти final pooling
        x = self.visual.conv1(x)  # patch embedding
        x = x.reshape(x.shape, x.shape, -1)  # [B, C, HW]
        x = x.permute(0, 2, 1)  # [B, HW, C]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + \
                      torch.zeros(x.shape, 1, x.shape[-1], dtype=x.dtype, device=x.device), 
                      x], dim=1)  # [B, HW+1, C]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # Reshape для GeM pooling [B, N, C] -> [B, C, H, W]
        # Убираем class token
        x = x[:, 1:, :]  # [B, HW, C]
        B, HW, C = x.shape
        H = W = int(np.sqrt(HW))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # GeM pooling
        pooled = self.gem(x)  # [B, C]
        
        # L2 normalization
        return F.normalize(pooled, p=2, dim=1)

# ========================= Triplet Loss =========================

class BatchHardTripletLoss(nn.Module):
    """Batch-Hard Triplet Loss для metric learning"""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        # embeddings: [B, D], labels: [B] (pano_id)
        B = embeddings.size(0)
        
        # Pairwise distances [B, B]
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        
        loss = 0.0
        n_valid = 0
        
        for i in range(B):
            # Hardest positive: максимальная дистанция с тем же pano_id
            pos_mask = (labels == labels[i]) & (torch.arange(B, device=labels.device) != i)
            if pos_mask.sum() == 0:
                continue
            hardest_pos_dist = dist_mat[i][pos_mask].max()
            
            # Hardest negative: минимальная дистанция с другим pano_id
            neg_mask = labels != labels[i]
            if neg_mask.sum() == 0:
                continue
            hardest_neg_dist = dist_mat[i][neg_mask].min()
            
            # Triplet loss
            loss += F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
            n_valid += 1
        
        return loss / max(n_valid, 1)

# ========================= Dataset =========================

class GeolocalizationDataset(Dataset):
    """Dataset для обучения геолокализации"""
    def __init__(self, crops_df, transform, train=True):
        self.crops_df = crops_df.reset_index(drop=True)
        self.transform = transform
        self.train = train
        
        # Создаём маппинг pano_id -> label для triplet loss
        unique_panos = crops_df['pano_id'].unique()
        self.pano_to_label = {pano: idx for idx, pano in enumerate(unique_panos)}
        
        # Для каждого pano_id храним индексы кропов (для sampling)
        self.pano_to_crops = {}
        for idx, row in crops_df.iterrows():
            pano = row['pano_id']
            if pano not in self.pano_to_crops:
                self.pano_to_crops[pano] = []
            self.pano_to_crops[pano].append(idx)
        
        print(f"[i] Dataset: {len(self.crops_df)} crops, {len(unique_panos)} unique panos")
    
    def __len__(self):
        return len(self.crops_df)
    
    def __getitem__(self, idx):
        row = self.crops_df.iloc[idx]
        
        # Загрузка изображения
        try:
            img = Image.open(row['path']).convert('RGB')
        except Exception as e:
            print(f"[!] Error loading {row['path']}: {e}")
            # Fallback: black image
            img = Image.new('RGB', (640, 640), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        # Label для triplet loss
        label = self.pano_to_label[row['pano_id']]
        
        return img, label, idx

# ========================= Training =========================

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", unit="batch")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                embeddings = model(images)
                loss = criterion(embeddings, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / max(n_batches, 1)

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Валидация"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for images, labels, _ in tqdm(dataloader, desc="Validation", unit="batch"):
        images = images.to(device)
        labels = labels.to(device)
        
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)

# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning CLIP + GeM для геолокализации"
    )
    
    # Данные
    parser.add_argument("--crops-meta", required=True,
                       help="CSV с метаданными кропов")
    parser.add_argument("--output-dir", default="models/clip_gem",
                       help="Папка для сохранения модели")
    
    # Модель
    parser.add_argument("--model-name", default="ViT-L-14",
                       help="CLIP модель (default: ViT-L-14)")
    parser.add_argument("--pretrained", default="openai",
                       help="Pretrained source (default: openai)")
    parser.add_argument("--gem-p", type=float, default=3.0,
                       help="GeM pooling parameter (default: 3.0)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50,
                       help="Количество эпох (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate для CLIP (default: 1e-5)")
    parser.add_argument("--lr-gem", type=float, default=1e-3,
                       help="Learning rate для GeM (default: 1e-3)")
    parser.add_argument("--margin", type=float, default=0.3,
                       help="Triplet loss margin (default: 0.3)")
    
    # Разделение на train/val
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Доля validation данных (default: 0.1)")
    
    # Оптимизации
    parser.add_argument("--fp16", action='store_true',
                       help="Mixed precision training (RTX 5090)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="DataLoader workers (default: 4)")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[i] Device: {device}")
    
    # Загрузка данных
    print(f"\n[1/6] Загрузка метаданных: {args.crops_meta}")
    crops_df = pd.read_csv(args.crops_meta)
    
    # Проверка существования файлов
    existing_mask = crops_df['path'].apply(lambda p: Path(p).exists())
    crops_df = crops_df[existing_mask].reset_index(drop=True)
    print(f"[✓] {len(crops_df)} кропов")
    
    # Train/Val split по панорамам (не по кропам!)
    unique_panos = crops_df['pano_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_panos)
    
    n_val_panos = int(len(unique_panos) * args.val_split)
    val_panos = set(unique_panos[:n_val_panos])
    
    train_df = crops_df[~crops_df['pano_id'].isin(val_panos)]
    val_df = crops_df[crops_df['pano_id'].isin(val_panos)]
    
    print(f"[i] Train: {len(train_df)} crops ({len(train_df['pano_id'].unique())} panos)")
    print(f"[i] Val: {len(val_df)} crops ({len(val_df['pano_id'].unique())} panos)")
    
    # Загрузка CLIP модели
    print(f"\n[2/6] Загрузка CLIP: {args.model_name}")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, 
        pretrained=args.pretrained
    )
    print("[✓] CLIP загружен")
    
    # Создание модели с GeM
    print(f"\n[3/6] Создание CLIPGeM модели (gem_p={args.gem_p})")
    model = CLIPGeM(clip_model, gem_p=args.gem_p, freeze_layers=True)
    model = model.to(device)
    
    # Подсчёт параметров
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[i] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Datasets
    print(f"\n[4/6] Создание datasets")
    train_dataset = GeolocalizationDataset(train_df, preprocess, train=True)
    val_dataset = GeolocalizationDataset(val_df, preprocess, train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Loss & Optimizer
    criterion = BatchHardTripletLoss(margin=args.margin)
    
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.visual.named_parameters() if p.requires_grad], 
         'lr': args.lr},
        {'params': model.gem.parameters(), 'lr': args.lr_gem}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training loop
    print(f"\n[5/6] Начало обучения ({args.epochs} epochs)")
    print(f"      Batch size: {args.batch_size}, FP16: {args.fp16}")
    
    best_val_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                device, scaler)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'gem_p': args.gem_p,
            }, checkpoint_path)
            print(f"[✓] Best model saved: {checkpoint_path}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    
    # Final model save
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    
    print(f"\n{'='*60}")
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*60}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Models saved in: {output_dir}/")
    
    print("\n🎯 Следующий шаг:")
    print(f"   python scripts/06_build_index.py --model-path {output_dir}/best_model.pt")

if __name__ == "__main__":
    main()
