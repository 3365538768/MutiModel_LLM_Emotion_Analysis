from torch.utils.data import Dataset
import os
from pathlib import Path
import torchaudio
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import  models

from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
class MultimodalTransformer(nn.Module):
    def __init__(self,
                 audio_feat_dim=128,
                 text_hidden=768,
                 image_feat_dim=2048,
                 n_heads=8,
                 dim_feedforward=512,
                 num_classes=4):
        super().__init__()
        # 线性投影：把各模态特征映射到同一维度
        self.audio_proj = nn.Linear(audio_feat_dim, text_hidden)
        self.image_proj = nn.Linear(image_feat_dim, text_hidden)
        self.text_proj = nn.Linear(text_hidden, text_hidden)
        # TransformerEncoder 作为融合模块
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_hidden, nhead=n_heads,
            dim_feedforward=dim_feedforward
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # 分类头
        self.cls_head = nn.Linear(text_hidden, num_classes)

    def forward(self, audio_feat, text_feat, image_feat, text_mask=None):
        """
        audio_feat: (B, T_a, 64)
        text_feat:  (B, T_t, 768)
        image_feat: (B, 1, 2048)
        text_mask:  (B, T_t) -> 可选的mask，暂时没用
        """
        B = audio_feat.size(0)

        # 投影到统一hidden size
        a = self.audio_proj(audio_feat)    # (B, T_a, fusion_hidden=768)
        i = self.image_proj(image_feat)    # (B, 1, 768)
        t=self.text_proj(text_feat)
        # 三模态特征拼接
        x = torch.cat([a, t, i], dim=1)     # (B, T_all, 768)

        # 注意力mask（可选）
        mask = None

        # Transformer处理
        x = x.transpose(0, 1)               # (T_all, B, 768)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)               # (B, T_all, 768)

        # 取第一个token作为代表（可以换成池化）
        cls = x[:, 0, :]                    # (B, 768)
        logits = self.cls_head(cls)         # (B, num_classes)

        return logits
class CAERMultimodalDataset(Dataset):
    def __init__(self, audio_root, image_root, text_root, label_map, tokenizer, max_length=128, transform=None):
        super().__init__()
        self.audio_root = audio_root
        self.image_root = image_root
        self.text_root = text_root
        self.label_map = label_map  # e.g., {"Anger": 0, "Disgust": 1, ...}

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

        # 收集所有样本路径 (用ASR文本那边来决定样本集合)
        self.samples = []
        for emotion_class in os.listdir(text_root):
            emotion_path = os.path.join(text_root, emotion_class)
            if not os.path.isdir(emotion_path):
                continue
            for file in os.listdir(emotion_path):
                if file.endswith(".txt"):
                    basename = file.replace(".txt", "")
                    self.samples.append((emotion_class, basename))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emotion_class, basename = self.samples[idx]

        # 路径
        audio_path = os.path.join(self.audio_root, emotion_class, basename + ".wav")
        image_path = os.path.join(self.image_root, emotion_class, basename + ".jpg")
        text_path = os.path.join(self.text_root, emotion_class, basename + ".txt")

        # 加载
        audio_feat = self.load_audio(audio_path)  # (n_mels, time_steps)
        img_feat = self.load_image(image_path)  # (3, 224, 224)
        text = self.load_text(text_path)

        label = self.label_map[emotion_class]

        return audio_feat, text, img_feat, label

    def load_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=128
        )
        mel = mel_transform(waveform)[0]  # (n_mels, time_steps)

        return mel.transpose(0, 1)  # 转成 (time_steps, n_mels)

    def load_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_text(self, text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            return f.read().strip()

# 准备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据路径
audio_root = "E:/New_project/MiniGPT-4/extract_data/CAER_validation_audio"
image_root = "E:/New_project/MiniGPT-4/extract_data/CAER_validation_image"
text_root  = "E:/New_project/MiniGPT-4/extract_data/CAER_validation_LLM-text"

label_map = {
    "Anger": 0,
    "Disgust": 1,
    "Happy": 2,
    "Sad": 3
}

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="E:/New_project/MiniGPT-4/checkpoints/bert")

# Dataset & DataLoader
dataset = CAERMultimodalDataset(audio_root, image_root, text_root, label_map, tokenizer, transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Text encoder and vision encoder
text_encoder = BertModel.from_pretrained('bert-base-uncased', cache_dir="E:/New_project/MiniGPT-4/checkpoints/bert").to(device).eval()
vision_encoder = models.resnet50(pretrained=True)
vision_encoder.fc = nn.Identity()
vision_encoder = vision_encoder.to(device).eval()

# Multimodal Transformer
model = MultimodalTransformer()
model.to(device)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

# 训练
num_epochs = 5
ckpt_save_path = "E:/New_project/MiniGPT-4/checkpoints/finetuned_model.pth"

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0

    pbar = tqdm(train_loader)
    for audio_feats, texts, img_feats, labels in pbar:
        audio_feats = audio_feats.to(device)
        img_feats = img_feats.to(device)
        labels = labels.to(device)

        # 文本处理
        input_ids, attn_mask = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").input_ids, tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").attention_mask
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

        with torch.no_grad():
            text_feats = text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state

        with torch.no_grad():
            img_feats = vision_encoder(img_feats).unsqueeze(1)  # (B,1,2048)

        logits = model(audio_feats, text_feats, img_feats, attn_mask)

        loss = criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += acc.item()

        pbar.set_description(f"Epoch {epoch+1} | Loss: {total_loss/len(pbar):.4f} | Acc: {total_correct/len(pbar):.4f}")

    # 每个epoch保存一次
    torch.save(model.state_dict(), ckpt_save_path)

print("训练完成")