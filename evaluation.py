import torch
import torch.nn as nn
import torchaudio
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import warnings
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# 你原来的MultimodalTransformer定义
class MultimodalTransformer(nn.Module):
    def __init__(self,
                 audio_feat_dim=128,
                 text_hidden=768,
                 image_feat_dim=2048,
                 n_heads=8,
                 dim_feedforward=512,
                 num_classes=4):
        super().__init__()
        self.audio_proj = nn.Linear(audio_feat_dim, text_hidden)
        self.image_proj = nn.Linear(image_feat_dim, text_hidden)
        self.text_proj = nn.Linear(text_hidden, text_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_hidden, nhead=n_heads,
            dim_feedforward=dim_feedforward
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls_head = nn.Linear(text_hidden, num_classes)

    def forward(self, audio_feat, text_feat, image_feat, text_mask=None):
        B = audio_feat.size(0)
        a = self.audio_proj(audio_feat)
        i = self.image_proj(image_feat)
        t = self.text_proj(text_feat)
        x = torch.cat([a, t, i], dim=1)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        cls = x[:, 0, :]
        logits = self.cls_head(cls)
        return logits

# 自定义collate
def custom_collate_fn(batch):
    audios, texts, images, labels = zip(*batch)
    max_audio_len = max([a.shape[0] for a in audios])
    padded_audios = []
    for a in audios:
        pad_len = max_audio_len - a.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, a.shape[1])
            a = torch.cat([a, padding], dim=0)
        padded_audios.append(a)
    audios = torch.stack(padded_audios, dim=0)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)
    return audios, texts, images, labels

# Dataset定义
class CAERMultimodalDataset(Dataset):
    def __init__(self, audio_root, image_root, text_root, label_map, tokenizer, transform=None):
        super().__init__()
        self.audio_root = audio_root
        self.image_root = image_root
        self.text_root = text_root
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.transform = transform

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
        audio_path = os.path.join(self.audio_root, emotion_class, basename + ".wav")
        image_path = os.path.join(self.image_root, emotion_class, basename + ".jpg")
        text_path = os.path.join(self.text_root, emotion_class, basename + ".txt")
        audio_feat = self.load_audio(audio_path)
        img_feat = self.load_image(image_path)
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
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=128)
        mel = mel_transform(waveform)[0]
        return mel.transpose(0, 1)

    def load_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_text(self, text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            return f.read().strip()

# 开始推理
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 路径
    audio_root = "E:/New_project/MiniGPT-4/extract_data/CAER_validation_audio"
    image_root = "E:/New_project/MiniGPT-4/extract_data/CAER_validation_image"
    text_root  = "E:/New_project/MiniGPT-4/extract_data/CAER_validation_LLM_text"

    label_map = {
        "Anger": 0,
        "Neutral": 1,
        "Happy": 2,
        "Sad": 3
    }
    idx2label = {v:k for k,v in label_map.items()}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="E:/New_project/MiniGPT-4/checkpoints/bert")

    test_dataset = CAERMultimodalDataset(audio_root, image_root, text_root, label_map, tokenizer, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    # 加载模型
    model = MultimodalTransformer()
    model.load_state_dict(torch.load("E:/New_project/MiniGPT-4/checkpoints/finetuned_model_train.pth", map_location=device))
    model = model.to(device).eval()

    text_encoder = BertModel.from_pretrained('bert-base-uncased', cache_dir="E:/New_project/MiniGPT-4/checkpoints/bert").to(device).eval()
    vision_encoder = models.resnet50(pretrained=True)
    vision_encoder.fc = nn.Identity()
    vision_encoder = vision_encoder.to(device).eval()

    # 开始验证
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio_feats, texts, img_feats, labels in tqdm(test_loader, desc="Validating"):
            audio_feats = audio_feats.to(device)
            img_feats = img_feats.to(device)
            labels = labels.to(device)

            input_ids, attn_mask = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").input_ids, tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").attention_mask
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

            text_feats = text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            img_feats = vision_encoder(img_feats).unsqueeze(1)

            logits = model(audio_feats, text_feats, img_feats, attn_mask)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx2label[i] for i in range(4)])
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix (Acc={acc:.2f})")
    plt.savefig("E:/New_project/MiniGPT-4/checkpoints/confusion_matrix.png")
    plt.show()
