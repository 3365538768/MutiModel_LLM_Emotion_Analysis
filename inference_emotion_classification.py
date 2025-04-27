import torch
import torch.nn as nn
import torchaudio
from transformers import BertTokenizer, BertModel
from PIL import Image
from torchvision import transforms, models


def preprocess_audio(audio_path):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,  # 采样率（你的音频如果不是16k，下面会重采样）
        n_fft=400,  # 窗口大小，常用25ms对应400点（如果采样率是16kHz）
        hop_length=160,  # 帧移，常用10ms对应160点
        n_mels=128  # 你希望提取多少个Mel频带
    )

    # 2. 加载音频

    waveform, sample_rate = torchaudio.load(audio_path)  # waveform.shape: (channels, time_samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # 3. 如果采样率不同，重采样
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    # 4. 生成梅尔频谱图
    mel_spectrogram = mel_transform(waveform)  # shape: (channels, n_mels, time_steps)

    # 5. （可选）如果是单声道，取第一个声道
    mel_feature = mel_spectrogram[0]  # shape: (n_mels, time_steps)


    return mel_feature

def preprocess_text(text, tokenizer, max_length=128):
    tokens = tokenizer(
        text, padding='max_length', truncation=True,
        max_length=max_length, return_tensors='pt'
    )
    return tokens.input_ids, tokens.attention_mask  # both (1, seq_len)


def preprocess_image(image_path, image_size=224):
    img = Image.open(image_path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf(img).unsqueeze(0)


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


def inference(audio_path, text, image_path,
              model_ckpt, device='cuda'):
    # 载入分词器和文本编码器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              cache_dir="E:/New_project/MiniGPT-4/checkpoints/bert")
    text_encoder = BertModel.from_pretrained('bert-base-uncased',
                                             cache_dir="E:/New_project/MiniGPT-4/checkpoints/bert",
                                             device_map="cuda")
    text_encoder.to(device).eval()

    # 载入图像特征提取器（ResNet50 去掉头）
    vision = models.resnet50(pretrained=True)
    vision.fc = nn.Identity()
    vision.to(device).eval()


    model = MultimodalTransformer()
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device).eval()

    mel_db = preprocess_audio(audio_path).to(device)
    mel_feats = mel_db.unsqueeze(0).transpose(1, 2)
    # print(mel_feats.size())
    #torch.Size([1, Times=288, n_mel=64])

    input_ids, attn_mask = preprocess_text(text, tokenizer)
    input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
    with torch.no_grad():
        txt_out = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    txt_feats = txt_out.last_hidden_state
    # print(txt_feats.size())
    # (1, sequence_length=128, hidden_size=768)

    img = preprocess_image(image_path).to(device)  # (1,3,224,224)
    with torch.no_grad():
        img_feats = vision(img)  # (1, image_feat_dim)
    # 拓展到 token 序列 (这里简单把全图表示当成 1 个 token)
    img_feats = img_feats.unsqueeze(1)
    # print(img_feats.size())
    # (1,1, image_feat_dim)


    logits = model(mel_feats, txt_feats, img_feats, attn_mask)
    probs = torch.softmax(logits, dim=-1)[0]  # (num_classes,)

    # 3) 输出
    labels = ["Anger","Disgust","Happy","Sad"]  # 根据你的实际类别顺序修改
    idx = torch.argmax(probs).item()
    return labels[idx], probs.cpu().tolist()


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--audio', type=str, required=True)
    # parser.add_argument('--text', type=str, required=True)
    # parser.add_argument('--image', type=str, required=True)
    # parser.add_argument('--ckpt', type=str, required=True)
    # parser.add_argument('--device', type=str, default='cuda')
    # args = parser.parse_args()

    # label, scores = inference(
    #     args.audio, args.text, args.image,
    #     args.ckpt, device=args.device
    # )
    model = MultimodalTransformer()
    ckpt_path = "E:/New_project/MiniGPT-4/checkpoints/random_init.ckpt"
    torch.save(model.state_dict(), ckpt_path)
    audio="E:/New_project/MiniGPT-4/test_data/extracted_audio.wav"
    image="E:/New_project/MiniGPT-4/test_data/middle_frame.jpg"
    device='cuda'
    # 正确写法：
    with open("E:/New_project/MiniGPT-4/test_data/final_llm.txt", 'r') as f:
        text = f.read()

    label, scores = inference(
        audio, text, image,
        ckpt_path, device=device
    )
    print(f'Predicted emotion: {label}')
    print('Scores:', {l: round(s, 4) for l, s in zip(["Anger","Disgust","Happy","Sad"], scores)})
