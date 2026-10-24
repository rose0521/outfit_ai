# -*- coding: utf-8 -*-
"""
================================================================================
CLIP 编码工具（图文共享）  clip_utils.py
================================================================================

供新脚本复用：
  - extract_features_from_files.py（商品库图片 -> 特征）
  - build_search.py（文字 / 图片 -> 检索）
  - serve_demo.py（/api/search 的文字编码）

重要：推理路径与 extract_features.py 完全一致——
      vision_model -> pooler_output -> visual_projection
      text_model    -> pooler_output -> text_projection
显式走 projection，规避 transformers 版本差异（get_image_features() /
get_text_features() 的返回值在不同版本不稳定）。所有特征输出均为
L2 归一化的 float32，保证和已有 Polyvore 特征在同一空间。
================================================================================
"""

import io
import urllib.request

import numpy as np
import torch

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def l2_normalize(x):
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def load_clip(model_name=DEFAULT_MODEL, device="cpu"):
    """加载 CLIP 模型与 processor，冻结参数。返回 (model, processor)。"""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, processor


def encode_images_pil(model, processor, images, device="cpu"):
    """images: list[PIL.Image]。返回 L2 归一化 float32 (N, projection_dim)。"""
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = (vision_outputs.pooler_output
                        if hasattr(vision_outputs, "pooler_output")
                        else vision_outputs[1])
        feats = model.visual_projection(image_embeds)
    return l2_normalize(feats.float().cpu().numpy())


def encode_texts(model, processor, texts, device="cpu"):
    """texts: list[str]。返回 L2 归一化 float32 (N, projection_dim)。"""
    inputs = processor(text=list(texts), return_tensors="pt",
                       padding=True, truncation=True, max_length=77)
    batch = {k: v.to(device) for k, v in inputs.items()
             if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        out = model.text_model(**batch, return_dict=True)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out[1]
        feats = model.text_projection(emb)
    return l2_normalize(feats.float().cpu().numpy())


def load_image(path_or_url, timeout=15):
    """从本地路径或 http(s) URL 加载图片，返回 RGB PIL.Image。失败抛异常。"""
    from PIL import Image
    s = str(path_or_url).strip()
    if s.lower().startswith(("http://", "https://")):
        req = urllib.request.Request(s, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(s).convert("RGB")
