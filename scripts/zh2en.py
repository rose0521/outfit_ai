# -*- coding: utf-8 -*-
"""
================================================================================
中文查询 -> 英文翻译（服装检索用，确定性词典）  zh2en.py
================================================================================

为什么需要
----------
商品库图片特征用 openai/clip-vit-base-patch32 提取，其文本编码器只对英文敏感。
中文查询（如"白色通勤衬衫"）直接过 CLIP 会得到几乎无区分度的相似度（实测 0.22~0.23，
英文 "white shirt" 可达 0.28+ 且结果高度相关）。因此检索前先把中文查询翻译成英文。

方案
----
纯离线词典（最长匹配分词 + 逐词映射），零外部依赖、零 API 调用，确定性可复现。
覆盖颜色、品类、版型、材质、风格、场合等 demo 常用词。词典覆盖不到的字会被丢弃
（CLIP 英文 tokenizer 对汉字也基本是无信息 token），英文/数字原样保留。

用法
----
    from zh2en import translate_zh2en
    print(translate_zh2en("白色通勤衬衫"))   # -> "white commuter shirt"
    print(translate_zh2en("denim jacket"))    # -> 原样返回（已英文）
================================================================================
"""

import re

# 最长匹配优先：键按长度降序排列时，先命中长词（"牛仔裤" 优先于 "牛仔"+"裤"）。
GLOSSARY = {
    # ---- 颜色 ----
    "白色": "white", "黑色": "black", "红色": "red", "蓝色": "blue",
    "灰色": "grey", "米色": "beige", "米白": "off-white", "粉色": "pink",
    "黄色": "yellow", "绿色": "green", "紫色": "purple", "棕色": "brown",
    "咖啡色": "brown", "咖色": "brown", "卡其色": "khaki", "卡其": "khaki",
    "橙色": "orange", "酒红": "burgundy", "藏青": "navy", "藏蓝": "navy",
    "天蓝": "sky blue", "军绿": "olive green", "浅蓝": "light blue",
    "深蓝": "dark blue", "牛仔蓝": "denim blue", "彩色": "colorful",
    "多色": "multicolor", "印花": "print", "条纹": "stripe", "格纹": "plaid",
    "格子": "plaid", "波点": "polka dot", "纯色": "solid", "撞色": "color block",
    "渐变色": "gradient", "白": "white", "黑": "black", "红": "red",
    "蓝": "blue", "灰": "grey", "粉": "pink", "黄": "yellow", "绿": "green",
    "紫": "purple", "棕": "brown", "咖": "brown", "橙": "orange",
    # ---- 品类 ----
    "连衣裙": "dress", "连体装": "jumpsuit", "半身裙": "skirt",
    "百褶裙": "pleated skirt", "A字裙": "A-line skirt", "裙子": "skirt",
    "裙": "skirt", "衬衫": "shirt", "上衣": "top", "牛仔裤": "jeans",
    "阔腿裤": "wide-leg pants", "铅笔裤": "pencil pants", "紧身裤": "leggings",
    "打底裤": "leggings", "运动裤": "sweatpants", "短裤": "shorts",
    "裤子": "pants", "长裤": "pants", "裤": "pants", "羽绒服": "down jacket",
    "风衣": "trench coat", "大衣": "coat", "外套": "jacket", "夹克": "jacket",
    "开衫": "cardigan", "针织衫": "knit sweater", "毛衣": "sweater",
    "针织": "knit", "连帽卫衣": "hoodie", "卫衣": "sweatshirt", "T恤": "t-shirt",
    "背心": "tank top", "马甲": "vest", "西装外套": "blazer", "西装": "blazer",
    "高跟鞋": "high heels", "运动鞋": "sneakers", "帆布鞋": "sneakers",
    "凉鞋": "sandals", "靴子": "boots", "靴": "boots", "鞋": "shoes",
    "双肩包": "backpack", "手提包": "handbag", "包包": "bag", "包": "bag",
    "太阳镜": "sunglasses", "墨镜": "sunglasses", "项链": "necklace",
    "耳环": "earrings", "腰带": "belt", "手套": "gloves", "袜子": "socks",
    "泳装": "swimsuit", "泳衣": "swimsuit", "帽子": "hat", "围巾": "scarf",
    # ---- 版型 / 细节 ----
    "通勤": "commuter", "休闲": "casual", "运动": "sporty", "宽松": "loose",
    "oversize": "oversized", "修身": "slim fit", "高腰": "high waist",
    "中腰": "mid waist", "低腰": "low waist", "长袖": "long sleeve",
    "短袖": "short sleeve", "无袖": "sleeveless", "V领": "v-neck",
    "圆领": "crew neck", "高领": "turtleneck", "一字领": "off-shoulder",
    "露肩": "off-shoulder", "收腰": "waisted", "伞摆": "flared",
    "喇叭": "flared", "A字": "A-line", "连帽": "hooded", "拉链": "zipper",
    "纽扣": "button", "扣子": "button", "开襟": "open front",
    "蝙蝠袖": "batwing sleeve", "荷叶边": "ruffle", "蕾丝": "lace",
    "雪纺": "chiffon", "真丝": "silk", "丝绸": "silk", "棉": "cotton",
    "羊毛": "wool", "兔毛": "rabbit fur", "牛仔": "denim", "灯芯绒": "corduroy",
    "皮革": "leather", "皮": "leather", "弹力": "stretch", "加绒": "fleece lined",
    "保暖": "warm", "轻薄": "lightweight", "透气": "breathable",
    "显瘦": "slimming", "厚": "thick", "薄": "thin",
    "长款": "long", "短款": "cropped", "中长款": "midi", "大码": "plus size",
    # ---- 场合 / 人群 ----
    "派对": "party", "婚礼": "wedding", "宴会": "evening", "晚宴": "evening",
    "上班": "work", "日常": "daily", "度假": "vacation",
    "女士": "women's", "女": "women's", "男": "men's",
}

_CJK = re.compile(r"[一-鿿]")
# 键按长度降序，保证最长匹配优先
_KEYS = sorted(GLOSSARY.keys(), key=len, reverse=True)


def has_chinese(text):
    return bool(_CJK.search(text))


def translate_zh2en(text):
    """把中文查询翻译成英文短语；无中文时原样返回。"""
    text = (text or "").strip()
    if not text:
        return text
    if not has_chinese(text):
        return text
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        # 英文/数字/空格/常见符号原样保留
        if (ch.isascii() and (ch.isalnum() or ch in " -/&")) or ch.isspace():
            out.append(ch)
            i += 1
            continue
        matched = False
        for k in _KEYS:
            if text.startswith(k, i):
                out.append(" " + GLOSSARY[k] + " ")
                i += len(k)
                matched = True
                break
        if not matched:
            i += 1  # 未收录的字，丢弃
    s = "".join(out)
    s = " ".join(s.split())          # 合并多余空格
    return s


if __name__ == "__main__":
    for q in ["白色通勤衬衫", "黑色连衣裙", "高腰牛仔裤", "米色针织开衫",
              "派对连衣裙", "黑色真丝衬衫", "denim jacket", "白色 长袖 衬衫"]:
        print(f"{q!r:30} -> {translate_zh2en(q)!r}")
