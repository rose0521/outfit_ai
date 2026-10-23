
from pathlib import Path
import json
import pyarrow.parquet as pq

PROJECT = Path(r"E:\code\outfit_ai")
DATA_DIR = PROJECT / "data" / "raw" / "polyvore" / "data" / "disjoint"

PARQUET_PATH = DATA_DIR / "train.parquet"
LABEL_PATH = (
    PROJECT
    / "data"
    / "raw"
    / "polyvore"
    / "disjoint"
    / "compatibility_train.txt"
)

METADATA_PATH = (
    PROJECT
    / "data"
    / "raw"
    / "polyvore"
    / "polyvore_item_metadata.json"
)

print("=" * 70)
print("1. 检查 Parquet 单品 ID 与图片")
print("=" * 70)

parquet_file = pq.ParquetFile(PARQUET_PATH)

# 只读取需要的字段，避免加载图片二进制内容
table = pq.read_table(
    PARQUET_PATH,
    columns=["item_id", "image"]
)

item_ids = table.column("item_id").to_pylist()
images = table.column("image").to_pylist()

print("单品总记录数：", len(item_ids))
print("唯一 item_id 数量：", len(set(item_ids)))
print("重复 ID 记录数：", len(item_ids) - len(set(item_ids)))

valid_bytes = 0
empty_bytes = 0
nonempty_paths = 0

for image in images:
    if not image:
        empty_bytes += 1
        continue

    content = image.get("bytes")
    path = image.get("path")

    if content:
        valid_bytes += 1
    else:
        empty_bytes += 1

    if path:
        nonempty_paths += 1

print("包含图片二进制的记录数：", valid_bytes)
print("图片二进制为空的记录数：", empty_bytes)
print("包含非空图片路径的记录数：", nonempty_paths)

print("\n前 10 个 item_id：")
print(item_ids[:10])

# 保存 ID 集合，供后续映射和特征提取使用
id_set = set(item_ids)

print("\n" + "=" * 70)
print("2. 检查兼容性标签文件中的单品 ID")
print("=" * 70)

label_ids = set()
label_rows = 0
label_examples = []

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) < 2:
            continue

        label_rows += 1

        # 第一列是标签，后续列是单品 ID
        ids = parts[1:]
        label_ids.update(ids)

        if len(label_examples) < 10:
            label_examples.append(ids)

print("兼容性标签行数：", label_rows)
print("标签文件中的唯一单品 ID 数量：", len(label_ids))
print("标签文件示例：", label_examples[:3])

# 检查两种 ID 是否可以直接对应
matched = label_ids & id_set
unmatched = label_ids - id_set

print("\n直接匹配成功的 ID 数量：", len(matched))
print("未直接匹配的 ID 数量：", len(unmatched))

if label_ids:
    print(
        "直接匹配比例：",
        f"{len(matched) / len(label_ids):.2%}"
    )

print("未匹配 ID 示例：", list(unmatched)[:20])

print("\n" + "=" * 70)
print("3. 检查 metadata 文件结构")
print("=" * 70)

if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("metadata 顶层类型：", type(metadata).__name__)

    if isinstance(metadata, dict):
        print("metadata 键数量：", len(metadata))
        print("metadata 键示例：", list(metadata.keys())[:10])

        first_key = next(iter(metadata), None)
        if first_key is not None:
            print("\n第一条 metadata 示例：")
            print("key:", first_key)
            print("value:", repr(metadata[first_key])[:1000])

    # 检查 metadata 的键是否与 Parquet ID 直接匹配
    if isinstance(metadata, dict):
        metadata_keys = set(metadata.keys())
        print(
            "\nmetadata 键与 Parquet ID 的交集数量：",
            len(metadata_keys & id_set)
        )
        print(
            "metadata 键与标签 ID 的交集数量：",
            len(metadata_keys & label_ids)
        )
else:
    print("未找到 metadata 文件：", METADATA_PATH)

print("\n检查结束。")
