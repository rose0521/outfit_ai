
from pathlib import Path
import json

PROJECT = Path(r"E:\code\outfit_ai")
JSON_PATH = (
    PROJECT
    / "data"
    / "raw"
    / "polyvore"
    / "disjoint"
    / "train.json"
)

if not JSON_PATH.exists():
    print(f"文件不存在：{JSON_PATH}")
    raise SystemExit(1)

print("=" * 70)
print("搭配 JSON 结构检查")
print("=" * 70)
print("文件：", JSON_PATH)
print(f"文件大小：{JSON_PATH.stat().st_size / 1024**2:.2f} MB")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\n顶层类型：", type(data).__name__)

if isinstance(data, dict):
    print("顶层记录数：", len(data))
    print("顶层键示例：", list(data.keys())[:10])

    first_key = next(iter(data), None)
    if first_key is not None:
        print("\n第一条记录的键：", first_key)
        record = data[first_key]
        print("第一条记录的类型：", type(record).__name__)
        print("第一条记录内容：")
        print(json.dumps(
            record,
            ensure_ascii=False,
            indent=2
        )[:5000])

elif isinstance(data, list):
    print("记录数：", len(data))

    if data:
        print("\n第一条记录的类型：", type(data[0]).__name__)
        print("第一条记录内容：")
        print(json.dumps(
            data[0],
            ensure_ascii=False,
            indent=2
        )[:5000])

else:
    print("请进一步检查该 JSON 的实际格式。")
