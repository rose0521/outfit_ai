
from pathlib import Path
import pyarrow.parquet as pq

PROJECT = Path(r"E:\code\outfit_ai")
DATA_DIR = PROJECT / "data" / "raw" / "polyvore"

# 先检查 disjoint 训练集，不一次性加载整个文件
parquet_path = DATA_DIR / "data" / "disjoint" / "train.parquet"

if not parquet_path.exists():
    print(f"文件不存在：{parquet_path}")
    raise SystemExit(1)

print("=" * 70)
print("Parquet 文件结构检查")
print("=" * 70)
print("文件：", parquet_path)
print(f"文件大小：{parquet_path.stat().st_size / 1024**2:.2f} MB")

try:
    parquet_file = pq.ParquetFile(parquet_path)

    print("\n总行数：", parquet_file.metadata.num_rows)
    print("Row groups：", parquet_file.metadata.num_row_groups)

    print("\n[1] 字段名称和类型")
    schema = parquet_file.schema_arrow

    for field in schema:
        print(f"- {field.name}: {field.type}")

    print("\n[2] 只读取前两条记录，检查真实字段结构")
    batch_iter = parquet_file.iter_batches(batch_size=2)

    try:
        batch = next(batch_iter)
        rows = batch.to_pylist()

        for index, row in enumerate(rows):
            print(f"\n记录 {index + 1}:")
            for key, value in row.items():
                # 避免打印过长的图片字节或嵌套数据
                value_repr = repr(value)
                if len(value_repr) > 500:
                    value_repr = value_repr[:500] + "...（已截断）"
                print(f"  {key}: {value_repr}")

    except StopIteration:
        print("文件中没有记录。")

except Exception as exc:
    print(f"检查失败：{type(exc).__name__}: {exc}")
    raise
