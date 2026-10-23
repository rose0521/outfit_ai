# -*- coding: utf-8 -*-
"""
================================================================================
数据集检查脚本：Polyvore Outfits（Hugging Face: mvasil/polyvore-outfits）
================================================================================

用途
----
在开始写训练代码之前，先摸清本地数据集的真实目录结构、文件类型、字段与存储格式。
本脚本**只做检查，不训练模型，不修改任何原始数据文件**。

它会：
  1. 打印数据目录的目录树与文件结构；
  2. 识别文件类型：Parquet / JSON / CSV / 图片 / 文本 / pickle 等；
  3. 对 Parquet 文件：只读元数据(schema)与少量行，打印列名、类型、前 3 行、总行数，
     并专门探测“图片以字节内嵌”这类嵌套结构（不做“图片=路径字符串”的假设）；
  4. 对 JSON 文件：识别顶层是 list 还是 dict，展示真实字段与结构；
  5. 对 CSV 文件：展示是否有表头、列数与样例；
  6. 对图片：统计散落的图片文件数量并抽样打开验证；同时对 Parquet 内嵌的图片字节
     做抽样解码验证；
  7. 最后根据“实际发现”的字段，输出字段语义映射（单品ID/图片/类别/穿搭ID/穿搭组合/
     训练划分/兼容性标签/FITB 标签），并对缺失字段明确说明（不虚构）。

运行方式
--------
  # 默认检查 E:\\code\\outfit_ai\\data\\raw\\polyvore
  python scripts/inspect_dataset.py

  # 指定其它目录
  python scripts/inspect_dataset.py --root "E:\\code\\outfit_ai\\data\\raw\\polyvore"

  # 只打印到控制台，不写报告文件
  python scripts/inspect_dataset.py --no-report

  # 指定报告文件位置（默认写到 scripts/dataset_inspection_report.txt）
  python scripts/inspect_dataset.py --report "E:\\code\\outfit_ai\\scripts\\report.txt"

依赖
----
  必选：Python 3.8+
  推荐：pandas、pyarrow、Pillow
  （若缺少某个可选库，脚本会跳过对应部分并在控制台给出提示，不会中断）

安全说明
--------
  * 脚本不会反序列化 .p / .pickle 文件（pickle 属于可执行代码，来源不明时不应加载），
    仅报告其存在与文本预览。
  * 脚本对数据目录只读，不写入数据目录；报告文件默认写到 scripts/ 目录下。

作者/时间：见文件生成日期。仅用于学术研究，遵循数据集 CC BY 4.0 许可。
================================================================================
"""

import argparse
import io
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------------------
# 0. 控制台编码处理（Windows 下防止中文/特殊字符输出报错）
# ------------------------------------------------------------------------------
try:
    # Python 3.7+ 支持 reconfigure；errors='replace' 保证即使控制台编码不匹配也不崩溃
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 某些环境没有 reconfigure，忽略即可

# ------------------------------------------------------------------------------
# 可选依赖导入（缺失时降级处理，不中断）
# ------------------------------------------------------------------------------
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

try:
    import pyarrow
    import pyarrow.parquet as pq
    _PYARROW_VERSION = pyarrow.__version__
    _HAS_PYARROW = True
except Exception:
    _PYARROW_VERSION = None
    _HAS_PYARROW = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# ------------------------------------------------------------------------------
# 常量定义
# ------------------------------------------------------------------------------
# 视为“图片文件”的扩展名
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
# 视为“Parquet 文件”的扩展名
PARQUET_EXTS = {".parquet", ".pq"}
# 视为“JSON 文件”的扩展名
JSON_EXTS = {".json"}
# 视为“CSV 文件”的扩展名
CSV_EXTS = {".csv", ".tsv"}
# 视为“pickle 文件”的扩展名（只识别，不反序列化）
PICKLE_EXTS = {".p", ".pkl", ".pickle"}
# 视为“文本文件”的扩展名
TEXT_EXTS = {".txt", ".md", ".gitattributes", ".gitignore"}

# 项目根目录 = 本脚本所在目录的上一级（scripts/ 的父目录）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# 默认数据目录
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw" / "polyvore"


# ------------------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------------------
def section(title):
    """打印一个带分隔线的章节标题。"""
    line = "=" * 80
    print()
    print(line)
    print(title)
    print(line)


def truncate(s, n=160):
    """截断过长字符串，避免刷屏；返回字符串（把非字符串安全转换）。"""
    if s is None:
        return "None"
    if isinstance(s, bytes):
        return f"<bytes, 长度={len(s)}>"
    if not isinstance(s, str):
        s = str(s)
    if len(s) > n:
        return s[:n] + f"...(共{len(s)}字符, 已截断)"
    return s


def detect_image_format(prefix: bytes) -> str:
    """根据图片字节头部的魔数判断图片格式。"""
    if prefix[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if prefix[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if prefix[:4] == b"GIF8":
        return "GIF"
    if prefix[:2] == b"BM":
        return "BMP"
    if prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
        return "WEBP"
    return "未知(魔数无法识别)"


def describe_value(v, depth=0, max_depth=3, key=None):
    """
    递归描述一个 Python 值的结构（用于展示嵌套字段、图片字节等真实结构）。
    返回一个字符串描述，不假设任何固定格式。
    """
    indent = "  " * depth
    prefix = f"{key} = " if key is not None else ""

    if depth > max_depth:
        return f"{indent}{prefix}<深度超过{max_depth}层，停止展开>"

    if isinstance(v, dict):
        n = len(v)
        head = f"{indent}{prefix}<dict, 共 {n} 个键>"
        lines = [head]
        for i, (k, vv) in enumerate(v.items()):
            if i >= 5:
                lines.append(f"{indent}  ...(其余 {n - 5} 个键省略)")
                break
            lines.append(describe_value(vv, depth + 1, max_depth, key=repr(k)))
        return "\n".join(lines)

    if isinstance(v, list):
        n = len(v)
        head = f"{indent}{prefix}<list, 共 {n} 个元素>"
        lines = [head]
        for i, vv in enumerate(v):
            if i >= 3:
                lines.append(f"{indent}  ...(其余 {n - 3} 个元素省略)")
                break
            lines.append(describe_value(vv, depth + 1, max_depth, key=f"[{i}]"))
        return "\n".join(lines)

    if isinstance(v, bytes):
        fmt = detect_image_format(v)
        return f"{indent}{prefix}<bytes, 长度={len(v)} 字节, 魔数识别={fmt}, 前16字节hex={v[:16].hex()}>"

    if isinstance(v, str):
        return f"{indent}{prefix}<str> {truncate(v)}"

    if isinstance(v, bool):
        return f"{indent}{prefix}<bool> {v}"

    if isinstance(v, int):
        return f"{indent}{prefix}<int> {v}"

    if isinstance(v, float):
        return f"{indent}{prefix}<float> {v}"

    if v is None:
        return f"{indent}{prefix}<None>"

    return f"{indent}{prefix}<{type(v).__name__}> {truncate(v)}"


def print_deps():
    """打印依赖检查结果。"""
    section("[1] 环境与依赖检查")
    print(f"Python 版本 : {sys.version.split()[0]}  ({sys.executable})")
    print(f"脚本目录    : {SCRIPT_DIR}")
    print(f"项目根目录  : {PROJECT_ROOT}")
    print(f"数据根目录  : {DEFAULT_ROOT}")
    print(f"pandas  : {'已安装 ' + pd.__version__ if _HAS_PANDAS else '未安装（跳过 Parquet/CSV 的结构化读取）'}")
    print(f"pyarrow : {'已安装 ' + _PYARROW_VERSION if _HAS_PYARROW else '未安装（无法读 Parquet，请 pip install pyarrow）'}")
    print(f"Pillow  : {'已安装' if _HAS_PIL else '未安装（无法抽样验证图片，请 pip install Pillow）'}")


def collect_files(root: Path):
    """
    遍历目录，收集所有文件（排除隐藏缓存目录 .cache / .git / .venv 等）。
    返回 (all_files, by_ext)：
      all_files: list[Path]
      by_ext:   Counter，扩展名(小写) -> 数量
    """
    skip_dirs = {".cache", ".git", ".venv", "__pycache__", ".hg", ".svn"}
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # 原地过滤掉不需要进入的目录
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            all_files.append(Path(dirpath) / fn)

    by_ext = Counter()
    for f in all_files:
        ext = f.suffix.lower()
        by_ext[ext] += 1
    return all_files, by_ext


def print_tree(root: Path):
    """打印目录树（最多若干层，避免过深）。"""
    section("[2] 目录文件结构")
    print(f"根目录: {root}")
    if not root.exists():
        print(f"!! 目录不存在: {root}")
        print("   请先下载数据集到该路径，或用 --root 指定正确路径。")
        return

    skip_dirs = {".cache", ".git", ".venv", "__pycache__", ".hg", ".svn"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        depth = len(Path(dirpath).relative_to(root).parts)
        indent = "  " * depth
        print(f"{indent}[目录] {Path(dirpath).name}/")
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            print(f"{indent}    {fn}   ({_human_size(size)})")


def _human_size(n):
    """字节数转可读大小。"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}TB"


class _Tee:
    """同时把输出写入多个流（控制台 + 内存缓冲区），用于边打印边生成报告文件。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def print_file_type_summary(by_ext):
    """打印文件类型统计。"""
    section("[3] 文件类型统计")
    if not by_ext:
        print("（未发现任何文件）")
        return
    print(f"{'扩展名':<12} {'数量':>8}   分类")
    for ext, cnt in sorted(by_ext.items(), key=lambda x: -x[1]):
        if ext in PARQUET_EXTS:
            kind = "Parquet（列式存储，可能内嵌图片字节）"
        elif ext in JSON_EXTS:
            kind = "JSON"
        elif ext in CSV_EXTS:
            kind = "CSV / 表格"
        elif ext in IMAGE_EXTS:
            kind = "图片文件"
        elif ext in PICKLE_EXTS:
            kind = "pickle（仅识别，不反序列化）"
        elif ext in TEXT_EXTS:
            kind = "文本文件"
        else:
            kind = "其它/未知"
        print(f"{ext or '(无扩展名)':<12} {cnt:>8}   {kind}")


# ------------------------------------------------------------------------------
# Parquet 检查
# ------------------------------------------------------------------------------
def inspect_parquet(files, findings):
    """检查所有 Parquet 文件：schema + 前 3 行 + 总行数 + 内嵌图片结构。"""
    section("[4] Parquet 文件检查")
    if not _HAS_PYARROW:
        print("!! 未安装 pyarrow，无法读取 Parquet。请执行: pip install pyarrow")
        return
    if not _HAS_PANDAS:
        print("!! 未安装 pandas，无法展示行样例。请执行: pip install pandas")

    findings["parquet"] = []
    for f in files:
        print("-" * 80)
        print(f"文件: {f.relative_to(findings['root']) if 'root' in findings else f}")
        try:
            pf = pq.ParquetFile(str(f))
            num_rows = pf.metadata.num_rows
            num_rg = pf.metadata.num_row_groups
            schema = pf.schema_arrow
            col_names = schema.names
            print(f"  总行数: {num_rows:,}    行组数: {num_rg}")

            print("  列名与类型（Arrow schema，含嵌套结构）:")
            for name in col_names:
                field = schema.field(name)
                print(f"    - {name}: {field.type}")

            # 只读前 3 行（不要整表读入内存，1.9GB 文件也能秒级完成）
            sample = None
            try:
                batch = next(pf.iter_batches(batch_size=3))
                sample = batch.to_pandas()
            except StopIteration:
                sample = None
            except Exception as e:
                print(f"  !! 读取前 3 行失败: {e}")

            entry = {
                "file": str(f),
                "cols": col_names,
                "num_rows": num_rows,
                "image_struct": None,
                "image_sample": [],
            }
            findings["parquet"].append(entry)

            if sample is not None and len(sample) > 0:
                print(f"  前 {len(sample)} 行示例（图片字节只显示摘要，不打印全部二进制）:")
                for i, (_, row) in enumerate(sample.iterrows()):
                    print(f"    --- 第 {i + 1} 行 ---")
                    for col in sample.columns:
                        v = row[col]
                        desc = describe_value(v, depth=1, max_depth=3, key=col)
                        print("      " + desc.replace("\n", "\n      "))

                    # 探测 image 字段是否为 struct{bytes, path}
                    if "image" in sample.columns:
                        img = row["image"]
                        if isinstance(img, dict) and "bytes" in img:
                            entry["image_struct"] = {"keys": list(img.keys()),
                                                     "path": img.get("path"),
                                                     "fmt": detect_image_format(img.get("bytes", b""))}
                            entry["image_sample"].append(img)
            else:
                print("  （空文件或无数据行）")

        except Exception as e:
            print(f"  !! 读取失败: {e}")
            continue

    # 汇总：内嵌图片情况
    print("-" * 80)
    print("Parquet 内嵌图片汇总:")
    n_img_files = 0
    n_img_rows = 0
    for e in findings["parquet"]:
        if e["image_struct"] is not None:
            n_img_files += 1
            n_img_rows += e["num_rows"]
    if n_img_files > 0:
        print(f"  {n_img_files} 个 Parquet 文件包含图片字节列(image)，内嵌图片行数合计 ≈ {n_img_rows:,} 行。")
    else:
        print("  未在 Parquet 中发现明显的图片字节列。")


# ------------------------------------------------------------------------------
# JSON 检查
# ------------------------------------------------------------------------------
def _load_json(path: Path, max_mb=300):
    """
    加载 JSON；超大文件(>max_mb MB)只返回 None 并打印提示，避免内存爆掉。
    """
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except Exception:
        size_mb = 0
    if size_mb > max_mb:
        print(f"  !! 文件过大({size_mb:.0f}MB)，跳过完整加载，仅展示前 1000 字符预览。")
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            print(truncate(fh.read(1000), 1000))
        return None, None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
        return data, size_mb
    except Exception as e:
        print(f"  !! JSON 解析失败: {e}")
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            print("     原始开头预览: " + truncate(fh.read(300), 300))
        return None, size_mb


def inspect_json(files, findings):
    """检查所有 JSON 文件：顶层类型 + 真实字段结构。"""
    section("[5] JSON 文件检查")
    findings["json"] = []
    for f in files:
        print("-" * 80)
        print(f"文件: {f.relative_to(findings['root']) if 'root' in findings else f}")
        data, size_mb = _load_json(f)
        if data is None:
            continue

        rec = {"file": str(f), "top_type": type(data).__name__}
        if isinstance(data, list):
            n = len(data)
            rec["n"] = n
            print(f"  顶层类型: list，共 {n} 个元素")
            if n == 0:
                print("  （空列表，无字段可展示）")
            else:
                print(f"  第 1 个元素的结构（真实字段，未做假设）:")
                print("  " + describe_value(data[0], depth=0, max_depth=3).replace("\n", "\n  "))
                # 收集第一个元素的键，供后续语义映射使用
                if isinstance(data[0], dict):
                    rec["first_keys"] = list(data[0].keys())
                # 如果元素是 dict 且包含 question/answers，记录为 FITB 候选
                if isinstance(data[0], dict) and "question" in data[0]:
                    findings["fitb_sample"] = data[0]
                    findings["fitb_n"] = n
                    findings["fitb_file"] = str(f)
                # 如果元素是 dict 且包含 items/set_id，记录为穿搭 JSON 候选
                if isinstance(data[0], dict) and ("items" in data[0] and "set_id" in data[0]):
                    findings["outfit_json_sample"] = data[0]
                    findings["outfit_json_n"] = n
                    findings["outfit_json_file"] = str(f)
        elif isinstance(data, dict):
            n = len(data)
            rec["n"] = n
            print(f"  顶层类型: dict，共 {n} 个键")
            keys = list(data.keys())
            print(f"  前 5 个键: {[truncate(k, 40) for k in keys[:5]]}")
            if keys:
                first_key = keys[0]
                print(f"  第 1 个键 {truncate(str(first_key), 40)!r} 对应的值结构:")
                print("  " + describe_value(data[first_key], depth=0, max_depth=3).replace("\n", "\n  "))
                rec["first_key"] = str(first_key)
                # 识别 item 元数据（值为 dict，含 category_id/semantic_category 等）
                fv = data[first_key]
                if isinstance(fv, dict):
                    rec["first_value_keys"] = list(fv.keys())
                    if "category_id" in fv or "semantic_category" in fv:
                        findings["metadata_sample"] = fv
                        findings["metadata_n"] = n
                        findings["metadata_file"] = str(f)
                    if "url_name" in fv and "title" in fv and "category_id" not in fv:
                        findings["titles_sample"] = fv
                        findings["titles_n"] = n
                        findings["titles_file"] = str(f)
        else:
            print(f"  顶层类型: {type(data).__name__}（非 list/dict）")
            print("  " + describe_value(data, depth=0, max_depth=2))
        findings["json"].append(rec)
        if size_mb is not None:
            print(f"  (文件大小约 {size_mb:.1f}MB)")


# ------------------------------------------------------------------------------
# CSV 检查
# ------------------------------------------------------------------------------
def inspect_csv(files, findings):
    """检查 CSV 文件：是否有表头、列数、样例。"""
    section("[6] CSV 文件检查")
    if not _HAS_PANDAS:
        print("!! 未安装 pandas，跳过 CSV 结构化读取。")
        return
    findings["csv"] = []
    for f in files:
        print("-" * 80)
        print(f"文件: {f.relative_to(findings['root']) if 'root' in findings else f}")
        # 先看原始前 5 行，判断是否有表头
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                raw_lines = [next(fh).rstrip("\n") for _ in range(5)]
        except Exception as e:
            print(f"  !! 读取失败: {e}")
            continue
        print("  原始前几行:")
        for ln in raw_lines:
            print(f"    | {truncate(ln, 120)}")

        # 尝试 pandas 读取（无表头），打印列数与样例
        for header_mode, label in [(None, "无表头(header=None)"), ("infer", "带表头(header=infer)")]:
            try:
                df = pd.read_csv(f, header=header_mode, nrows=5, encoding="utf-8", dtype=str)
                print(f"  按 {label} 读取 -> 形状(样例)={df.shape}, 列={list(df.columns)[:10]}")
                if header_mode is None:
                    findings["csv"].append({"file": str(f), "n_cols": df.shape[1],
                                            "first_row": df.iloc[0].tolist() if len(df) else []})
                break
            except Exception as e:
                print(f"  按 {label} 读取失败: {e}")


# ------------------------------------------------------------------------------
# 文本文件检查（compatibility_*.txt 等）
# ------------------------------------------------------------------------------
def inspect_txt(files, findings):
    """检查文本文件：行数、预览、若是 compatibility 则统计 0/1 标签。"""
    section("[7] 文本文件(.txt) 检查")
    findings["txt"] = []
    for f in files:
        print("-" * 80)
        print(f"文件: {f.relative_to(findings['root']) if 'root' in findings else f}")
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            print(f"  !! 读取失败: {e}")
            continue
        print(f"  总行数: {len(lines)}")
        for ln in lines[:5]:
            print(f"    样例行: {truncate(ln, 120)}")

        # compatibility 文件：首列通常是标签 0/1
        if f.name.startswith("compatibility"):
            labels = []
            for ln in lines:
                parts = ln.split()
                if parts:
                    labels.append(parts[0])
            if labels:
                cnt = Counter(labels)
                print(f"  首列取值分布: {dict(cnt)}   (1=正样本/真实穿搭, 0=负样本/随机不搭配)")
                findings["txt"].append({"file": str(f), "labels": dict(cnt),
                                        "sample_line": lines[0] if lines else ""})
                if "compat_labels" not in findings:
                    findings["compat_labels"] = cnt
                else:
                    findings["compat_labels"] += cnt
        else:
            findings["txt"].append({"file": str(f), "n_lines": len(lines)})


# ------------------------------------------------------------------------------
# pickle 文件检查（只识别，不反序列化）
# ------------------------------------------------------------------------------
def inspect_pickle(files):
    """报告 pickle 文件存在与文本预览，绝不执行 pickle.load。"""
    section("[8] pickle 文件检查（安全起见，不反序列化）")
    for f in files:
        print("-" * 80)
        print(f"文件: {f}")
        try:
            size = f.stat().st_size
        except Exception:
            size = 0
        print(f"  大小: {_human_size(size)}")
        # 只做文本预览（protocol 0 的 pickle 是可读 ASCII，能看出内容，但依然不加载）
        try:
            raw = f.read_bytes()[:300]
            print(f"  前 300 字节文本预览: {truncate(raw.decode('latin-1', errors='replace'), 300)}")
        except Exception as e:
            print(f"  !! 预览失败: {e}")
        print("  说明: 该文件为 pickle，可能含“类型组合空间(typespaces)”等结构。")
        print("        脚本不会自动加载它，如需使用请确认来源可信后再自行 pickle.load。")


# ------------------------------------------------------------------------------
# 图片检查
# ------------------------------------------------------------------------------
def inspect_images(files, findings):
    """
    图片检查：
      1) 统计散落的图片文件数量并抽样打开验证；
      2) 对 Parquet 内嵌图片字节抽样做解码验证。
    """
    section("[9] 图片检查")
    img_files = [f for f in files if f.suffix.lower() in IMAGE_EXTS]
    print(f"散落的图片文件数量: {len(img_files)}")
    if img_files:
        sample = random.sample(img_files, min(10, len(img_files)))
        print(f"抽样 {len(sample)} 张验证能否打开:")
        ok = 0
        for f in sample:
            try:
                if not _HAS_PIL:
                    print(f"  !! 未安装 Pillow，跳过打开验证: {f.name}")
                    break
                im = Image.open(str(f))
                im.load()
                print(f"  [OK] {f.name}: 格式={im.format}, 尺寸={im.size}, 模式={im.mode}")
                ok += 1
            except Exception as e:
                print(f"  [FAIL] {f.name}: {e}")
        print(f"  抽样结果: {ok}/{len(sample)} 张可正常打开")

    # Parquet 内嵌图片抽样解码
    print("-" * 80)
    print("Parquet 内嵌图片字节抽样解码验证:")
    if "parquet" not in findings:
        print("  （未检查到 Parquet）")
        return
    all_samples = []
    for e in findings["parquet"]:
        all_samples.extend(e["image_sample"])
    if not all_samples:
        print("  （Parquet 中未发现图片字节列，或未读到样例行）")
        return
    random.shuffle(all_samples)
    sample = all_samples[:min(5, len(all_samples))]
    ok = 0
    for img in sample:
        b = img.get("bytes")
        path = img.get("path")
        try:
            if not _HAS_PIL:
                print("  !! 未安装 Pillow，跳过解码验证")
                break
            im = Image.open(io.BytesIO(b))
            im.load()
            print(f"  [OK] path={path!r}: 格式={im.format}, 尺寸={im.size}, 模式={im.mode}, 字节数={len(b)}")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] path={path!r}: {e}")
    print(f"  抽样结果: {ok}/{len(sample)} 张可正常解码")


# ------------------------------------------------------------------------------
# 字段语义映射与关键字段存在性检查
# ------------------------------------------------------------------------------
def semantic_mapping(findings):
    """
    根据脚本实际发现的结构，输出字段语义映射，并检查关键字段是否缺失。
    原则：只陈述“实际发现”，发现不了就明确写“未发现”，不虚构字段。
    """
    section("[10] 字段语义映射与关键字段存在性检查")
    print("以下结论完全基于本脚本在本地数据中的实际发现，未做任何假设。\n")

    rows = []

    # ---- 单品 ID ----
    where_id = []
    if any("item_id" in e["cols"] for e in findings.get("parquet", [])):
        where_id.append("Parquet 列 'item_id'（data/{disjoint,nondisjoint} 下的全部 .parquet）")
    if findings.get("outfit_json_sample") is not None:
        s = findings["outfit_json_sample"]
        if isinstance(s.get("items"), list) and s["items"] and isinstance(s["items"][0], dict) and "item_id" in s["items"][0]:
            where_id.append("穿搭 JSON 中 items[].item_id")
    if findings.get("metadata_sample") is not None:
        where_id.append("item 元数据 JSON 的顶层 key（item_id 作为字典键）")
    rows.append(("单品 ID", where_id))

    # ---- 图片 ----
    where_img = []
    if any("image" in e["cols"] and e.get("image_struct") is not None for e in findings.get("parquet", [])):
        where_img.append("Parquet 列 'image'（struct{bytes, path}，图片以**字节**内嵌，path 形如 '{item_id}.jpg'）")
    loose = sum(1 for _ in findings.get("all_files", []) if Path(_).suffix.lower() in IMAGE_EXTS)
    if loose:
        where_img.append(f"散落图片文件 {loose} 张")
    rows.append(("图片", where_img))

    # ---- 服装类别 ----
    where_cat = []
    for c in findings.get("csv", []):
        if "categories" in c["file"].lower() and c.get("n_cols") is not None and c["n_cols"] >= 3:
            where_cat.append("categories.csv（无表头，3 列: category_id / 类别名 / semantic_category 粗粒度类型）")
    if findings.get("metadata_sample") is not None:
        keys = list(findings["metadata_sample"].keys())
        if "category_id" in keys or "semantic_category" in keys:
            where_cat.append("item 元数据 JSON 中 category_id / semantic_category 字段")
    rows.append(("服装类别", where_cat))

    # ---- 穿搭 ID ----
    where_outfit = []
    if findings.get("outfit_json_sample") is not None:
        s = findings["outfit_json_sample"]
        if "set_id" in s:
            where_outfit.append("穿搭 JSON 中 set_id")
    if findings.get("fitb_sample") is not None:
        where_outfit.append("fill_in_blank JSON / compatibility txt 中 '{set_id}_{index}' 的前缀部分")
    rows.append(("穿搭 ID", where_outfit))

    # ---- 穿搭组合 ----
    where_combo = []
    if findings.get("outfit_json_sample") is not None:
        s = findings["outfit_json_sample"]
        if isinstance(s.get("items"), list):
            where_combo.append("穿搭 JSON 中 items 列表（每个元素含 item_id 与 index=单品在穿搭中的位置）")
    rows.append(("穿搭组合", where_combo))

    # ---- 训练划分 ----
    where_split = []
    splits = findings.get("split_dirs", [])
    if splits:
        where_split.append("目录结构: " + ", ".join(str(s) for s in splits[:12]) + ("..." if len(splits) > 12 else ""))
    rows.append(("训练划分", where_split))

    # ---- 兼容性标签 ----
    where_label = []
    if findings.get("compat_labels"):
        where_label.append(f"compatibility_*.txt 首列 0/1，本目录合计分布={dict(findings['compat_labels'])}")
    rows.append(("兼容性标签", where_label))

    # ---- FITB 标签 ----
    where_fitb = []
    if findings.get("fitb_sample") is not None:
        keys = list(findings["fitb_sample"].keys())
        where_fitb.append(f"fill_in_blank JSON: 字段={keys}（question=已知单品, blank_position=挖空位置, answers=4个候选）")
    rows.append(("FITB 标签", where_fitb))

    # ---- 打印表格 ----
    for role, places in rows:
        if places:
            print(f"  ✓ {role}:")
            for p in places:
                print(f"      - {p}")
        else:
            print(f"  ✗ {role}: **未发现**（请在原始数据中进一步确认，勿自行假定）")
        print()

    # ---- 明确的风险提示 ----
    print("需要特别说明的“非直观”结构（重要）:")
    notes = []
    if any(e.get("image_struct") is not None for e in findings.get("parquet", [])):
        notes.append("图片并非独立文件/字符串路径，而是以 JPEG 字节内嵌在 Parquet 的 image.bytes 中；"
                     "image.path 仅是文件名(如 '12345.jpg')，与 item_id 一致。")
    if findings.get("csv"):
        notes.append("categories.csv **没有表头**，第一列是数字 category_id，第三列才是粗粒度类型(semantic_category)。")
    if findings.get("metadata_sample") is not None and "catgeories" in findings["metadata_sample"]:
        notes.append("item 元数据中存在拼写错误字段 'catgeories'（应为 categories）。")
    if not notes:
        notes.append("未发现明显的非直观结构。")
    for n in notes:
        print(f"  * {n}")

    return rows


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Polyvore Outfits 数据集检查脚本（只检查，不训练，不修改数据）")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT),
                        help="数据集根目录（默认: data/raw/polyvore）")
    parser.add_argument("--report", type=str, default=None,
                        help="报告输出文件路径（默认: scripts/dataset_inspection_report.txt）")
    parser.add_argument("--no-report", action="store_true",
                        help="不写报告文件，只在控制台打印")
    args = parser.parse_args()

    root = Path(args.root)
    report_path = Path(args.report) if args.report else (SCRIPT_DIR / "dataset_inspection_report.txt")

    # 把控制台输出同步写入内存缓冲区，结束时统一写入报告文件（tee 模式）
    _buffer = io.StringIO()
    _real_stdout = sys.stdout
    sys.stdout = _Tee(_real_stdout, _buffer)

    try:
        # 依赖与目录
        print_deps()
        print_tree(root)
        if not root.exists():
            print(f"!! 数据目录不存在: {root}")
            print("   请先下载数据集，或使用 --root 指定正确路径。")
            sys.exit(1)

        all_files, by_ext = collect_files(root)
        print_file_type_summary(by_ext)

        # 分类文件
        parquet_files = sorted(f for f in all_files if f.suffix.lower() in PARQUET_EXTS)
        json_files = sorted(f for f in all_files if f.suffix.lower() in JSON_EXTS)
        csv_files = sorted(f for f in all_files if f.suffix.lower() in CSV_EXTS)
        txt_files = sorted(f for f in all_files if f.suffix.lower() in TEXT_EXTS)
        pickle_files = sorted(f for f in all_files if f.suffix.lower() in PICKLE_EXTS)
        other_files = sorted(f for f in all_files
                             if f.suffix.lower() not in (PARQUET_EXTS | JSON_EXTS | CSV_EXTS
                                                         | TEXT_EXTS | PICKLE_EXTS | IMAGE_EXTS))

        # 训练划分目录（disjoint / nondisjoint × train/validation/test），排除 .cache
        split_dirs = []
        for d in sorted(root.rglob("*")):
            if d.is_dir() and d.name in {"disjoint", "nondisjoint"}:
                rel = d.relative_to(root)
                if ".cache" in rel.parts or ".venv" in rel.parts:
                    continue
                split_dirs.append(rel)
        findings = {
            "root": root,
            "all_files": [str(f) for f in all_files],
            "split_dirs": split_dirs,
        }

        # 依次检查各类文件
        if parquet_files:
            inspect_parquet(parquet_files, findings)
        else:
            print("\n[4] Parquet: 未发现 .parquet 文件")

        if json_files:
            inspect_json(json_files, findings)
        else:
            print("\n[5] JSON: 未发现 .json 文件")

        if csv_files:
            inspect_csv(csv_files, findings)
        else:
            print("\n[6] CSV: 未发现 .csv/.tsv 文件")

        if txt_files:
            inspect_txt(txt_files, findings)
        else:
            print("\n[7] 文本: 未发现 .txt/.md 文件")

        if pickle_files:
            inspect_pickle(pickle_files)
        else:
            print("\n[8] pickle: 未发现 .p/.pkl/.pickle 文件")

        if other_files:
            print("\n[8b] 其它文件:")
            for f in other_files:
                try:
                    sz = _human_size(f.stat().st_size)
                except Exception:
                    sz = "?"
                print(f"    {f.relative_to(root)}  ({sz})")

        # 图片检查（含 Parquet 内嵌图片抽样）
        inspect_images(all_files, findings)

        # 字段语义映射
        semantic_mapping(findings)

        # 最终结论
        section("[11] 结论：请把以下内容发回给算法开发方")
        print("1. 本脚本的完整控制台输出（或脚本自动生成的报告文件）。")
        print("2. 重点确认这几项（若脚本里显示“未发现”，也请一并说明）：")
        print("   - Parquet 的列名与类型（尤其 image 是否为 struct{bytes,path}）")
        print("   - 穿搭 JSON 的 items/set_id 结构")
        print("   - fill_in_blank JSON 的 question/blank_position/answers 结构")
        print("   - compatibility txt 的首列 0/1 分布")
        print("   - categories.csv 是否有表头、共几列")
        print("3. 你的 Python 依赖情况（pandas/pyarrow/Pillow 是否安装、版本号）。")
    finally:
        # 恢复 stdout，并写出报告文件
        sys.stdout = _real_stdout
        if not args.no_report:
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(_buffer.getvalue(), encoding="utf-8")
                print(f"报告已写入: {report_path}")
            except Exception as e:
                print(f"!! 写报告文件失败: {e}")


if __name__ == "__main__":
    main()
