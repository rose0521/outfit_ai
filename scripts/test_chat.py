# -*- coding: utf-8 -*-
"""
================================================================================
测 /api/chat 的小工具  test_chat.py
================================================================================

避开 Windows PowerShell 里 curl 的引号/中文编码坑，直接用标准库 urllib 发请求。
用法（先确保 serve_demo.py 在跑）：

  python -X utf8 scripts/test_chat.py "预算300以内买柔软的连衣裙"
  python -X utf8 scripts/test_chat.py "帮我找一件白色衬衫" 8000
================================================================================
"""

import json
import sys
import urllib.error
import urllib.request


def main():
    message = sys.argv[1] if len(sys.argv) > 1 else "预算300以内买柔软的连衣裙"
    port = sys.argv[2] if len(sys.argv) > 2 else "8000"
    url = f"http://127.0.0.1:{port}/api/chat"

    payload = json.dumps({"message": message}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    print(f"POST {url}")
    print(f"message: {message}\n")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, "->", e.read().decode("utf-8"))
        raise SystemExit(1)
    except urllib.error.URLError as e:
        print("连接失败（serve_demo.py 没在跑？）:", e.reason)
        raise SystemExit(1)

    print("reply :", data.get("reply"))
    print("actions:")
    print(json.dumps(data.get("actions"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
