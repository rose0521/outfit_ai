# -*- coding: utf-8 -*-
"""
================================================================================
DeepSeek LLM 接入自测脚本  test_llm.py
================================================================================

用途：验证 API key、模型名、网络和 function calling 是否正常，不依赖完整服务。

用法：
  export DEEPSEEK_API_KEY=sk-xxxxxxxx            # Windows: set DEEPSEEK_API_KEY=sk-xxx
  python -X utf8 scripts/test_llm.py "帮我找一件白色衬衫"

会在三步里分别验证：
  1) 配置读取（key / base_url / model）
  2) 最小连通性（让模型回两个字）
  3) function calling 路由 + 参数抽取（用占位执行器，实际工具在 serve_demo 里跑）

如果第 2 步报错：检查 key、网络、base_url、model 名。
如果第 3 步报错但第 2 步正常：大概率是 model 不支持 tools/function calling，换 deepseek-chat 试试。
================================================================================
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_agent import LLMAgent


def fake_execute(name, args):
    """测试占位执行器：只回显工具名和参数，证明 LLM 路由正确。"""
    return {"tool": name, "args": args, "note": "测试占位，实际由 serve_demo 执行真实工具"}


def main():
    message = sys.argv[1] if len(sys.argv) > 1 else "帮我找一件白色衬衫"
    agent = LLMAgent()

    print("=" * 60)
    print("DeepSeek LLM 接入自测")
    print("=" * 60)
    print(f"base_url : {agent.base_url}")
    print(f"model    : {agent.model}")
    if not agent.is_configured():
        print("\n!! 未检测到 DEEPSEEK_API_KEY")
        print("   请先设置：export DEEPSEEK_API_KEY=sk-xxx")
        print("   或在项目根目录建 .env 文件，写入 DEEPSEEK_API_KEY=sk-xxx")
        raise SystemExit(1)
    print(f"api_key  : {agent.api_key[:6]}...（已配置，{len(agent.api_key)} 位）")

    print("\n[1] 最小连通性测试（让模型回两个字）...")
    try:
        pong = agent.ping()
        print(f"    模型回复：{pong}")
    except Exception as e:
        print(f"    !! 失败：{e}")
        print("    请检查：key 是否正确 / 网络能否访问 api.deepseek.com / model 名是否正确")
        raise SystemExit(1)

    print(f"\n[2] function calling 路由测试：\"{message}\"")
    try:
        reply, actions = agent.run(message, fake_execute)
    except Exception as e:
        print(f"    !! 失败：{e}")
        print("    若第[1]步正常而此步失败，可能是该模型不支持 tools/function calling")
        raise SystemExit(1)

    print(f"    回复：{reply or '(无文字回复)'}")
    print(f"    动作：{json.dumps(actions, ensure_ascii=False, indent=2)}")

    print("\n自测通过：API 已接通，function calling 路由正常。")
    print("接下来跑起服务后，POST /api/chat 即走这套 LLM 路由。")


if __name__ == "__main__":
    main()
