# -*- coding: utf-8 -*-
"""
================================================================================
LLM 自然语言路由（OpenAI 兼容 function calling，适配 DeepSeek）  llm_agent.py
================================================================================

零第三方依赖：只用标准库 urllib 直接调 DeepSeek 的 OpenAI 兼容接口。
作用：把 /api/chat 的一句话需求路由到「检索 / 推荐 / 质量 / 搭配打分 / 补全」，
由 LLM 决定调哪个工具、抽取参数；工具仍在本地执行（复用已有逻辑）。

配置（优先级从高到低）：
  1) 环境变量 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
  2) 项目根目录 .env 文件（KEY=VALUE 每行一行，已加入 .gitignore，不会上传）
  3) 代码内默认值（仅 base_url / model）

示例：
  export DEEPSEEK_API_KEY=sk-xxxxxxxx
  python scripts/test_llm.py "帮我找一件白色衬衫"

未配置 DEEPSEEK_API_KEY 时 is_configured() 返回 False，/api/chat 自动回退到规则路由，
不影响主线。API 调用失败也会由调用方回退，不会崩服务。
================================================================================
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

SYSTEM_PROMPT = """你是女装购物助手。根据用户的一句话需求，判断该调用哪个工具并抽取参数：
- 想"找/搜"某类商品（按外观、款式、颜色）→ search_products
- 有预算、面料、版型、场景等明确购买要求 → recommend_products
- 问某件具体商品的质量/评价/好不好 → get_quality
- 问几件商品搭不搭 → score_outfit
- 要补全一套搭配缺的那件 → fill_outfit
- 闲聊、或信息不足无法判断时：直接文字回答，不要调工具。

注意：get_quality / score_outfit / fill_outfit 需要商品编号 item_id（形如 amazon23_ 开头）。
只有当用户消息里明确出现这类编号时才调用；否则先 search_products 或 recommend_products。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "按关键词/描述检索商品（CLIP 图文检索，看外观、款式、颜色）。适合'找一件白色衬衫'这类需求。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索词或描述，如'白色通勤衬衫'"},
                    "top_k": {"type": "integer", "description": "返回数量，默认5，最大20"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_products",
            "description": "按购买需求推荐商品（解析品类/预算/面料/版型/场景等偏好）。适合'预算300以内买柔软的连衣裙'这类有明确要求的需求。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户的完整需求描述"},
                    "limit": {"type": "integer", "description": "返回数量，默认3，最大20"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_quality",
            "description": "查看某件商品的质量评估（评分、好评率、优缺点维度分析）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "商品编号，如 amazon23_B01C7UW6QQ"},
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_outfit",
            "description": "给一套穿搭（多件商品）打搭配兼容度分数（0~1，越高越搭）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "商品编号列表，如 ['amazon23_xxx','amazon23_yyy']",
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill_outfit",
            "description": "补全搭配：给定已有单品和候选单品，推荐最搭的缺件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}, "description": "已有单品编号列表"},
                    "candidates": {"type": "array", "items": {"type": "string"}, "description": "候选补全单品编号列表"},
                },
                "required": ["items", "candidates"],
            },
        },
    },
]


def _load_env_file(path):
    """极简 .env 解析（KEY=VALUE 每行一行，支持 # 注释）。"""
    if not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _post_json(url, payload, api_key, timeout=60):
    """POST JSON 到 OpenAI 兼容接口，返回解析后的 dict；非 2xx 抛 HTTPError。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class LLMAgent:
    def __init__(self, api_key=None, base_url=None, model=None):
        env = _load_env_file(ENV_FILE)
        self.api_key = (api_key or os.environ.get("DEEPSEEK_API_KEY")
                        or env.get("DEEPSEEK_API_KEY") or "").strip()
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL")
                         or env.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = (model or os.environ.get("DEEPSEEK_MODEL")
                      or env.get("DEEPSEEK_MODEL") or DEFAULT_MODEL)

    def is_configured(self):
        return bool(self.api_key)

    def _chat(self, messages, tools=None):
        payload = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return _post_json(f"{self.base_url}/chat/completions", payload, self.api_key)

    def ping(self):
        """最小连通性测试：让模型回一个字。返回模型文本回复。"""
        resp = self._chat([{"role": "user", "content": "只回复两个字：正常"}])
        return resp["choices"][0]["message"].get("content") or ""

    def run(self, message, execute_tool, history=None):
        """路由一句话需求并执行工具，返回 (reply: str, actions: list)。

        execute_tool(name, args) -> dict：由调用方（serve_demo）实现，本地执行真实工具。
        history: 可选的多轮历史，形如 [{"role": "user", "content": ...}, ...]。
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        # 第一轮：让模型决定要不要调工具、调哪个、参数是什么
        resp = self._chat(messages, tools=TOOLS)
        msg = resp["choices"][0]["message"]

        if not msg.get("tool_calls"):
            # 模型直接文字回答（闲聊 / 拒答 / 信息不足）
            return (msg.get("content") or "").strip(), []

        tool_calls = msg["tool_calls"]
        actions = []
        # 执行工具，并把结果回传给模型
        messages.append(msg)  # assistant 消息（含 tool_calls）
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                args = {}
            try:
                result = execute_tool(name, args)
            except Exception as e:
                result = {"error": str(e)}
            actions.append({"type": name, "args": args, "result": result})
            # 工具结果回传（截断，避免超长 token）
            content = json.dumps(result, ensure_ascii=False)
            if len(content) > 3000:
                content = content[:3000] + "...(截断)"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": content,
            })

        # 第二轮：让模型基于工具结果生成自然语言回答
        try:
            resp2 = self._chat(messages)
            reply = (resp2["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            reply = ""
        return reply, actions
