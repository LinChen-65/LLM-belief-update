"""
Integrated version of the first-person judgment experiment.

This file merges the original four scripts into one script:
- SiliconFlow models: Qwen, GLM, DeepSeek, MiniMax
- OpenRouter model: Gemini
- Azure OpenAI deployment: gpt-4o-mini

Important:
1. Do NOT hard-code API keys in this file.
2. Set API keys with environment variables before running.
3. Keep single_turn_pairs.json and 2262_unique_ops.json in the same directory, or edit PAIRS_FILE / OPS_FILE.
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import AzureOpenAI, OpenAI


# ==========================================
# 1. 基础配置与文件路径
# ==========================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

PAIRS_FILE = os.path.join(DATA, "dataset", "single_turn_pairs.json")
OPS_FILE = os.path.join(DATA, "dataset", "2262_unique_ops.json")
OUTPUT_DIR = os.path.join(DATA, "first_person")

TIMEOUT_SECONDS = 60.0
TEMPERATURE = 0.1
MAX_RETRIES = 3
SAVE_EVERY = 20

# 如果你在服务器上需要走 Clash 代理，可以在运行前设置：
# export USE_CLASH_PROXY=1
# export CLASH_PROXY_URL=http://127.0.0.1:7897
if os.getenv("USE_CLASH_PROXY", "0") == "1":
    proxy_url = os.getenv("CLASH_PROXY_URL", "http://127.0.0.1:7897")
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url


# ==========================================
# 2. 模型与服务商配置
# ==========================================
@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    result_prefix: str = "final_v3_results"
    sleep_seconds: float = 0.5


MODEL_CONFIGS: List[ModelConfig] = [
    # SiliconFlow
    ModelConfig(provider="siliconflow", model="Qwen/Qwen2.5-32B-Instruct"),
    ModelConfig(provider="siliconflow", model="Qwen/Qwen2.5-72B-Instruct"),
    ModelConfig(provider="siliconflow", model="Pro/zai-org/GLM-4.7"),
    ModelConfig(provider="siliconflow", model="deepseek-ai/DeepSeek-V3"),
    ModelConfig(provider="siliconflow", model="Pro/MiniMaxAI/MiniMax-M2.5"),

    # OpenRouter
    ModelConfig(provider="openrouter", model="google/gemini-2.5-flash-lite"),

    # Azure OpenAI。这里的 model 应该填写 Azure 的 deployment name。
    ModelConfig(
        provider="azure",
        model="gpt-4o-mini",
        result_prefix="final_new_results",
        sleep_seconds=0.3,
    ),
]


# ==========================================
# 3. Prompt 模板设计
# ==========================================
SYSTEM_PROMPT_STEP1 = """You are the Original Poster (OP) who wrote the following post on an online forum.
Title: {title}
Body: {text}
"""

USER_PROMPT = """A challenger has replied to your post with the following argument:

Challenger's Reply:
{challenger_text}

As the OP, evaluate whether this reply has changed your view.

AWARDING A DELTA:
Award a delta ("delta_awarded": true) if the challenger's reply has actually changed your mind on this issue. This means your original claim, as stated in your post, no longer fully represents what you believe. The change can be a total shift or a substantial adjustment of your original stance.

Output your response strictly as a JSON object.

JSON RULES:
1. Use DOUBLE QUOTES for all keys and string values.
2. Escape any internal double quotes with a backslash.
3. Use \\n instead of actual line breaks inside string values.

Format:
{{
"delta_awarded": <true or false>,
"justification": "<Explain why the reply did or did not change your view.>"
}}
"""


# ==========================================
# 4. Client 初始化
# ==========================================
def build_client(provider: str):
    """根据 provider 创建对应客户端。"""
    timeout = httpx.Timeout(TIMEOUT_SECONDS)

    if provider == "siliconflow":
        api_key = os.getenv("SF_API_KEY")
        if not api_key:
            raise RuntimeError("缺少环境变量 SF_API_KEY，请先设置 SiliconFlow API Key。")
        return OpenAI(
            api_key=api_key,
            base_url=os.getenv("SF_BASE_URL", "https://api.siliconflow.cn/v1"),
            timeout=timeout,
        )

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("缺少环境变量 OPENROUTER_API_KEY，请先设置 OpenRouter API Key。")
        return OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=timeout,
        )

    if provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        if not api_key:
            raise RuntimeError("缺少环境变量 AZURE_OPENAI_API_KEY，请先设置 Azure OpenAI API Key。")
        if not endpoint:
            raise RuntimeError("缺少环境变量 AZURE_OPENAI_ENDPOINT，请先设置 Azure OpenAI Endpoint。")
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            timeout=timeout,
        )

    raise ValueError(f"未知 provider: {provider}")


def get_clients(configs: List[ModelConfig]) -> Dict[str, Any]:
    """同一个 provider 只初始化一个 client。"""
    clients: Dict[str, Any] = {}
    for cfg in configs:
        if cfg.provider not in clients:
            clients[cfg.provider] = build_client(cfg.provider)
    return clients


# ==========================================
# 5. 辅助函数：安全调用大模型
# ==========================================
def extract_json_object(content: str) -> Dict[str, Any]:
    """从模型输出中尽量提取 JSON 对象。"""
    content = content.strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    clean_content = match.group(0) if match else content
    return json.loads(clean_content)


def call_llm(client: Any, model_name: str, messages: list, max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
    """调用大模型并强制解析为 JSON 格式。"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
            )
            content = response.choices[0].message.content or ""
            return extract_json_object(content)

        except Exception as e:
            print(f"  [API Error - {model_name}] 尝试 {attempt + 1}/{max_retries} 失败: {e}")
            time.sleep(3)

    return {
        "error": "Failed after max retries",
        "delta_awarded": False,
        "justification": "API Error",
    }


def safe_model_filename(model_name: str) -> str:
    """把模型名转换为适合文件名的字符串。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ==========================================
# 6. 核心实验流程：支持所有模型 + 断点续传
# ==========================================
def run_experiment(
    configs: Optional[List[ModelConfig]] = None,
    pairs_file: str = PAIRS_FILE,
    ops_file: str = OPS_FILE,
    output_dir: str = OUTPUT_DIR,
) -> None:
    configs = configs or MODEL_CONFIGS

    print("正在加载数据集...")
    pairs = load_json(pairs_file)
    ops_list = load_json(ops_file)

    ops_dict = {op["id"]: op for op in ops_list}
    total_pairs = len(pairs)
    print(f"共加载 {total_pairs} 对单轮测试数据。准备执行全量测试！")

    clients = get_clients(configs)

    for cfg in configs:
        model = cfg.model
        client = clients[cfg.provider]

        print("\n==================================================")
        print(f"🚀 开始测试模型: {model} ({cfg.provider})")
        print("==================================================")

        safe_name = safe_model_filename(model)
        result_path = Path(output_dir) / f"{cfg.result_prefix}_{safe_name}.json"

        experiment_results = []
        processed_pairs = set()

        # 断点续传
        if result_path.exists():
            try:
                experiment_results = load_json(str(result_path))
                processed_pairs = {item["pair_id"] for item in experiment_results if "pair_id" in item}
                print(
                    f"📦 发现本地存档，{model} 已完成 "
                    f"{len(processed_pairs)}/{total_pairs} 条测试，自动跳过已完成数据..."
                )
            except json.JSONDecodeError:
                print("⚠️ 存档文件损坏或为空，将从头开始测试。")
                experiment_results = []
                processed_pairs = set()

        count = len(processed_pairs)

        for p_id, pair_data in pairs.items():
            if p_id in processed_pairs:
                continue

            success_reply = pair_data["success"]
            failure_reply = pair_data["failure"]
            root_id = success_reply["root"]

            if root_id not in ops_dict:
                continue

            op_data = ops_dict[root_id]
            print(f"\n--- [{model}] 进度: {count + 1}/{total_pairs} (Pair ID: {p_id}) ---")

            # ---------- Step 1: 角色注入 ----------
            prompt1 = SYSTEM_PROMPT_STEP1.format(
                title=op_data.get("title", ""),
                text=op_data.get("text", ""),
            )
            messages_step1 = [{"role": "system", "content": prompt1}]

            # ---------- Step 2 & 3: 平行时空刺激 ----------
            prompt_success = USER_PROMPT.format(challenger_text=success_reply["text"])
            messages_branch_a = messages_step1 + [{"role": "user", "content": prompt_success}]
            res_branch_a = call_llm(client, model, messages_branch_a)

            prompt_failure = USER_PROMPT.format(challenger_text=failure_reply["text"])
            messages_branch_b = messages_step1 + [{"role": "user", "content": prompt_failure}]
            res_branch_b = call_llm(client, model, messages_branch_b)

            # ---------- 记录结果 ----------
            result_entry = {
                "provider": cfg.provider,
                "model": model,
                "pair_id": p_id,
                "root_id": root_id,
                "branch_A_human_success": {
                    "human_label": 1,
                    "agent_delta": res_branch_a.get("delta_awarded"),
                    "justification": res_branch_a.get("justification"),
                    "raw_error": res_branch_a.get("error"),
                },
                "branch_B_human_failure": {
                    "human_label": 0,
                    "agent_delta": res_branch_b.get("delta_awarded"),
                    "justification": res_branch_b.get("justification"),
                    "raw_error": res_branch_b.get("error"),
                },
            }

            experiment_results.append(result_entry)
            processed_pairs.add(p_id)
            count += 1

            # 实时保存
            if count % SAVE_EVERY == 0:
                save_json(result_path, experiment_results)
                print(f"💾 [自动存盘] {model} 的数据已保存 ({count}/{total_pairs})")

            time.sleep(cfg.sleep_seconds)

        # 模型跑完后统一保存
        save_json(result_path, experiment_results)
        print(f"\n✅ {model} 测试完成！最终结果保存在 {result_path}")


# ==========================================
# 7. 命令行入口
# ==========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first-person judgment experiment across multiple models.")
    parser.add_argument("--provider", choices=["siliconflow", "openrouter", "azure"], help="只运行某个服务商的模型。")
    parser.add_argument("--model", help="只运行某一个模型/Deployment Name，必须与 MODEL_CONFIGS 中的 model 完全一致。")
    parser.add_argument("--pairs-file", default=PAIRS_FILE, help="single_turn_pairs.json 的路径。")
    parser.add_argument("--ops-file", default=OPS_FILE, help="2262_unique_ops.json 的路径。")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="结果文件输出目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_configs = MODEL_CONFIGS
    if args.provider:
        selected_configs = [cfg for cfg in selected_configs if cfg.provider == args.provider]
    if args.model:
        selected_configs = [cfg for cfg in selected_configs if cfg.model == args.model]

    if not selected_configs:
        raise RuntimeError("没有匹配到要运行的模型，请检查 --provider 或 --model 参数。")

    run_experiment(
        configs=selected_configs,
        pairs_file=args.pairs_file,
        ops_file=args.ops_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
