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

DEFAULT_PAIRS_FILE = os.path.join(DATA, "dataset", "single_turn_pairs.json")
DEFAULT_OPS_FILE = os.path.join(DATA, "dataset", "2262_unique_ops.json")
DEFAULT_OUTPUT_DIR = os.path.join(DATA, "third_person")

# SiliconFlow API 配置
SF_BASE_URL = "https://api.siliconflow.cn/v1"

# OpenRouter API 配置
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Azure OpenAI API 配置
DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"


@dataclass(frozen=True)
class ModelConfig:
    """
    provider:
        - siliconflow: 走 SiliconFlow OpenAI-compatible API
        - openrouter: 走 OpenRouter OpenAI-compatible API
        - azure: 走 Azure OpenAI API

    model:
        对 SiliconFlow/OpenRouter 来说是模型名；
        对 Azure 来说是 deployment name。
    """
    provider: str
    model: str


# ==========================================
# 2. 需要轮流测试的所有模型
# ==========================================
MODELS_TO_TEST: List[ModelConfig] = [
    # SiliconFlow models
    ModelConfig(provider="siliconflow", model="Qwen/Qwen2.5-32B-Instruct"),
    ModelConfig(provider="siliconflow", model="Qwen/Qwen2.5-72B-Instruct"),
    ModelConfig(provider="siliconflow", model="Pro/MiniMaxAI/MiniMax-M2.5"),
    ModelConfig(provider="siliconflow", model="Pro/zai-org/GLM-4.7"),
    ModelConfig(provider="siliconflow", model="deepseek-ai/DeepSeek-V3"),

    # OpenRouter models
    ModelConfig(provider="openrouter", model="google/gemini-2.5-flash-lite"),

    # Azure OpenAI deployments
    ModelConfig(provider="azure", model="gpt-4o-mini"),
]


# ==========================================
# 3. Prompt 模板设计：第三人称旁观者视角对齐版
# ==========================================
SYSTEM_PROMPT_STEP1 = """You are an impartial external observer reading an exchange on an online forum. The Original Poster (OP) wrote the following post.
Title: {title}
Body: {text}
"""

USER_PROMPT = """A challenger has replied to the OP's post with the following argument:

Challenger's Reply:
{challenger_text}

As an external observer, predict whether this reply has changed the OP's view.

AWARDING A DELTA:
Predict whether a delta is awarded. A delta should be awarded if the challenger's reply would actually change the OP's mind on this issue. This means the OP's original claim, as stated in their post, would no longer fully represent what they believe. The change can be a total shift or a substantial adjustment of the OP's original stance.

Output your response strictly as a JSON object.

JSON RULES:
1. Use DOUBLE QUOTES for all keys and string values.
2. Escape any internal double quotes with a backslash.
3. Use \\n instead of actual line breaks inside string values.

Format:
{{
"delta_awarded": <true or false>,
"justification": "<Explain why you predict the reply would or would not change the OP's view.>"
}}
"""


# ==========================================
# 4. API Client 构建
# ==========================================
def maybe_enable_proxy() -> None:
    """
    如需让 Python 走 Clash 代理，运行前设置：
        export USE_CLASH_PROXY=1
        export CLASH_PROXY_URL="http://127.0.0.1:7897"

    默认不强制开启代理，避免在不需要代理的服务器上影响请求。
    """
    if os.getenv("USE_CLASH_PROXY", "0") == "1":
        proxy_url = os.getenv("CLASH_PROXY_URL", "http://127.0.0.1:7897")
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        print(f"🌐 已启用代理: {proxy_url}")


def build_client(provider: str):
    """
    根据 provider 创建对应 client。
    注意：不要把 API Key 写死在代码里，统一用环境变量读取。
    """
    timeout = httpx.Timeout(60.0)

    if provider == "siliconflow":
        api_key = os.getenv("SF_API_KEY")
        if not api_key:
            raise RuntimeError(
                "缺少环境变量 SF_API_KEY。请先执行：export SF_API_KEY='你的 SiliconFlow API Key'"
            )
        return OpenAI(api_key=api_key, base_url=SF_BASE_URL, timeout=timeout)

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "缺少环境变量 OPENROUTER_API_KEY。请先执行：export OPENROUTER_API_KEY='你的 OpenRouter API Key'"
            )
        return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, timeout=timeout)

    if provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION)

        if not api_key:
            raise RuntimeError(
                "缺少环境变量 AZURE_OPENAI_API_KEY。请先执行：export AZURE_OPENAI_API_KEY='你的 Azure Key'"
            )
        if not endpoint:
            raise RuntimeError(
                "缺少环境变量 AZURE_OPENAI_ENDPOINT。请先执行：export AZURE_OPENAI_ENDPOINT='你的 Azure Endpoint'"
            )

        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            timeout=timeout,
        )

    raise ValueError(f"未知 provider: {provider}")


# ==========================================
# 5. 辅助函数：安全调用大模型
# ==========================================
def extract_json_object(content: str) -> Dict[str, Any]:
    """
    尽量从模型输出中解析 JSON。
    保留你原代码中的强力正则提取逻辑。
    """
    content = content.strip()

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        content = match.group(0)

    return json.loads(content)


def call_llm(
    client,
    model_name: str,
    messages: list,
    max_retries: int = 3,
    sleep_seconds: float = 3.0,
) -> Dict[str, Any]:
    """
    调用大模型并强制解析为 JSON 格式。
    基本逻辑沿用原始四个脚本：response_format + 正则提取 + 最多重试。
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            content = response.choices[0].message.content or ""
            return extract_json_object(content)

        except Exception as e:
            print(f"  [API Error - {model_name}] 尝试 {attempt + 1}/{max_retries} 解析失败: {e}")
            time.sleep(sleep_seconds)

    return {
        "error": "Failed after max retries",
        "delta_awarded": False,
        "justification": "API Error",
    }


def safe_model_filename(model: str) -> str:
    """
    把模型名转换成适合作为文件名的形式。
    """
    return (
        model.replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def result_filename_for_model(model_config: ModelConfig, output_dir: Path) -> Path:
    """
    统一保存文件名。
    保留 observer 标识，与第一人称实验结果区分。
    """
    safe_name = safe_model_filename(model_config.model)
    return output_dir / f"final_v3_results_observer_{safe_name}.json"


# ==========================================
# 6. 数据加载与实验逻辑
# ==========================================
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results(result_file: Path, total_pairs: int, model_name: str):
    """
    断点续传：如果已有结果文件，就读取已经完成的 pair_id。
    """
    experiment_results = []
    processed_pairs = set()

    if result_file.exists():
        try:
            with result_file.open("r", encoding="utf-8") as f:
                experiment_results = json.load(f)

            processed_pairs = {
                item["pair_id"]
                for item in experiment_results
                if isinstance(item, dict) and "pair_id" in item
            }

            print(
                f"📦 发现本地存档，{model_name} 已完成 "
                f"{len(processed_pairs)}/{total_pairs} 条测试，自动跳过已完成数据..."
            )

        except json.JSONDecodeError:
            print("⚠️ 存档文件损坏或为空，将从头开始测试。")
            experiment_results = []
            processed_pairs = set()

    return experiment_results, processed_pairs


def save_results(result_file: Path, experiment_results: list) -> None:
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=4, ensure_ascii=False)


def run_one_model(
    model_config: ModelConfig,
    pairs: Dict[str, Any],
    ops_dict: Dict[str, Any],
    total_pairs: int,
    output_dir: Path,
    save_every: int = 20,
    request_interval: float = 0.5,
    max_retries: int = 3,
) -> None:
    """
    对单个模型跑完整个 observer-mode 实验。
    """
    model = model_config.model
    provider = model_config.provider

    print("\n==================================================")
    print(f"🚀 开始测试模型 (Observer Mode): {model} [{provider}]")
    print("==================================================")

    client = build_client(provider)
    result_file = result_filename_for_model(model_config, output_dir)

    experiment_results, processed_pairs = load_existing_results(
        result_file=result_file,
        total_pairs=total_pairs,
        model_name=model,
    )

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
        prompt2_success = USER_PROMPT.format(challenger_text=success_reply["text"])
        messages_branch_a = messages_step1 + [
            {"role": "user", "content": prompt2_success}
        ]
        res_branch_a = call_llm(
            client=client,
            model_name=model,
            messages=messages_branch_a,
            max_retries=max_retries,
        )

        prompt2_failure = USER_PROMPT.format(challenger_text=failure_reply["text"])
        messages_branch_b = messages_step1 + [
            {"role": "user", "content": prompt2_failure}
        ]
        res_branch_b = call_llm(
            client=client,
            model_name=model,
            messages=messages_branch_b,
            max_retries=max_retries,
        )

        # ---------- 记录结果 ----------
        result_entry = {
            "pair_id": p_id,
            "root_id": root_id,
            "evaluation_mode": "observer",
            "provider": provider,
            "model": model,
            "branch_A_human_success": {
                "human_label": 1,
                "delta_awarded": res_branch_a.get("delta_awarded"),
                "justification": res_branch_a.get("justification"),
            },
            "branch_B_human_failure": {
                "human_label": 0,
                "delta_awarded": res_branch_b.get("delta_awarded"),
                "justification": res_branch_b.get("justification"),
            },
        }

        experiment_results.append(result_entry)
        processed_pairs.add(p_id)
        count += 1

        # 实时保存
        if count % save_every == 0:
            save_results(result_file, experiment_results)
            print(f"💾 [自动存盘] {model} 的数据已实时安全保存 ({count}/{total_pairs})")

        time.sleep(request_interval)

    # 模型跑完最后的统一保存
    save_results(result_file, experiment_results)
    print(f"\n✅ {model} 彻底测试完成！最终结果保存在 {result_file}")


def select_models(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[ModelConfig]:
    """
    支持只跑某个 provider 或某个模型。
    """
    selected = MODELS_TO_TEST

    if provider:
        selected = [m for m in selected if m.provider == provider]

    if model:
        selected = [m for m in selected if m.model == model]

    if not selected:
        available = "\n".join(f"- [{m.provider}] {m.model}" for m in MODELS_TO_TEST)
        raise ValueError(
            f"没有匹配到要测试的模型。当前可用模型：\n{available}"
        )

    return selected


def run_experiment(
    pairs_file: Path,
    ops_file: Path,
    output_dir: Path,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    save_every: int = 20,
    request_interval: float = 0.5,
    max_retries: int = 3,
) -> None:
    """
    核心实验入口。
    """
    maybe_enable_proxy()

    print("正在加载数据集...")
    pairs = load_json(pairs_file)
    ops_list = load_json(ops_file)

    ops_dict = {op["id"]: op for op in ops_list}
    total_pairs = len(pairs)
    print(f"共加载 {total_pairs} 对单轮测试数据。准备执行全量第三人称预测测试！")

    selected_models = select_models(provider=provider, model=model)
    print("本次将测试以下模型：")
    for m in selected_models:
        print(f"  - [{m.provider}] {m.model}")

    for model_config in selected_models:
        run_one_model(
            model_config=model_config,
            pairs=pairs,
            ops_dict=ops_dict,
            total_pairs=total_pairs,
            output_dir=output_dir,
            save_every=save_every,
            request_interval=request_interval,
            max_retries=max_retries,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run observer-mode delta prediction experiment across all configured models."
    )

    parser.add_argument(
        "--pairs-file",
        default=DEFAULT_PAIRS_FILE,
        help="Path to single_turn_pairs.json",
    )
    parser.add_argument(
        "--ops-file",
        default=DEFAULT_OPS_FILE,
        help="Path to 2262_unique_ops.json",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save result JSON files",
    )
    parser.add_argument(
        "--provider",
        choices=["siliconflow", "openrouter", "azure"],
        default=None,
        help="Only run models from one provider",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Only run one exact model/deployment name",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save results every N processed pairs",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.5,
        help="Sleep seconds after each pair",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for each LLM call",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_experiment(
        pairs_file=Path(args.pairs_file),
        ops_file=Path(args.ops_file),
        output_dir=Path(args.output_dir),
        provider=args.provider,
        model=args.model,
        save_every=args.save_every,
        request_interval=args.request_interval,
        max_retries=args.max_retries,
    )
