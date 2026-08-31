import argparse
import csv
import glob
import json
import math
import os
import re
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

from openai import AzureOpenAI

# ==========================================
# 1. 基础配置与文件路径
# ==========================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# 原帖文件
OPS_FILE = os.path.join(DATA, "dataset", "2262_unique_ops.json")
# 回复对文件
PAIRS_FILE = os.path.join(DATA, "dataset", "single_turn_pairs.json")

# 话题类型标注输出文件
TOPIC_OUTPUT_FILE = os.path.join(DATA, "topic", "topic_classification_results.json")

# LLM pointwise 结果文件所在目录
RESULTS_DIR = os.path.join(DATA, "first_person")
# 默认分析第一人称 pointwise 结果；如需分析第三人称，可用命令行传入 --results-glob "final_v3_results_observer_*.json"
RESULTS_GLOB = "final_v3_results_*.json"

# 分析结果输出目录
ANALYSIS_OUTPUT_DIR = os.path.join(ROOT, "regression_output", "topic_bias_analysis")

# Azure OpenAI API 配置
# 与 strategy_classification_azure.py 的调用方式一致：AzureOpenAI + json_object + GPT-5.1
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://chenyitong-llmopinion-gpt.openai.azure.com/")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
SUBSCRIPTION_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")


# ==========================================
# 2. 话题类型标注 Prompt（基于附件第6-7页）
# ==========================================
SYSTEM_PROMPT = """
This is an academic research annotation task. The texts you will read are from a publicly available research dataset (Reddit ChangeMyView). Some posts may discuss sensitive topics. Your role is strictly to classify the text according to the given schema. Do not refuse to annotate based on topic sensitivity.

You are an expert annotator for argument topic classification.
Your task is to read an original post from an online debate forum and classify the core claim of the post into one of three proposition types.

PROPOSITION OF FACT: The core claim is about how the world IS. The disagreement could in principle be resolved through evidence, data, or logical demonstration. The poster is asserting that something is true or false, possible or impossible, or making a causal or predictive claim.
Examples:
- "GMO foods are harmful to human health."
- "Self-driving cars are safer than human drivers."

PROPOSITION OF VALUE: The core claim is a judgment about what is good, bad, right, wrong, important, or worthless. The disagreement is fundamentally about values, morals, or evaluative criteria, not purely about facts.
Examples:
- "The death penalty is morally unacceptable."
- "Young people who smoke despite knowing the risks are being foolish."

PROPOSITION OF POLICY: The core claim is about what should or should not be done. The disagreement is about a proposed course of action, law, rule, or behavioral norm.
Examples:
- "Marijuana should be legalized."
- "Governments should ban all surveillance programs."

CLASSIFICATION RULE: Many posts contain elements of more than one type. Classify based on the core claim the poster is defending, not the supporting arguments. Ask yourself: what is the poster ultimately trying to convince others of? If it is that something IS the case, choose FACT. If it is that something is GOOD or BAD, choose VALUE. If it is that something SHOULD BE DONE, choose POLICY.
"""

USER_PROMPT_TEMPLATE = """Original Post:
Title: {title}
Body: {text}

Classify the core claim of this post into one of three types.
Output strictly as a JSON object:
{{
"proposition_type": "<fact | value | policy>"
}}"""


# ==========================================
# 3. Azure 调用与 JSON 校验
# ==========================================
def build_client() -> AzureOpenAI:
    if not SUBSCRIPTION_KEY:
        raise RuntimeError(
            "未检测到 AZURE_OPENAI_API_KEY。请先执行：\n"
            "export AZURE_OPENAI_API_KEY='你的 Azure API Key'\n"
            "或在 Windows PowerShell 中执行：\n"
            "$env:AZURE_OPENAI_API_KEY='你的 Azure API Key'"
        )

    return AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=SUBSCRIPTION_KEY,
    )


def extract_json_dict(content: str) -> Dict[str, Any]:
    if not content or content.strip() == "":
        raise ValueError("API 返回了空内容。")

    match = re.search(r"\{.*\}", content.strip(), re.DOTALL)
    clean_json_str = match.group(0) if match else content.strip()
    return json.loads(clean_json_str)


def call_azure_openai_with_check(client: AzureOpenAI, messages: List[Dict[str, str]], max_retries: int = 3) -> Dict[str, Any]:
    """调用 GPT-5.1 标注话题类型，并强制解析为 {"proposition_type": "..."}。"""
    valid_labels = {"fact", "value", "policy"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_completion_tokens=300,
            )
            content = response.choices[0].message.content
            result_dict = extract_json_dict(content)

            if "proposition_type" not in result_dict:
                raise ValueError(f"JSON 缺失 proposition_type 字段。当前返回: {result_dict}")

            label = str(result_dict["proposition_type"]).strip().lower()
            if label not in valid_labels:
                raise ValueError(f"proposition_type 必须是 fact/value/policy 之一。当前返回: {result_dict}")

            return {"proposition_type": label}

        except Exception as e:
            print(f"  [API/解析异常] 尝试 {attempt + 1}/{max_retries} 失败: {e}")
            time.sleep(3)

    return {"proposition_type": None, "error": "FAILED"}


# ==========================================
# 4. 话题类型标注：整条原帖级别，单标签三选一
# ==========================================
def annotate_topics(
    ops_file: str = OPS_FILE,
    pairs_file: str = PAIRS_FILE,
    output_file: str = TOPIC_OUTPUT_FILE,
    sleep_seconds: float = 0.3,
    save_every: int = 20,
) -> None:
    print("正在加载数据集...")

    if not os.path.exists(ops_file):
        print(f"❌ 找不到原帖文件: {ops_file}")
        return
    with open(ops_file, "r", encoding="utf-8") as f:
        ops_list = json.load(f)
    ops_dict = {op["id"]: op for op in ops_list}
    print(f"✅ 成功加载 {len(ops_dict)} 条原帖数据。")

    # 只标注 single_turn_pairs 中实际出现的 root_id，避免标注无关 OP
    target_root_ids = set(ops_dict.keys())
    if pairs_file and os.path.exists(pairs_file):
        with open(pairs_file, "r", encoding="utf-8") as f:
            pairs_data = json.load(f)
        target_root_ids = set()
        for _, branches in pairs_data.items():
            for branch_type in ("success", "failure"):
                reply_data = branches.get(branch_type, {})
                root_id = reply_data.get("root")
                if root_id:
                    target_root_ids.add(root_id)
        print(f"✅ 从 pairs 文件中识别出 {len(target_root_ids)} 个需要标注的话题 root_id。")
    else:
        print("⚠️ 未找到 pairs 文件，将标注 ops_file 中全部原帖。")

    annotated_ops = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # 兼容 list 或 dict 两种存档结构
            if isinstance(loaded, list):
                annotated_ops = {op["id"]: op for op in loaded if "id" in op}
            elif isinstance(loaded, dict):
                annotated_ops = loaded
            print(f"📦 发现本地存档，已完成 {len(annotated_ops)}/{len(target_root_ids)} 条，自动跳过。")
        except Exception as e:
            print(f"⚠️ 读取存档失败，将从头开始。错误: {e}")

    client = build_client()
    count = sum(1 for rid in target_root_ids if rid in annotated_ops and annotated_ops[rid].get("proposition_type"))

    for root_id in sorted(target_root_ids):
        if root_id in annotated_ops and annotated_ops[root_id].get("proposition_type"):
            continue

        op_data = ops_dict.get(root_id)
        if not op_data:
            continue

        print(f"\n--- 进度: {count + 1}/{len(target_root_ids)} (Root ID: {root_id}) ---")

        user_content = USER_PROMPT_TEMPLATE.format(
            title=op_data.get("title", "N/A"),
            text=op_data.get("text", "N/A"),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        print("  > 正在标注原帖 proposition_type...")
        topic_res = call_azure_openai_with_check(client, messages)
        print(f"    - 结果: {topic_res}")

        current_op_result = json.loads(json.dumps(op_data))
        current_op_result["proposition_type"] = topic_res.get("proposition_type")
        if topic_res.get("error"):
            current_op_result["proposition_type_error"] = topic_res.get("error")

        annotated_ops[root_id] = current_op_result
        count += 1

        if count % save_every == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(annotated_ops, f, indent=4, ensure_ascii=False)
            print(f"💾 [自动存盘] 进度已安全保存 ({count}/{len(target_root_ids)})")

        time.sleep(sleep_seconds)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotated_ops, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 话题类型标注完成！完整数据保存在: {output_file}")


# ==========================================
# 5. 偏差分析工具函数
# ==========================================
def safe_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return None


def get_agent_delta(branch_result: Dict[str, Any]) -> Optional[bool]:
    """兼容第一人称结果中的 agent_delta 和第三人称结果中的 delta_awarded。"""
    if "agent_delta" in branch_result:
        return safe_bool(branch_result.get("agent_delta"))
    if "delta_awarded" in branch_result:
        return safe_bool(branch_result.get("delta_awarded"))
    return None


def cohen_kappa(tp: int, fp: int, tn: int, fn: int) -> Optional[float]:
    n = tp + fp + tn + fn
    if n == 0:
        return None

    po = (tp + tn) / n

    human_pos = tp + fn
    human_neg = fp + tn
    agent_pos = tp + fp
    agent_neg = fn + tn

    pe = ((human_pos / n) * (agent_pos / n)) + ((human_neg / n) * (agent_neg / n))
    if abs(1 - pe) < 1e-12:
        return None

    return (po - pe) / (1 - pe)


def safe_rate(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def format_float(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x:.6f}"


def infer_model_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name = re.sub(r"\.json$", "", base)
    name = re.sub(r"^final_v3_results_observer_", "", name)
    name = re.sub(r"^final_v3_results_", "", name)
    name = re.sub(r"^final_new_results_", "", name)
    name = re.sub(r"_v3$", "", name)
    return name


def load_topic_labels(topic_file: str) -> Dict[str, str]:
    with open(topic_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.values()
    else:
        raise ValueError(f"不支持的话题标注文件格式: {type(data)}")

    topic_by_root = {}
    for op in items:
        root_id = op.get("id")
        label = op.get("proposition_type")
        if root_id and label in {"fact", "value", "policy"}:
            topic_by_root[root_id] = label

    return topic_by_root


def collect_records_from_result_file(result_file: str, topic_by_root: Dict[str, str]) -> List[Dict[str, Any]]:
    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    model_name = infer_model_name_from_path(result_file)
    records = []

    for item in results:
        pair_id = item.get("pair_id")
        root_id = item.get("root_id")
        topic = topic_by_root.get(root_id)
        if topic not in {"fact", "value", "policy"}:
            continue

        branch_a = item.get("branch_A_human_success", {})
        pred_a = get_agent_delta(branch_a)
        if pred_a is not None:
            records.append({
                "model": model_name,
                "result_file": os.path.basename(result_file),
                "pair_id": pair_id,
                "root_id": root_id,
                "proposition_type": topic,
                "branch": "success",
                "human_label": 1,
                "agent_delta": int(pred_a),
            })

        branch_b = item.get("branch_B_human_failure", {})
        pred_b = get_agent_delta(branch_b)
        if pred_b is not None:
            records.append({
                "model": model_name,
                "result_file": os.path.basename(result_file),
                "pair_id": pair_id,
                "root_id": root_id,
                "proposition_type": topic,
                "branch": "failure",
                "human_label": 0,
                "agent_delta": int(pred_b),
            })

    return records


def summarize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["model"], r["proposition_type"])].append(r)

    rows = []
    for (model, topic), group in sorted(grouped.items()):
        tp = fp = tn = fn = 0
        for r in group:
            human = int(r["human_label"])
            agent = int(r["agent_delta"])
            if human == 1 and agent == 1:
                tp += 1
            elif human == 1 and agent == 0:
                fn += 1
            elif human == 0 and agent == 1:
                fp += 1
            elif human == 0 and agent == 0:
                tn += 1

        n = tp + fp + tn + fn
        human_success_rate = safe_rate(tp + fn, n)
        llm_delta_rate = safe_rate(tp + fp, n)
        bias_gap = None
        abs_bias_gap = None
        if human_success_rate is not None and llm_delta_rate is not None:
            bias_gap = llm_delta_rate - human_success_rate
            abs_bias_gap = abs(bias_gap)

        rows.append({
            "model": model,
            "proposition_type": topic,
            "n": n,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": safe_rate(tp + tn, n),
            "cohen_kappa": cohen_kappa(tp, fp, tn, fn),
            "tpr": safe_rate(tp, tp + fn),
            "fnr": safe_rate(fn, tp + fn),
            "fpr": safe_rate(fp, fp + tn),
            "tnr": safe_rate(tn, fp + tn),
            "human_success_rate": human_success_rate,
            "llm_delta_rate": llm_delta_rate,
            "bias_gap_llm_minus_human": bias_gap,
            "abs_bias_gap": abs_bias_gap,
        })

    return rows


def summarize_across_models(model_topic_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in model_topic_rows:
        grouped[row["proposition_type"]].append(row)

    metric_names = [
        "accuracy", "cohen_kappa", "tpr", "fnr", "fpr", "human_success_rate",
        "llm_delta_rate", "bias_gap_llm_minus_human", "abs_bias_gap"
    ]

    out = []
    for topic, rows in sorted(grouped.items()):
        summary = {"proposition_type": topic, "num_models": len(rows)}
        for m in metric_names:
            vals = [r[m] for r in rows if r.get(m) is not None]
            if vals:
                summary[f"mean_{m}"] = sum(vals) / len(vals)
                if len(vals) > 1:
                    mean = summary[f"mean_{m}"]
                    summary[f"sd_{m}"] = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
                else:
                    summary[f"sd_{m}"] = None
            else:
                summary[f"mean_{m}"] = None
                summary[f"sd_{m}"] = None
        out.append(summary)

    return out


def chi_square_tests(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """检验 topic type 与 FN/FP 是否有关。若 scipy 不可用，则只输出列联表，不输出 p 值。"""
    try:
        from scipy.stats import chi2_contingency  # type: ignore
    except Exception:
        chi2_contingency = None

    tests = []
    grouped_models = sorted(set(r["model"] for r in records))

    for model in grouped_models:
        model_records = [r for r in records if r["model"] == model]

        # FN率差异：只看 human_label=1 的 success 分支，比较 FN vs TP
        table_fn = []
        for topic in ("fact", "value", "policy"):
            sub = [r for r in model_records if r["proposition_type"] == topic and r["human_label"] == 1]
            fn = sum(1 for r in sub if r["agent_delta"] == 0)
            tp = sum(1 for r in sub if r["agent_delta"] == 1)
            table_fn.append([fn, tp])

        tests.append(run_chi_square_or_table(model, "fn_rate_by_topic", table_fn, chi2_contingency))

        # FPR差异：只看 human_label=0 的 failure 分支，比较 FP vs TN
        table_fp = []
        for topic in ("fact", "value", "policy"):
            sub = [r for r in model_records if r["proposition_type"] == topic and r["human_label"] == 0]
            fp = sum(1 for r in sub if r["agent_delta"] == 1)
            tn = sum(1 for r in sub if r["agent_delta"] == 0)
            table_fp.append([fp, tn])

        tests.append(run_chi_square_or_table(model, "fpr_by_topic", table_fp, chi2_contingency))

    return tests


def run_chi_square_or_table(model: str, test_name: str, table: List[List[int]], chi2_contingency_func) -> Dict[str, Any]:
    row = {
        "model": model,
        "test": test_name,
        "table_rows": "fact,value,policy",
        "table_cols": "FN,TP" if test_name == "fn_rate_by_topic" else "FP,TN",
        "table_json": json.dumps(table, ensure_ascii=False),
        "chi2": None,
        "p_value": None,
        "dof": None,
    }

    # 如果某一行全0，scipy会报错；这种情况下样本不足，不做显著性检验
    if any(sum(x) == 0 for x in table):
        row["note"] = "某些话题类别样本量为0，未做卡方检验。"
        return row

    if chi2_contingency_func is None:
        row["note"] = "当前环境未安装 scipy，仅输出列联表。"
        return row

    try:
        chi2, p, dof, _ = chi2_contingency_func(table)
        row["chi2"] = float(chi2)
        row["p_value"] = float(p)
        row["dof"] = int(dof)
        row["note"] = ""
    except Exception as e:
        row["note"] = f"卡方检验失败: {e}"

    return row


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    # 保持所有字段
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for k, v in row.items():
                if isinstance(v, float):
                    cleaned[k] = format_float(v)
                else:
                    cleaned[k] = v
            writer.writerow(cleaned)


def analyze_topic_bias(
    topic_file: str = TOPIC_OUTPUT_FILE,
    results_dir: str = RESULTS_DIR,
    results_glob: str = RESULTS_GLOB,
    output_dir: str = ANALYSIS_OUTPUT_DIR,
    include_observer: bool = False,
) -> None:
    print("正在加载话题类型标注...")
    if not os.path.exists(topic_file):
        print(f"❌ 找不到话题标注文件: {topic_file}")
        print("请先运行：python topic_bias_experiment.py --mode annotate")
        return

    topic_by_root = load_topic_labels(topic_file)
    print(f"✅ 成功加载 {len(topic_by_root)} 条 proposition_type 标注。")

    result_files = sorted(glob.glob(os.path.join(results_dir, results_glob)))
    if not include_observer:
        result_files = [p for p in result_files if "observer" not in os.path.basename(p)]

    if not result_files:
        print(f"❌ 没有找到结果文件: {os.path.join(results_dir, results_glob)}")
        print("请确认 --results-dir 和 --results-glob 是否正确。")
        return

    print(f"✅ 找到 {len(result_files)} 个 LLM pointwise 结果文件。")

    all_records = []
    for result_file in result_files:
        print(f"  > 读取结果文件: {result_file}")
        file_records = collect_records_from_result_file(result_file, topic_by_root)
        print(f"    - 可分析记录数: {len(file_records)}")
        all_records.extend(file_records)

    if not all_records:
        print("❌ 没有可分析记录。请检查 root_id 是否能与话题标注文件匹配。")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 保存逐条长表，方便后续 R / Python 进一步建模
    long_path = os.path.join(output_dir, "topic_bias_long_records.csv")
    write_csv(long_path, all_records)

    # 分模型、分话题汇总
    model_topic_rows = summarize_records(all_records)
    summary_path = os.path.join(output_dir, "topic_bias_by_model_and_type.csv")
    write_csv(summary_path, model_topic_rows)

    # 跨模型均值
    across_rows = summarize_across_models(model_topic_rows)
    across_path = os.path.join(output_dir, "topic_bias_across_models_mean.csv")
    write_csv(across_path, across_rows)

    # 显著性检验：话题类型是否影响 FN率 / FPR
    test_rows = chi_square_tests(all_records)
    tests_path = os.path.join(output_dir, "topic_bias_chi_square_tests.csv")
    write_csv(tests_path, test_rows)

    print("\n🎉 话题类型偏差分析完成！输出文件：")
    print(f"1. 逐条长表: {long_path}")
    print(f"2. 分模型分话题指标: {summary_path}")
    print(f"3. 跨模型均值汇总: {across_path}")
    print(f"4. 话题类型显著性检验: {tests_path}")

    print("\n重点阅读指标：")
    print("- cohen_kappa：LLM 与人类在该话题类型内的一致性，越低偏差越大")
    print("- fnr：人类认为成功说服但 LLM 判为未说服的比例，越高表示 LLM 越保守")
    print("- fpr：人类认为未成功说服但 LLM 判为成功说服的比例")
    print("- bias_gap_llm_minus_human：LLM delta 率 - 人类 delta 率，负值表示 LLM 比人类更少授予 delta")


# ==========================================
# 6. 命令行入口
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="话题类型标注 + LLM/人类偏差分组分析实验")

    parser.add_argument("--mode", choices=["annotate", "analyze", "all"], default="all",
                        help="annotate=只标注话题类型；analyze=只做偏差分析；all=先标注再分析")
    parser.add_argument("--ops-file", default=OPS_FILE)
    parser.add_argument("--pairs-file", default=PAIRS_FILE)
    parser.add_argument("--topic-output-file", default=TOPIC_OUTPUT_FILE)

    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--results-glob", default=RESULTS_GLOB)
    parser.add_argument("--analysis-output-dir", default=ANALYSIS_OUTPUT_DIR)
    parser.add_argument("--include-observer", action="store_true",
                        help="默认排除 observer 结果；打开后允许分析 observer 结果。")

    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--save-every", type=int, default=20)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in {"annotate", "all"}:
        annotate_topics(
            ops_file=args.ops_file,
            pairs_file=args.pairs_file,
            output_file=args.topic_output_file,
            sleep_seconds=args.sleep_seconds,
            save_every=args.save_every,
        )

    if args.mode in {"analyze", "all"}:
        analyze_topic_bias(
            topic_file=args.topic_output_file,
            results_dir=args.results_dir,
            results_glob=args.results_glob,
            output_dir=args.analysis_output_dir,
            include_observer=args.include_observer,
        )


if __name__ == "__main__":
    main()
