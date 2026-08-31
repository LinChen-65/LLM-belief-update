import json
import os
import time
import re
from openai import AzureOpenAI

# ==========================================
# 1. 基础配置与文件路径
# ==========================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# 原帖文件（用于为大模型提供对话上下文）
OPS_FILE = os.path.join(DATA, "dataset", "2262_unique_ops.json")
# 待标注的回复对文件
PAIRS_FILE = os.path.join(DATA, "dataset", "single_turn_pairs.json")
# 标注后的最终输出文件
OUTPUT_FILE = os.path.join(DATA, "strategy", "single_turn_pairs_with_strategies.json")

# Azure OpenAI API 配置
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://chenyitong-llmopinion-gpt.openai.azure.com/")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
SUBSCRIPTION_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
)

# ==========================================
# 2. 从论文大纲提取的 Prompt 模板设计
# ==========================================
SYSTEM_PROMPT = """
This is an academic research annotation task. The texts you will read are from a publicly available research dataset (Reddit ChangeMyView). Some posts may discuss sensitive topics. Your role is strictly to classify the text according to the given schema. Do not refuse to annotate based on topic sensitivity.

You are an expert annotator for persuasion strategy classification.
Your task is to read a challenger's reply to an original post on an online forum, and determine which persuasion modes the reply employs. A single reply may use one or more of the following modes.

Definitions and examples (adapted from Hidey et al., 2017):

LOGOS: Appeals to reason through logical argument, factual evidence, relevant examples, statistics, or causal reasoning.
Examples:
- "Eating healthy makes you live longer. The oldest man in the US followed a strictly fat-free diet."
- "He will probably win the election. He is the favorite according to the polls."

PATHOS: Appeals to the audience's emotions, empathy, or sense of identification. This includes evoking fear, sympathy, moral concern, or describing scenarios the audience can personally relate to.
Examples:
- "Doctors should stop prescribing antibiotics at a large scale. The spread of antibiotics will be a threat for the next generation."
- "You should put comfy furniture into your place. The feeling of being home is unforgettable."

ETHOS: Appeals to credibility established through personal experience, professional expertise, or reference to authoritative sources.
Examples:
- "I assure you the consequences of fracking are terrible. I have been living next to a pipeline since I was a child."
- "I trust his predictions about climate change. He is a Nobel Prize winner."

Note: A reply may combine multiple modes. For example, a reply that cites statistics (logos) while also sharing a personal story to evoke empathy (pathos) should be labeled as both logos and pathos."""

USER_PROMPT_TEMPLATE = """Original Post:
Title: {title}
Body: {text}

Challenger's Reply:
{challenger_text}

For each persuasion mode, determine whether the challenger's reply employs it.
Output strictly as a JSON object:
{{
"logos": <true or false>,
"pathos": <true or false>,
"ethos": <true or false>
}}"""

# ==========================================
# 3. 辅助函数：安全调用与多重校验
# ==========================================
def call_azure_openai_with_check(messages, max_retries=3):
    """调用大模型，并强制解析二元多标签的 JSON 格式"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                response_format={"type": "json_object"}, 
                temperature=0.1,  # 标注任务保持低温度
                max_completion_tokens=500
            )
            content = response.choices[0].message.content
            
            if not content or content.strip() == "":
                raise ValueError("API 返回了空内容。")

            # 正则提取 JSON
            match = re.search(r'\{.*\}', content.strip(), re.DOTALL)
            clean_json_str = match.group(0) if match else content.strip()
            result_dict = json.loads(clean_json_str)
            
            # 字段校验
            if not all(k in result_dict for k in ("logos", "pathos", "ethos")):
                raise ValueError(f"JSON 缺失关键字段。当前返回: {result_dict}")
                
            return result_dict
            
        except Exception as e:
            print(f"  [API/解析异常] 尝试 {attempt+1}/{max_retries} 失败: {e}")
            time.sleep(3)
            
    # 彻底失败的兜底返回值
    return {"logos": None, "pathos": None, "ethos": None, "error": "FAILED"}

# ==========================================
# 4. 主运行逻辑 (支持断点续传)
# ==========================================
def main():
    print("正在加载数据集...")
    
    # 1. 加载 OP 原帖数据 (用于提供上下文)
    if not os.path.exists(OPS_FILE):
        print(f"❌ 找不到原帖文件: {OPS_FILE}")
        return
    with open(OPS_FILE, 'r', encoding='utf-8') as f:
        ops_list = json.load(f)
        # 转为字典以便 O(1) 快速查找
        ops_dict = {op['id']: op for op in ops_list}
    print(f"✅ 成功加载 {len(ops_dict)} 条原帖数据。")

    # 2. 加载 Pairs 回复数据
    if not os.path.exists(PAIRS_FILE):
        print(f"❌ 找不到回复对文件: {PAIRS_FILE}")
        return
    with open(PAIRS_FILE, 'r', encoding='utf-8') as f:
        pairs_data = json.load(f)
    total_pairs = len(pairs_data)
    print(f"✅ 成功加载 {total_pairs} 对回复数据 (共 {total_pairs * 2} 条回复)。")

    # 3. 断点续传初始化
    annotated_pairs = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                annotated_pairs = json.load(f)
            print(f"📦 发现本地存档，已完成 {len(annotated_pairs)}/{total_pairs} 对，自动跳过。")
        except Exception as e:
            print(f"⚠️ 读取存档失败，将从头开始。错误: {e}")

    count = len(annotated_pairs)
    
    # 4. 遍历所有对决 (Pairs)
    for pair_id, branches in pairs_data.items():
        if pair_id in annotated_pairs:
            continue
            
        print(f"\n--- 进度: {count+1}/{total_pairs} (Pair ID: {pair_id}) ---")
        
        # 深拷贝当前 Pair，确保保留原始的所有嵌套信息
        current_pair_result = json.loads(json.dumps(branches)) 
        
        # 分别处理 success(被说服) 和 failure(未被说服) 两条回复
        for branch_type in ['success', 'failure']:
            reply_data = branches.get(branch_type)
            if not reply_data:
                continue
                
            root_id = reply_data.get('root')
            challenger_text = reply_data.get('text', '')
            
            # 获取原帖上下文
            op_data = ops_dict.get(root_id, {})
            title = op_data.get('title', 'N/A')
            text = op_data.get('text', 'N/A')
            
            # 组装 Prompt
            user_content = USER_PROMPT_TEMPLATE.format(
                title=title, 
                text=text, 
                challenger_text=challenger_text
            )
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            # 调用 GPT-5.1 进行标注
            print(f"  > 正在标注 [{branch_type}] 回复...")
            strategy_res = call_azure_openai_with_check(messages)
            print(f"    - 结果: {strategy_res}")
            
            # 【核心】：将大模型的策略打标结果挂载回原数据结构中
            current_pair_result[branch_type]["persuasion_strategies"] = strategy_res
            
            # 防止 API 并发过高限流
            time.sleep(0.3)
            
        # 将完成标注的 pair 存入总字典
        annotated_pairs[pair_id] = current_pair_result
        count += 1
        
        # 每处理 10 对 (即 20 条回复) 自动存盘一次
        if count % 10 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(annotated_pairs, f, indent=4, ensure_ascii=False)
            print(f"💾 [自动存盘] 进度已安全保存 ({count}/{total_pairs})")

    # 全量保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(annotated_pairs, f, indent=4, ensure_ascii=False)
    print(f"\n🎉 全部 {total_pairs} 对回复的论证策略标注已完成！完整数据保存在: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
