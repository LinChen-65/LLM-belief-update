import json
import os
import time
import re
from openai import AzureOpenAI

# ==========================================
# 1. 配置文件路径
# ==========================================
# OP 原帖数据（用于提供上下文）
OPS_FILE = '/data7/chenyitong/Winning_Arguments/2262_unique_ops.json'
# 包含策略标注结果的文件（需要被修补的文件）
RESULTS_FILE = '/data7/chenyitong/Winning_Arguments/single_turn_pairs_with_strategies.json'

# Azure OpenAI 配置
ENDPOINT = "https://chenyitong-llmopinion-gpt.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-5.1"
SUBSCRIPTION_KEY = your api key
API_VERSION = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=API_VERSION, 
    azure_endpoint=ENDPOINT, 
    api_key=SUBSCRIPTION_KEY
)

# ==========================================
# 2. Prompt 模板 (与策略标注脚本保持一致)
# ==========================================
SYSTEM_PROMPT = """You are an expert annotator for persuasion strategy classification.
Your task is to read a challenger's reply to an original post on an online forum, and determine which persuasion modes the reply employs. A single reply may use one or more of the following modes.

Definitions and examples (adapted from Hidey et al., 2017):

LOGOS: Appeals to reason through logical argument, factual evidence, relevant examples, statistics, or causal reasoning.
PATHOS: Appeals to the audience's emotions, empathy, or sense of identification. This includes evoking fear, sympathy, moral concern, or describing scenarios the audience can personally relate to.
ETHOS: Appeals to credibility established through personal experience, professional expertise, or reference to authoritative sources.

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
# 3. 核心调用与检测机制 (增强容错)
# ==========================================
def call_azure_openai_with_check(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                response_format={"type": "json_object"}, 
                temperature=0.1,
                max_completion_tokens=500
            )
            content = response.choices[0].message.content
            
            if not content or content.strip() == "":
                raise ValueError("API 返回了空内容。")

            # 正则提取 JSON
            match = re.search(r'\{.*\}', content.strip(), re.DOTALL)
            clean_json_str = match.group(0) if match else content.strip()
            result_dict = json.loads(clean_json_str)
            
            # 严格校验是否包含三个策略字段
            if not all(k in result_dict for k in ("logos", "pathos", "ethos")):
                raise ValueError(f"JSON 缺失关键字段。当前返回: {result_dict}")
                
            return result_dict
            
        except Exception as e:
            print(f"    [重试异常] {attempt+1}/{max_retries} 失败: {e}")
            time.sleep(5)  # 失败后休眠 5 秒，避开 Azure 峰值限流
            
    # 彻底失败的兜底返回值
    return {"error": "FAILED"}

# ==========================================
# 4. 补漏主逻辑
# ==========================================
def fix_errors():
    print("🛠️ 正在初始化策略标注补漏程序...")
    
    # 1. 加载已被标注的结果文件
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ 找不到结果文件: {RESULTS_FILE}")
        return
        
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 2. 精准定位所有分类失败的节点
    # 我们需要记录 failed_items = [(pair_id, branch_type), ...]
    failed_items = []
    for pair_id, branches in data.items():
        for branch_type in ['success', 'failure']:
            reply_data = branches.get(branch_type)
            if not reply_data:
                continue
            
            strategies = reply_data.get('persuasion_strategies', {})
            # 判断是否失败：没有策略字典，或者包含 error，或者缺字段
            if not strategies or strategies.get('error') or 'logos' not in strategies:
                failed_items.append((pair_id, branch_type))
                
    if not failed_items:
        print("✅ 恭喜！扫描完毕，结果文件中没有找到任何异常数据，数据完美无瑕！")
        return
        
    print(f"🔍 扫描到 {len(failed_items)} 条失败或异常数据，准备进行定向修复。")
    
    # 3. 加载原帖数据，用于构建上下文
    with open(OPS_FILE, 'r', encoding='utf-8') as f:
        ops_data = json.load(f)
        ops_dict = {op['id']: op for op in ops_data} # 转换为字典 O(1) 查询
        
    # 4. 开始遍历修复
    fixed_count = 0
    total_to_fix = len(failed_items)
    
    for pair_id, branch_type in failed_items:
        print(f"--- 正在修复第 {fixed_count + 1}/{total_to_fix} 条 (Pair ID: {pair_id} | 分支: {branch_type}) ---")
        
        reply_data = data[pair_id][branch_type]
        root_id = reply_data.get('root')
        challenger_text = reply_data.get('text', '')
        
        op_data = ops_dict.get(root_id, {})
        title = op_data.get('title', 'N/A')
        text = op_data.get('text', 'N/A')
        
        user_content = USER_PROMPT_TEMPLATE.format(
            title=title, 
            text=text, 
            challenger_text=challenger_text
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        # 发起新的补漏请求
        new_strategy = call_azure_openai_with_check(messages)
        
        # 如果成功（没有 error 字段）
        if "error" not in new_strategy:
            print(f"  > 修复成功: ➔ {new_strategy}")
            # 原地更新内存中的字典
            data[pair_id][branch_type]['persuasion_strategies'] = new_strategy
            fixed_count += 1
            
            # 每修复 5 条局部存盘一次，防止再次断联
            if fixed_count % 5 == 0:
                with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"💾 [局部存盘] 已安全保存 {fixed_count} 条修复数据")
        else:
            print(f"  > ❌ 再次修复失败，保留原状。")
            
        time.sleep(1) # 正常请求间隔 1 秒，保证重试的稳定性

    # 5. 循环结束，全量覆盖保存
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 修复任务结束！本次成功修复 {fixed_count}/{total_to_fix} 条数据！")
    print(f"结果已原地更新至原始文件: {RESULTS_FILE}")
    print("您可以再次运行画饼图的代码，145 这个失败数字应该会归零了！")

if __name__ == "__main__":
    fix_errors()
