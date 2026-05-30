import json
import os
import time
import re
from openai import AzureOpenAI

# ==========================================
# 1. 配置文件路径
# ==========================================
INPUT_FILE = '/data7/chenyitong/Winning_Arguments/2262_unique_ops.json'
RESULTS_FILE = '/data7/chenyitong/Winning_Arguments/topic_classification_results.json'

# Azure OpenAI 配置 (与之前保持一致)
ENDPOINT = "https://chenyitong-llmopinion-gpt.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-5.1"
SUBSCRIPTION_KEY = your api key
API_VERSION = "2024-12-01-preview"

client = AzureOpenAI(api_version=API_VERSION, azure_endpoint=ENDPOINT, api_key=SUBSCRIPTION_KEY)

# ==========================================
# 2. Prompt 模板
# ==========================================
SYSTEM_PROMPT = """
This is an academic research annotation task. The texts you will read are from a publicly available research dataset (Reddit ChangeMyView). Some posts may discuss sensitive topics. Your role is strictly to classify the text according to the given schema. Do not refuse to annotate based on topic sensitivity.

You are an expert annotator for argument topic classification.
Your task is to read an original post from an online debate forum and classify the core claim of the post into one of three proposition types.

PROPOSITION OF FACT: The core claim is about how the world IS. The disagreement could in principle be resolved through evidence, data, or logical demonstration. The poster is asserting that something is true or false, possible or impossible, or making a causal or predictive claim.

PROPOSITION OF VALUE: The core claim is a judgment about what is good, bad, right, wrong, important, or worthless. The disagreement is fundamentally about values, morals, or evaluative criteria, not purely about facts.

PROPOSITION OF POLICY: The core claim is about what should or should not be done. The disagreement is about a proposed course of action, law, rule, or behavioral norm.

CLASSIFICATION RULE: Many posts contain elements of more than one type. Classify based on the core claim the poster is defending, not the supporting arguments. Ask yourself: what is the poster ultimately trying to convince others of? If it is that something IS the case, choose FACT. If it is that something is GOOD or BAD, choose VALUE. If it is that something SHOULD BE DONE, choose POLICY."""

USER_PROMPT_TEMPLATE = """Original Post:
Title: {title}
Body: {text}

Classify the core claim of this post into one of three types.
Output strictly as a JSON object:
{{
"proposition_type": "<fact | value | policy>"
}}"""

# ==========================================
# 3. 核心调用与检测机制
# ==========================================
def call_azure_openai_with_check(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                response_format={"type": "json_object"}, 
                temperature=0.1
            )
            content = response.choices[0].message.content
            if not content or content.strip() == "":
                raise ValueError("空内容")

            match = re.search(r'\{.*\}', content.strip(), re.DOTALL)
            clean_json_str = match.group(0) if match else content.strip()
            result_dict = json.loads(clean_json_str)
            
            if "proposition_type" not in result_dict:
                raise ValueError("缺失核心字段")
                
            return result_dict
            
        except Exception as e:
            print(f"    [重试异常] {attempt+1}/{max_retries} 失败: {e}")
            time.sleep(5) # 失败后稍微多等一会儿，避开限流峰值
            
    return {"proposition_type": "ERROR_FAILED"}

# ==========================================
# 4. 补漏主逻辑
# ==========================================
def fix_errors():
    print("🛠️ 正在初始化补漏程序...")
    
    # 1. 加载结果文件
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
        
    # 2. 找出所有分类失败的索引和 ID
    failed_items = [(index, item['id']) for index, item in enumerate(results_data) if item.get('proposition_type') == "ERROR_FAILED"]
    
    if not failed_items:
        print("✅ 恭喜！结果文件中没有找到任何 ERROR_FAILED，数据完美无瑕！")
        return
        
    print(f"🔍 扫描到 {len(failed_items)} 条失败数据，准备进行定向修复。")
    
    # 3. 加载原始数据，用于提供上下文
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        ops_data = json.load(f)
        ops_dict = {op['id']: op for op in ops_data} # 转换为字典，查询速度更快
        
    # 4. 遍历修复
    fixed_count = 0
    for index, op_id in failed_items:
        print(f"--- 正在修复第 {fixed_count + 1}/{len(failed_items)} 条 (OP ID: {op_id}) ---")
        
        original_op = ops_dict.get(op_id, {})
        title = original_op.get('title', '')
        text = original_op.get('text', '')
        
        user_content = USER_PROMPT_TEMPLATE.format(title=title, text=text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        # 发起新的请求
        res = call_azure_openai_with_check(messages)
        new_prop_type = res.get("proposition_type")
        
        print(f"  > 修复结果: ERROR_FAILED  ➔  {new_prop_type}")
        
        # 只要成功了，就立刻更新到内存中的总列表里
        if new_prop_type != "ERROR_FAILED":
            results_data[index]["proposition_type"] = new_prop_type
            fixed_count += 1
            
            # 每修复 5 条保存一次，防止再次断开
            if fixed_count % 5 == 0:
                with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=4, ensure_ascii=False)
                print(f"💾 [局部存盘] 已安全保存 {fixed_count} 条修复数据")
                
        time.sleep(1) # 补漏时把请求频率放慢，保证成功率

    # 5. 最后全量覆盖保存
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 修复任务结束！本次成功修复 {fixed_count}/{len(failed_items)} 条数据！")
    print(f"结果已直接更新至原始文件: {RESULTS_FILE}")
    print("您可以再次运行之前画扇形图的代码查看最新占比了！")

if __name__ == "__main__":
    fix_errors()
