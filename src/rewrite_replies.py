import json
import os
import time
import re
from openai import OpenAI

# ==========================================
# 1. 基础配置与文件路径
# ==========================================
INPUT_FILE = '/data7/chenyitong/Winning_Arguments/non_true_strategy_cases.json'
OUTPUT_FILE = '/data7/chenyitong/Winning_Arguments/rewritten_non_true_strategy_cases.json'

API_KEY = "sk-bzsitegoadikvrvitcrnsjqkeppmdotnbuvzrsupgsdzxdnz"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# ==========================================
# 2. 深入理解 Azure 过滤规则的 Prompt 模板
# ==========================================
AZURE_POLICIES = """
Azure OpenAI Content Management Policies strictly filter content in four main categories:
1. Hate Speech: Language that attacks, insults, or demeans groups based on race, ethnicity, religion, gender, sexual orientation, etc.
2. Sexual: Explicit descriptions of sexual acts, organs, or highly inappropriate sexual references.
3. Violence: Graphic, bloody descriptions of physical violence, abuse, weapons, or terrorism.
4. Self-Harm: Promotion or detailed description of suicide, cutting, or eating disorders.
(Additionally, excessive profanity or aggressive slurs can trigger the filters).
"""

# 【修改点 1】：在 Prompt 末尾增加了严格的 JSON 规则，禁止直接换行
SYSTEM_PROMPT = f"""You are an expert academic editor and content moderator.
Your task is to rewrite a challenger's reply from an online forum so that it safely bypasses Azure OpenAI's strict content filters, while preserving 100% of the original author's core argument, reasoning, logic, persuasion strategy, and intent.

{AZURE_POLICIES}

Guidelines for your rewrite:
- Identify any potentially triggering words (slurs, graphic violence, extreme profanity, explicit sexual terms, or aggressive hate speech).
- Replace these triggers with neutral, academic, objective, or abstract equivalents.
- Tone down highly inflammatory or aggressive rhetoric into polite, structured discourse.
- CRITICAL: Do NOT change the stance or the argument itself.

JSON RULES:
1. Output strictly as a JSON object.
2. Escape all internal double quotes with a backslash (`\"`).
3. Use `\\n` instead of actual line breaks inside the string. DO NOT use raw unescaped newlines.

Output Format:
{{
    "rewritten_text": "<Sanitized text here>"
}}
"""

# ==========================================
# 3. 核心调用逻辑
# ==========================================
def rewrite_reply(text):
    user_prompt = f"Original Reply Text:\n{text}"
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2  # 保持较低温度，确保不随意发挥
            )
            content = response.choices[0].message.content.strip()
            
            # 正则提取 JSON
            match = re.search(r'\{.*\}', content, re.DOTALL)
            clean_json_str = match.group(0) if match else content
            
            # 【修改点 2】：加上 strict=False，允许 JSON 字符串中包含真实的换行符或控制字符
            return json.loads(clean_json_str, strict=False)
            
        except Exception as e:
            print(f"  [API Error] 尝试 {attempt+1}/3 失败: {e}")
            time.sleep(3)
            
    return None

# ==========================================
# 4. 主程序运行
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"📦 成功加载 {len(data)} 对 (Pairs) 异常案例，准备开始 DeepSeek 净化改写...")
    
    pair_count = 0
    rewritten_branch_count = 0
    
    # 支持断点续传（如果中途中断，已经改写的不会重改）
    for pair_id, branches in data.items():
        pair_count += 1
        
        # 分别检查 success 和 failure 两个分支
        for branch_name in ['success', 'failure']:
            if branch_name not in branches:
                continue
                
            branch_data = branches[branch_name]
            
            # 1. 检查该分支是否是我们需要改写的异常分支 (没有任意一个策略为 True)
            strategies = branch_data.get('persuasion_strategies', {})
            has_any_true = (
                strategies.get('logos') is True or 
                strategies.get('pathos') is True or 
                strategies.get('ethos') is True
            )
            
            # 如果该分支是正常的（包含 True），我们就跳过它
            if has_any_true:
                continue
                
            # 2. 如果已经改写过了（断点续传），跳过
            if 'rewritten_text' in branch_data:
                continue
                
            # 3. 开始执行改写
            original_text = branch_data.get('text', '')
            reply_id = branch_data.get('id', 'Unknown')
            
            print(f"[{pair_count}/{len(data)}] 正在改写 Pair: {pair_id} -> Branch: [{branch_name}] (Reply ID: {reply_id})")
            
            result = rewrite_reply(original_text)
            
            if result and 'rewritten_text' in result:
                # 将改写后的结果挂载回原字典的新字段中
                branch_data['rewritten_text'] = result['rewritten_text']
                print("  ✅ 改写成功！")
            else:
                print("  ❌ 改写失败，保留原文本。")
                branch_data['rewritten_text'] = original_text
                
            rewritten_branch_count += 1
            
            # 频率控制，防止触发 SiliconFlow 速率限制
            time.sleep(1)
            
        # 每处理 5 个 Pair 保存一次 (实时存盘)
        if pair_count % 5 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
    # 最终全量保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 净化改写任务全部完成！")
    print(f"📊 本次共计对 {rewritten_branch_count} 个异常回复分支进行了清洗。")
    print(f"📂 结果已保存至: {OUTPUT_FILE}")
    print("您可以拿着这份新文件里的 rewritten_text 重新去跑分类实验了！")

if __name__ == "__main__":
    main()
