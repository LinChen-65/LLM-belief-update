import json
import os
import time
import re
from openai import OpenAI

# ==========================================
# 1. 基础配置与文件路径
# ==========================================
INPUT_FILE = '/data7/chenyitong/Winning_Arguments/extracted_error_ops.json'
OUTPUT_FILE = '/data7/chenyitong/Winning_Arguments/rewritten_error_ops.json'

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

SYSTEM_PROMPT = f"""You are an expert academic editor and content moderator.
Your task is to rewrite an online forum post (Original Post / OP) so that it safely bypasses Azure OpenAI's strict content filters, while preserving 100% of the original author's core argument, reasoning, logic, and intent.

{AZURE_POLICIES}

Guidelines for your rewrite:
- Identify any potentially triggering words (slurs, graphic violence, extreme profanity, explicit sexual terms, or aggressive hate speech).
- Replace these triggers with neutral, academic, objective, or abstract equivalents. (e.g., instead of graphic violent verbs, use "inflict severe physical harm"; instead of slurs, use "marginalized groups" or the specific group name neutrally).
- Tone down highly inflammatory or aggressive rhetoric into polite, structured discourse.
- CRITICAL: Do NOT change the stance or the argument itself. If the author is arguing a controversial opinion, keep the opinion but express it in a civilized, sterile manner.

Output your response strictly as a JSON object containing the rewritten title and text:
{{
    "rewritten_title": "<Sanitized title here>",
    "rewritten_text": "<Sanitized text here>"
}}
"""

# ==========================================
# 3. 核心调用逻辑
# ==========================================
def rewrite_content(title, text):
    user_prompt = f"Original Title:\n{title}\n\nOriginal Text:\n{text}"
    
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
            return json.loads(clean_json_str)
            
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
        
    print(f"📦 成功加载 {len(data)} 个异常案例，准备开始 DeepSeek 净化改写...")
    
    # 支持断点续传（如果中途中断，已经改写的不会重改）
    for i, item in enumerate(data):
        op_id = item.get('original_id', f"Unknown_{i}")
        
        # 如果已经改写过，跳过
        if 'rewritten_title' in item and 'rewritten_text' in item:
            continue
            
        op_data = item.get('full_op_data', {})
        original_title = op_data.get('title', '')
        original_text = op_data.get('text', '')
        
        print(f"[{i+1}/{len(data)}] 正在改写 OP ID: {op_id} ...")
        
        result = rewrite_content(original_title, original_text)
        
        if result:
            # 将改写后的结果挂载回原字典
            item['rewritten_title'] = result.get('rewritten_title', original_title)
            item['rewritten_text'] = result.get('rewritten_text', original_text)
            print("  ✅ 改写成功！")
        else:
            print("  ❌ 改写失败，保留原文本。")
            item['rewritten_title'] = original_title
            item['rewritten_text'] = original_text
            
        # 频率控制，防止触发 SiliconFlow 速率限制
        time.sleep(1)
        
        # 每处理 5 个保存一次
        if (i + 1) % 5 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
    # 最终保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 净化改写任务全部完成！")
    print(f"📂 结果已保存至: {OUTPUT_FILE}")
    print("您可以拿着这份新文件里的 rewritten_text 重新去跑 GPT-5.1 的分类实验了！")

if __name__ == "__main__":
    main()
