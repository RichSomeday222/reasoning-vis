import os
import openai

openai.api_key = os.getenv("DEEPSEEK_API_KEY")
# 如果需要指定 custom base URL（按 GR.inc 文档）
openai.api_base = "https://api.deepseek.com/v1"

resp = openai.completions.create(
    model="DeepSeek-R1",
    prompt="1+1 等于多少？请一步步解释。",
    max_tokens=50,
    temperature=0.5,
)
print(resp.choices[0].text)
