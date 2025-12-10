import json

input_file = "context.jsonl"
output_file = "batch_input.jsonl"

model = "gpt-4.1-mini"  # เปลี่ยนได้ตามต้องการ
system_prompt = "Translate this to Thai."  # เปลี่ยนได้ตามต้องการ

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for i, line in enumerate(fin, start=1):
        record = json.loads(line.strip())
        
        prompt_text = record.get("prompt", "")

        batch_item = {
            "custom_id": f"task-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ]
            }
        }

        fout.write(json.dumps(batch_item, ensure_ascii=False) + "\n")

print("✔ Done! Created:", output_file)
