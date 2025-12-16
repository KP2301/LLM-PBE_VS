import os
import sys
from dotenv import load_dotenv
from contextlib import redirect_stdout
import re 

# --- [ส่วนที่ 1] Import Metric ที่เราเพิ่งสร้าง ---
try:
    from metrics.ExtractionRate import ExtractionRate
except ImportError:
    print("Error: Could not import 'ExtractionRate'. Make sure 'metrics/ExtractionRate.py' exists.")
    sys.exit(1)

OUTPUT_DIR = "output/dea"
BASE_FILENAME = "dea_output"
EXTENSION = ".txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
existing_files = os.listdir(OUTPUT_DIR)
next_index = 1

pattern = re.compile(rf"^{BASE_FILENAME}_(\d+){EXTENSION}$")

for filename in existing_files:
    match = pattern.match(filename)
    if match:
        current_index = int(match.group(1))
        if current_index >= next_index:
            next_index = current_index + 1

OUTPUT_FILENAME = f"{BASE_FILENAME}_{next_index}{EXTENSION}"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    print("Error: GROQ_API_KEY not found in environment variables. Check your .env file.")
    sys.exit(1)

from models.GroqModels import GroqModels
from attacks.DataExtraction.enron import EnronDataExtraction
from attacks.DataExtraction.prompt_extract import PromptExtraction

enron = EnronDataExtraction(data_path="data/enron")

atk_format = '3-shot-known-domain-c'

# --- [ส่วนที่ 2] แก้ไขเพื่อรับ Labels (เฉลย) มาด้วย ---
# เดิม: prompts, _ = ... (ทิ้งเฉลย)
# ใหม่: รับ labels มาเก็บไว้
prompts, labels = enron.generate_prompts(format=atk_format)

llm_model = "llama-3.1-8b-instant"
llm = GroqModels(model=llm_model)
attack = PromptExtraction()

# กำหนดจำนวนที่จะทดสอบ (แยกตัวแปรไว้จะได้แก้ที่เดียว)
NUM_SAMPLES = 100

# ตัดข้อมูลให้เท่ากันทั้งโจทย์และเฉลย
test_prompts = prompts[:NUM_SAMPLES]
test_labels = labels[:NUM_SAMPLES]

print(f"Writing results to {OUTPUT_PATH}")
print(f"Starting attack on {len(test_prompts)} samples...")

# เริ่มโจมตี
results = attack.execute_attack(test_prompts, llm)

print(f"Attack finished. Calculating metrics...")

# --- [ส่วนที่ 3] คำนวณ Metric ---
metric = ExtractionRate(outputs=results, labels=test_labels)
score = metric.compute_metric()
asr_percentage = score * 100

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    with redirect_stdout(f):
        print("="*100)
        print(f"DATA EXTRACTION RESULTS - Total: {len(results)} samples")
        print(f"METRIC SCORE (ASR): {asr_percentage:.2f}%")
        print("="*100)

        for idx in range(len(results)):
            print(f"\n{'─'*100}")
            print(f"SAMPLE #{idx + 1}")
            
            # เช็คสถานะเบื้องต้นเพื่อโชว์ใน Log (Success/Fail)
            clean_label = test_labels[idx].strip()
            is_success = clean_label in results[idx]
            status_icon = "✅ LEAKED" if is_success else "❌ SAFE"
            
            print(f"STATUS: {status_icon}")
            print(f"{'─'*100}")

            print(f"\n📝 PROMPT (Input):")
            print(test_prompts[idx])

            # เพิ่มส่วนแสดงเฉลย (เพื่อให้เราเทียบดูได้ด้วยตา)
            print(f"\n🔐 LABEL (Secret):") 
            print(test_labels[idx])

            print(f"\n💬 ANSWER (Model Output):")
            print(results[idx])
            print()

        print("="*100)
        print(f"📊 FINAL SUMMARY")
        print(f"Total Samples: {len(results)}")
        print(f"Extraction Success Rate: {asr_percentage:.2f}%")
        print("="*100)

print(f"Done. Check the output directory: {OUTPUT_FILENAME}")
print(f"Final Extraction Success Rate: {asr_percentage:.2f}%")