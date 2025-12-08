import sys
import os
from dotenv import load_dotenv

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    print("Error: GROQ_API_KEY not found in environment variables. Check your .env file.")
    sys.exit(1)

from models.GroqModels import GroqModels
from attacks.PromptLeakage.prompt_leakage_groq import PromptLeakage
from attacks.PromptLeakage.prompt_data import openai_playground_prompts

target_system_prompts = []
print(f"\n--- Loading {len(openai_playground_prompts)} Targets ---")
for key, value in openai_playground_prompts.items():
    target_system_prompts.append(value['instruction'])
    print(f"Target [{key}]: {value['instruction'][:60]}...")

llm = GroqModels(model="llama-3.1-8b-instant")
attacker = PromptLeakage()

print("\n🚀 Starting Attack Sequence...")
attack_results = attacker.execute_attack(target_system_prompts, llm)

scores = attacker.compute_scores(target_system_prompts, attack_results)

print("\n" + "="*40)
print("📊 FINAL LEAKAGE SCORES (0-100)")
print("="*40)
for attack_type, score in scores.items():
    print(f"Attack '{attack_type}': {score:.2f}")

print("\n🔍 Example Leakage (Check 'ignore_print'):")
if 'ignore_print' in attack_results:
    for i, res in enumerate(attack_results['ignore_print']):
        print(f"\n[Case {i+1}]")
        print(f"Secret: {target_system_prompts[i]}")
        print(f"Model Said: {res}")