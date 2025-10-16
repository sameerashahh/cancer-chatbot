import json
import re

INPUT_FILE = "phi3_model_answers.json"
OUTPUT_FILE = "prompts_with_accuracy.json"

def extract_model_choice(model_answer: str) -> str:
    """
    Extracts the first choice letter from model answer.
    Handles formats like "(B) No" or "B" or "B\n\nSome text".
    """
    match = re.search(r"\(?([A-Z])\)?", model_answer)
    if match:
        return match.group(1)
    return ""  # fallback if nothing found

def add_accuracy_field(data):
    for entry in data:
        model_choice = extract_model_choice(entry.get("model_answer", ""))
        true_answer = entry.get("true_answer", "").strip().upper()
        entry["accuracy"] = 1 if model_choice.upper() == true_answer else 0
    return data

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    updated_data = add_accuracy_field(data)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(updated_data, f, indent=2)

    print(f"Processed {len(updated_data)} entries. Results saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
