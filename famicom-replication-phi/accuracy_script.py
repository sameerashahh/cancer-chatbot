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
    correct_count = 0
    for entry in data:
        model_choice = extract_model_choice(entry.get("model_answer", ""))
        true_answer = entry.get("true_answer", "").strip().upper()
        entry["accuracy"] = 1 if model_choice.upper() == true_answer else 0
        if entry["accuracy"] == 1:
            correct_count += 1
    return data, correct_count

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    updated_data, correct_count = add_accuracy_field(data)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(updated_data, f, indent=2)

    total_entries = len(updated_data)
    accuracy_percent = (correct_count / total_entries) * 100 if total_entries > 0 else 0

    print(f"Processed {total_entries} entries. Results saved to {OUTPUT_FILE}.")
    print(f"Model Accuracy: {accuracy_percent:.2f}%")

if __name__ == "__main__":
    main()
