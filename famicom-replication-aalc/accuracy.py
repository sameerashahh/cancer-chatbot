import re
import json

INPUT = "complexity_results.json"
OUTPUT = "results_with_accuracy_hybrid.json"

def extract_predicted_answer(text):
    """Extract predicted answer letter (A–Z) from model output using multiple heuristics."""
    text = text.strip()
    text_upper = text.upper()

    # 1️⃣ Strict match: "answer is" or "answer:"
    m = re.search(r'(?i)\b(?:the\s+)?answer\s*(?:is|:)\s*(.*)', text)
    if m:
        after = m.group(1)
        m2 = re.search(r'([A-Z])', after.upper())
        if m2:
            return m2.group(1)

    # 2️⃣ Flexible: "the correct answer is" or "i think the answer is"
    m = re.search(r'(?i)\b(?:the\s+correct\s+answer|i\s+think\s+the\s+answer)\s*(?:is|:)?\s*(.*)', text)
    if m:
        after = m.group(1)
        m2 = re.search(r'([A-Z])', after.upper())
        if m2:
            return m2.group(1)

    # 3️⃣ Match “option X” pattern
    m = re.search(r'(?i)\boption\s*([A-Z])', text)
    if m:
        return m.group(1)

    # 4️⃣ Fallback: look near the end for something like "(A)" or "A."
    tail = text_upper[-200:]
    m = re.search(r'\b\(?([A-Z])\)?[.\)]?\s*$', tail)
    if m:
        return m.group(1)

    return None


with open(INPUT, "r") as f:
    data = json.load(f)

total = len(data)
extracted = 0
correct = 0

for item in data:
    true = item.get("true_answer", "")
    if true is None:
        true = ""
    true = true.strip().upper()

    output = item.get("output", "") or ""
    pred = extract_predicted_answer(output)

    if pred is not None:
        extracted += 1
        acc = 1 if pred == true and true != "" else 0
    else:
        acc = None

    item["predicted_answer"] = pred
    item["accuracy"] = acc
    if acc == 1:
        correct += 1

# Save annotated results
with open(OUTPUT, "w") as f:
    json.dump(data, f, indent=2)

# Print summary
overall_acc = correct / extracted if extracted else 0
print(f"Total examples: {total}")
print(f"Extracted answers: {extracted} ({extracted/total:.2%})")
print(f"Correct among extracted: {correct}/{extracted}")
print(f"Accuracy (on extracted only): {overall_acc:.2%}")
