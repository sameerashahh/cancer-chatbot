import json
import os
import re
from decimal import Decimal, InvalidOperation

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================

MODELS = [
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "NoesisLab/Spartacus-1B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "stabilityai/stablelm-2-1_6b-chat",
    "tiiuae/Falcon-H1-Tiny-90M-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/flan-t5-base",
    "google/flan-t5-large",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLES_PER_DATASET = 200
SHUFFLE_SEED = 42
BATCH_SIZE = 8
MAX_NEW_TOKENS = 96
DO_SAMPLE = False
MAX_INPUT_TOKENS = 2048

OUTPUT_DIR = "results_revised"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {DEVICE}")
print(f"Writing results to: {OUTPUT_DIR}")

DBPEDIA_LABEL_MAP = {
    "0": "Company",
    "1": "Educational Institution",
    "2": "Artist",
    "3": "Athlete",
    "4": "Office Holder",
    "5": "Mean Of Transportation",
    "6": "Building",
    "7": "Natural Place",
    "8": "Village",
    "9": "Animal",
    "10": "Plant",
    "11": "Album",
    "12": "Film",
    "13": "Written Work",
}

BANKING77_LABEL_MAP = {
    "0": "activate_my_card",
    "1": "age_limit",
    "2": "apple_pay_or_google_pay",
    "3": "atm_support",
    "4": "automatic_top_up",
    "5": "balance_not_updated_after_bank_transfer",
    "6": "balance_not_updated_after_cheque_or_cash_deposit",
    "7": "beneficiary_not_allowed",
    "8": "cancel_transfer",
    "9": "card_about_to_expire",
    "10": "card_acceptance",
    "11": "card_arrival",
    "12": "card_delivery_estimate",
    "13": "card_linking",
    "14": "card_not_working",
    "15": "card_payment_fee_charged",
    "16": "card_payment_not_recognised",
    "17": "card_payment_wrong_exchange_rate",
    "18": "card_swallowed",
    "19": "cash_withdrawal_charge",
    "20": "cash_withdrawal_not_recognised",
    "21": "change_pin",
    "22": "compromised_card",
    "23": "contactless_not_working",
    "24": "country_support",
    "25": "declined_card_payment",
    "26": "declined_cash_withdrawal",
    "27": "declined_transfer",
    "28": "direct_debit_payment_not_recognised",
    "29": "disposable_card_limits",
    "30": "edit_personal_details",
    "31": "exchange_charge",
    "32": "exchange_rate",
    "33": "exchange_via_app",
    "34": "extra_charge_on_statement",
    "35": "failed_transfer",
    "36": "fiat_currency_support",
    "37": "get_disposable_virtual_card",
    "38": "get_physical_card",
    "39": "getting_spare_card",
    "40": "getting_virtual_card",
    "41": "lost_or_stolen_card",
    "42": "lost_or_stolen_phone",
    "43": "order_physical_card",
    "44": "passcode_forgotten",
    "45": "pending_card_payment",
    "46": "pending_cash_withdrawal",
    "47": "pending_top_up",
    "48": "pending_transfer",
    "49": "pin_blocked",
    "50": "receiving_money",
    "51": "Refund_not_showing_up",
    "52": "request_refund",
    "53": "reverted_card_payment?",
    "54": "supported_cards_and_currencies",
    "55": "terminate_account",
    "56": "top_up_by_bank_transfer_charge",
    "57": "top_up_by_card_charge",
    "58": "top_up_by_cash_or_cheque",
    "59": "top_up_failed",
    "60": "top_up_limits",
    "61": "top_up_reverted",
    "62": "topping_up_by_card",
    "63": "transaction_charged_twice",
    "64": "transfer_fee_charged",
    "65": "transfer_into_account",
    "66": "transfer_not_received_by_recipient",
    "67": "transfer_timing",
    "68": "unable_to_verify_identity",
    "69": "verify_my_identity",
    "70": "verify_source_of_funds",
    "71": "verify_top_up",
    "72": "virtual_card_not_working",
    "73": "visa_or_mastercard",
    "74": "why_verify_identity",
    "75": "wrong_amount_of_cash_received",
    "76": "wrong_exchange_rate_for_cash_withdrawal",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_number(token: str):
    token = token.strip().replace(",", "")
    if not token:
        return None
    try:
        value = Decimal(token)
    except (InvalidOperation, ValueError):
        return None
    out = format(value.normalize(), "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    if out in ("", "-0"):
        out = "0"
    return out


def label_aliases_for_eval(labels):
    aliases = {}
    for label in labels:
        key = label.lower()
        aliases[key] = label
        aliases[key.replace("_", " ")] = label
        aliases[key.replace("_", "-")] = label
    # Common variants
    if "not_hate" in labels:
        aliases["not hate"] = "not_hate"
        aliases["non hate"] = "not_hate"
        aliases["non-hate"] = "not_hate"
    if "sci_tech" in labels:
        aliases["sci tech"] = "sci_tech"
        aliases["science and technology"] = "sci_tech"
        aliases["science & technology"] = "sci_tech"
    return aliases


def extract_last_line_payload(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", "empty_output"
    last = lines[-1]
    m = re.match(r"(?is)^final\s*answer\s*[:\-]\s*(.+)$", last)
    if m:
        return m.group(1).strip(), "last_line_final_answer"
    return last, "last_line"


def extract_final_answer_anywhere(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        m = re.match(r"(?is)^final\s*answer\s*[:\-]\s*(.+)$", line)
        if m:
            return m.group(1).strip(), "full_output_final_answer"
    return None, "no_final_answer"


def extract_categorical(text: str, labels):
    aliases = label_aliases_for_eval(labels)
    payload, src = extract_last_line_payload(text)

    norm_payload = normalize_text(payload)
    if norm_payload in aliases:
        return aliases[norm_payload], f"{src}_exact"

    # Last matching alias in payload.
    best = None
    for alias, label in aliases.items():
        pat = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        for m in pat.finditer(norm_payload):
            cand = (m.end(), len(alias), label)
            if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand
    if best is not None:
        return best[2], f"{src}_alias"

    # If last line failed, parse from full output.
    payload2, src2 = extract_final_answer_anywhere(text)
    if payload2 is not None:
        norm_payload2 = normalize_text(payload2)
        if norm_payload2 in aliases:
            return aliases[norm_payload2], f"{src2}_exact"
        best2 = None
        for alias, label in aliases.items():
            pat = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
            for m in pat.finditer(norm_payload2):
                cand = (m.end(), len(alias), label)
                if best2 is None or cand[0] > best2[0] or (cand[0] == best2[0] and cand[1] > best2[1]):
                    best2 = cand
        if best2 is not None:
            return best2[2], f"{src2}_alias"

    norm_all = normalize_text(text)
    best3 = None
    for alias, label in aliases.items():
        pat = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        for m in pat.finditer(norm_all):
            cand = (m.end(), len(alias), label)
            if best3 is None or cand[0] > best3[0] or (cand[0] == best3[0] and cand[1] > best3[1]):
                best3 = cand
    if best3 is not None:
        return best3[2], "full_output_alias"

    # Deterministic fallback: always emit one valid label.
    return labels[0], "fallback_default_label"


def extract_letter_ae(text: str):
    payload, src = extract_last_line_payload(text)
    # \\boxed{A}
    boxed = re.findall(r"\\boxed\{\s*([A-Ea-e])\s*\}", payload)
    if boxed:
        return boxed[-1].upper(), f"{src}_boxed"

    letters = re.findall(r"(?<!\w)([A-Ea-e])(?!\w)", payload)
    if letters:
        return letters[-1].upper(), f"{src}_letter"

    digits = re.findall(r"(?<!\w)([1-5])(?!\w)", payload)
    if digits:
        mapped = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}[digits[-1]]
        return mapped, f"{src}_digit"

    payload2, src2 = extract_final_answer_anywhere(text)
    if payload2 is not None:
        boxed2 = re.findall(r"\\boxed\{\s*([A-Ea-e])\s*\}", payload2)
        if boxed2:
            return boxed2[-1].upper(), f"{src2}_boxed"
        letters2 = re.findall(r"(?<!\w)([A-Ea-e])(?!\w)", payload2)
        if letters2:
            return letters2[-1].upper(), f"{src2}_letter"
        digits2 = re.findall(r"(?<!\w)([1-5])(?!\w)", payload2)
        if digits2:
            mapped = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}[digits2[-1]]
            return mapped, f"{src2}_digit"

    letters3 = re.findall(r"(?<!\w)([A-Ea-e])(?!\w)", text)
    if letters3:
        return letters3[-1].upper(), "full_output_letter"

    return "A", "fallback_default_A"


def build_label_name_map(id_to_name):
    label_to_names = {}
    for label_id, name in id_to_name.items():
        name = str(name)
        label_to_names[str(label_id)] = [
            name,
            name.replace("_", " "),
            name.replace("_", "-"),
        ]
    return label_to_names


def extract_id(text: str, min_id: int, max_id: int, label_name_map=None):
    payload, src = extract_last_line_payload(text)

    # Prefer IDs directly.
    ids = re.findall(r"-?\d+", payload)
    for tok in reversed(ids):
        try:
            v = int(tok)
        except ValueError:
            continue
        if min_id <= v <= max_id:
            return str(v), f"{src}_id"

    # Then label names (if available).
    if label_name_map:
        norm_payload = normalize_text(payload)
        best = None
        for label_id, names in label_name_map.items():
            for nm in names:
                alias = normalize_text(nm)
                pat = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
                for m in pat.finditer(norm_payload):
                    cand = (m.end(), len(alias), label_id)
                    if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                        best = cand
        if best is not None:
            return best[2], f"{src}_name"

    payload2, src2 = extract_final_answer_anywhere(text)
    if payload2 is not None:
        ids2 = re.findall(r"-?\d+", payload2)
        for tok in reversed(ids2):
            try:
                v = int(tok)
            except ValueError:
                continue
            if min_id <= v <= max_id:
                return str(v), f"{src2}_id"
        if label_name_map:
            norm_payload2 = normalize_text(payload2)
            best2 = None
            for label_id, names in label_name_map.items():
                for nm in names:
                    alias = normalize_text(nm)
                    pat = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
                    for m in pat.finditer(norm_payload2):
                        cand = (m.end(), len(alias), label_id)
                        if best2 is None or cand[0] > best2[0] or (cand[0] == best2[0] and cand[1] > best2[1]):
                            best2 = cand
            if best2 is not None:
                return best2[2], f"{src2}_name"

    ids3 = re.findall(r"-?\d+", text)
    for tok in reversed(ids3):
        try:
            v = int(tok)
        except ValueError:
            continue
        if min_id <= v <= max_id:
            return str(v), "full_output_id"

    if label_name_map:
        norm_all = normalize_text(text)
        best3 = None
        for label_id, names in label_name_map.items():
            for nm in names:
                alias = normalize_text(nm)
                pat = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
                for m in pat.finditer(norm_all):
                    cand = (m.end(), len(alias), label_id)
                    if best3 is None or cand[0] > best3[0] or (cand[0] == best3[0] and cand[1] > best3[1]):
                        best3 = cand
        if best3 is not None:
            return best3[2], "full_output_name"

    return str(min_id), "fallback_default_id"


def extract_gsm8k_ref(answer_text: str):
    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", answer_text)
    if m:
        val = canonicalize_number(m.group(1))
        if val is not None:
            return val
    # Guaranteed fallback for completeness.
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", answer_text)
    if nums:
        val = canonicalize_number(nums[-1])
        if val is not None:
            return val
    return "0"


def extract_gsm8k_pred(text: str):
    payload, src = extract_last_line_payload(text)

    boxed = re.findall(r"\\boxed\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}", payload)
    if boxed:
        val = canonicalize_number(boxed[-1])
        if val is not None:
            return val, f"{src}_boxed"

    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", payload)
    if nums:
        val = canonicalize_number(nums[-1])
        if val is not None:
            return val, f"{src}_number"

    payload2, src2 = extract_final_answer_anywhere(text)
    if payload2 is not None:
        boxed2 = re.findall(r"\\boxed\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}", payload2)
        if boxed2:
            val = canonicalize_number(boxed2[-1])
            if val is not None:
                return val, f"{src2}_boxed"
        nums2 = re.findall(r"-?\d[\d,]*(?:\.\d+)?", payload2)
        if nums2:
            val = canonicalize_number(nums2[-1])
            if val is not None:
                return val, f"{src2}_number"

    nums3 = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if nums3:
        val = canonicalize_number(nums3[-1])
        if val is not None:
            return val, "full_output_number"

    return "0", "fallback_default_0"


def parse_aqua_option_lines(raw_options):
    if isinstance(raw_options, list):
        return [str(x) for x in raw_options]
    if isinstance(raw_options, str):
        parts = [x for x in raw_options.splitlines() if x.strip()]
        return parts if parts else [raw_options]
    return []


def format_aqua_text(sample):
    opts = parse_aqua_option_lines(sample.get("options"))
    lines = []
    if opts:
        for i, opt in enumerate(opts[:5]):
            letter = ["A", "B", "C", "D", "E"][i]
            m = re.match(r"^\s*([A-E])[\)\].:\-]?\s*(.*)$", opt, re.I)
            body = m.group(2).strip() if m else opt.strip()
            lines.append(f"{letter}) {body}")
    return f"{sample['question']}\nOptions:\n" + "\n".join(lines)


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "imdb": {
        "path": "imdb",
        "split": "test",
        "labels": ["negative", "positive"],
        "text": lambda s, rt: s["text"],
        "reference": lambda s, rt: {0: "negative", 1: "positive"}[s["label"]],
        "prompt": lambda text, rt: f"""
You are a sentiment classifier.
Label must be one of: negative, positive.

Text:
{text}

Return exactly one line:
Final answer: <negative or positive>
""",
        "extract": lambda out, sample, rt: extract_categorical(out, ["negative", "positive"]),
    },
    "yelp_polarity": {
        "path": "yelp_polarity",
        "split": "test",
        "labels": ["negative", "positive"],
        "text": lambda s, rt: s["text"],
        "reference": lambda s, rt: {0: "negative", 1: "positive"}[s["label"]],
        "prompt": lambda text, rt: f"""
You are a sentiment classifier.
Label must be one of: negative, positive.

Text:
{text}

Return exactly one line:
Final answer: <negative or positive>
""",
        "extract": lambda out, sample, rt: extract_categorical(out, ["negative", "positive"]),
    },
    "sst2": {
        "path": "glue",
        "config": "sst2",
        "split": "validation",
        "labels": ["negative", "positive"],
        "text": lambda s, rt: s["sentence"],
        "reference": lambda s, rt: {0: "negative", 1: "positive"}[s["label"]],
        "prompt": lambda text, rt: f"""
You are a sentiment classifier.
Label must be one of: negative, positive.

Text:
{text}

Return exactly one line:
Final answer: <negative or positive>
""",
        "extract": lambda out, sample, rt: extract_categorical(out, ["negative", "positive"]),
    },
    "tweet_eval_hate": {
        "path": "tweet_eval",
        "config": "hate",
        "split": "test",
        "labels": ["not_hate", "hate"],
        "text": lambda s, rt: s["text"],
        "reference": lambda s, rt: {0: "not_hate", 1: "hate"}[s["label"]],
        "prompt": lambda text, rt: f"""
You are a hate-speech classifier.
Label must be one of: not_hate, hate.

Text:
{text}

Return exactly one line:
Final answer: <not_hate or hate>
""",
        "extract": lambda out, sample, rt: extract_categorical(out, ["not_hate", "hate"]),
    },
    "boolq": {
        "path": "boolq",
        "split": "validation",
        "labels": ["no", "yes"],
        "text": lambda s, rt: f"Passage: {s['passage']}\nQuestion: {s['question']}",
        "reference": lambda s, rt: "yes" if s["answer"] else "no",
        "prompt": lambda text, rt: f"""
Answer the question based on the passage.
Label must be one of: yes, no.

{text}

Return exactly one line:
Final answer: <yes or no>
""",
        "extract": lambda out, sample, rt: extract_categorical(out, ["no", "yes"]),
    },
    "ag_news": {
        "path": "ag_news",
        "split": "test",
        "labels": ["world", "sports", "business", "sci_tech"],
        "text": lambda s, rt: s["text"],
        "reference": lambda s, rt: {0: "world", 1: "sports", 2: "business", 3: "sci_tech"}[s["label"]],
        "prompt": lambda text, rt: f"""
Classify the news article.
Label must be one of: world, sports, business, sci_tech.

Article:
{text}

Return exactly one line:
Final answer: <world or sports or business or sci_tech>
""",
        "extract": lambda out, sample, rt: extract_categorical(out, ["world", "sports", "business", "sci_tech"]),
    },
    "dbpedia_14": {
        "path": "dbpedia_14",
        "split": "test",
        "prepare_runtime": lambda ds: {"label_name_map": build_label_name_map(DBPEDIA_LABEL_MAP)},
        "text": lambda s, rt: f"{s['title']} {s['content']}",
        "reference": lambda s, rt: str(s["label"]),
        "prompt": lambda text, rt: f"""
Classify into one DBPedia label.
Return the label name exactly from this list:
Company | Educational Institution | Artist | Athlete | Office Holder | Mean Of Transportation | Building | Natural Place | Village | Animal | Plant | Album | Film | Written Work

Text:
{text}

Return exactly one line:
Final answer: <one label name from the list>
""",
        "extract": lambda out, sample, rt: extract_id(out, 0, 13, rt.get("label_name_map")),
    },
    "banking77": {
        "path": "banking77",
        "split": "test",
        "prepare_runtime": lambda ds: {"label_name_map": build_label_name_map(BANKING77_LABEL_MAP)},
        "text": lambda s, rt: s["text"],
        "reference": lambda s, rt: str(s["label"]),
        "prompt": lambda text, rt: f"""
Classify into one Banking77 intent.
Return the intent name exactly (lowercase with underscores), e.g. activate_my_card.

Text:
{text}

Return exactly one line:
Final answer: <one Banking77 intent name>
""",
        "extract": lambda out, sample, rt: extract_id(out, 0, 76, rt.get("label_name_map")),
    },
    "gsm8k": {
        "path": "gsm8k",
        "config": "main",
        "split": "test",
        "max_new_tokens": 256,
        "text": lambda s, rt: s["question"],
        "reference": lambda s, rt: extract_gsm8k_ref(s["answer"]),
        "prompt": lambda text, rt: f"""
Solve the math problem.

Problem:
{text}

Return exactly one line:
Final answer: \\boxed{{<number>}}
""",
        "extract": lambda out, sample, rt: extract_gsm8k_pred(out),
    },
    "aqua_rat": {
        "path": "aqua_rat",
        "split": "test",
        "text": lambda s, rt: format_aqua_text(s),
        "reference": lambda s, rt: str(s.get("correct", s.get("answer", "A"))).strip().upper(),
        "prompt": lambda text, rt: f"""
Solve the multiple-choice problem.
Answer must be one of: A, B, C, D, E.

{text}

Return exactly one line:
Final answer: <A or B or C or D or E>
""",
        "extract": lambda out, sample, rt: extract_letter_ae(out),
    },
}


# ============================================================
# TOKENIZATION / DECODE
# ============================================================

def make_batch_inputs(tokenizer, prompts, is_seq2seq):
    if (not is_seq2seq) and hasattr(tokenizer, "apply_chat_template"):
        batch_messages = [[{"role": "user", "content": p}] for p in prompts]
        inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

    inputs = inputs.to(DEVICE)

    input_lengths = None
    if not is_seq2seq:
        if "attention_mask" in inputs:
            input_lengths = inputs["attention_mask"].sum(dim=1)
        else:
            input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

    return inputs, input_lengths


def decode_batch_outputs(tokenizer, outputs, input_lengths, is_seq2seq):
    if is_seq2seq:
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    decoded = []
    for i in range(outputs.size(0)):
        start = int(input_lengths[i].item()) if input_lengths is not None else 0
        gen_tokens = outputs[i, start:]
        decoded.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    return decoded


# ============================================================
# GENERATION
# ============================================================

def generate_oom_safe(model, tokenizer, prompts, is_seq2seq, max_new_tokens):
    results = []
    bs = len(prompts)
    i = 0
    while i < len(prompts):
        chunk = prompts[i : i + bs]
        try:
            inputs, input_lengths = make_batch_inputs(tokenizer, chunk, is_seq2seq)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=DO_SAMPLE,
                    pad_token_id=tokenizer.pad_token_id,
                )
            results.extend(decode_batch_outputs(tokenizer, outputs, input_lengths, is_seq2seq))
            i += bs
        except torch.cuda.OutOfMemoryError:
            if DEVICE != "cuda":
                raise
            torch.cuda.empty_cache()
            bs //= 2
            if bs < 1:
                raise RuntimeError("OOM even with batch_size=1. Reduce max_new_tokens or prompt size.")
            print(f"  [OOM] reducing sub-batch size -> {bs} and retrying...")
    return results


# ============================================================
# MAIN
# ============================================================

for model_name in MODELS:
    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype="auto").to(DEVICE)
        is_seq2seq = True
    else:
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        ).to(DEVICE)
        is_seq2seq = False

    model.eval()

    model_results = {
        "model": model_name,
        "device": DEVICE,
        "samples_per_dataset": SAMPLES_PER_DATASET,
        "shuffle_seed": SHUFFLE_SEED,
        "batch_size": BATCH_SIZE,
        "max_new_tokens_default": MAX_NEW_TOKENS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "datasets": {},
    }

    for ds_name, cfg in DATASETS.items():
        print("\n" + "-" * 60)
        print(f"DATASET: {ds_name}")
        print("-" * 60)

        try:
            ds = load_dataset(cfg["path"], cfg.get("config"), split=cfg["split"])
            runtime = cfg.get("prepare_runtime", lambda _ds: {})(ds)

            n = min(SAMPLES_PER_DATASET, len(ds))
            ds = ds.shuffle(seed=SHUFFLE_SEED).select(range(n))

            correct = 0
            total = 0
            none_preds = 0
            parsed_count = 0
            correct_parsed = 0

            dataset_max_new_tokens = cfg.get("max_new_tokens", MAX_NEW_TOKENS)

            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                batch = [ds[i] for i in range(start, end)]

                texts = [cfg["text"](sample, runtime) for sample in batch]
                prompts = [cfg["prompt"](text, runtime) for text in texts]
                refs = [cfg["reference"](sample, runtime) for sample in batch]

                decoded_list = generate_oom_safe(
                    model,
                    tokenizer,
                    prompts,
                    is_seq2seq,
                    max_new_tokens=dataset_max_new_tokens,
                )

                for sample, text, decoded, ref in zip(batch, texts, decoded_list, refs):
                    pred, src = cfg["extract"](decoded, sample, runtime)

                    total += 1

                    if pred is None:
                        none_preds += 1

                    is_parsed = not str(src).startswith("fallback_")
                    if is_parsed:
                        parsed_count += 1

                    if pred == ref:
                        correct += 1
                        if is_parsed:
                            correct_parsed += 1

                if end % 25 == 0 or end == n:
                    print(f"  processed {end}/{n}...")

            acc = correct / total if total else 0.0
            parse_rate = parsed_count / total if total else 0.0
            acc_parsed = correct_parsed / parsed_count if parsed_count else 0.0
            print(f"Accuracy: {acc:.4f} ({correct}/{total}), None preds: {none_preds}")
            print(f"Parse rate: {parse_rate:.4f} ({parsed_count}/{total})")
            print(f"Accuracy on parsed outputs: {acc_parsed:.4f} ({correct_parsed}/{parsed_count})")

            model_results["datasets"][ds_name] = {
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "none_predictions": none_preds,
                "parse_rate": parse_rate,
                "acc_parsed": acc_parsed,
            }

        except Exception as exc:
            print(f"[ERROR] {model_name} on {ds_name}: {repr(exc)}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            model_results["datasets"][ds_name] = {"error": repr(exc)}
            continue

    safe_name = model_name.replace("/", "__").replace(":", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)

    print(f"\nSaved: {out_path}")

print("\nDone.")
