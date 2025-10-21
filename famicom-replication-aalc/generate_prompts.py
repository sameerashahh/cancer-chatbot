#!/usr/bin/env python3
"""
Generate cross-task prompts for the AALC replication using the requested datasets.

Behavior implemented:
- Build an evaluation pool from the specified datasets.
- Select 28 task-subtask pairs (prefer harder MMLU subtasks where provided).
- For inference, pair each sampled question with three cross-task CoT demonstrations.
- Sample exactly 1000 prompts total, distributed (as evenly as possible) across the 28 tasks.

Notes:
- The script expects CoT JSON files under `data/cot_hub/` with filenames matching
  the dataset keys used below (e.g. `mmlu.json`, `bigbench.json`, `commonsense2.json`, etc.).
"""

import json
import random
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import re

# deterministic behavior
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


@dataclass
class CoTExample:
    question: str
    cot: str
    answer: str
    task: str = ""
    subtask: str = ""


class PromptGeneratorAALC:
    def __init__(self):
        self.cot_hub_data: Dict[str, List[CoTExample]] = {}
        self.selected_tasks: List[Tuple[str, str]] = []

    def load_cot_hub_data(self) -> None:
        base_dir = os.path.dirname(__file__)
        cot_hub_dir = os.path.join(base_dir, "data", "cot_hub")
        task_files = [
            "bigbench.json",
            "mmlu.json",
            "commonsense2.json",
            "hotpotqa_mc.json",
            "arc_challenge.json",
            "anli.json",
            "copa.json",
            "math_gsm8k.json",
            "openbookqa.json",
            "socialiqa.json",
            "piqa.json",
        ]

        for filename in task_files:
            task_name = filename.replace('.json', '')
            path = os.path.join(cot_hub_dir, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
            except FileNotFoundError:
                print(f"Warning: missing {path} — continuing with empty dataset")
                self.cot_hub_data[task_name] = []
                continue
            except json.JSONDecodeError as e:
                print(f"Warning: invalid JSON in {path}: {e}")
                self.cot_hub_data[task_name] = []
                continue

            examples: List[CoTExample] = []
            for item in raw:
                examples.append(CoTExample(
                    question=item.get('question', ''),
                    cot=item.get('cot', ''),
                    answer=item.get('answer', ''),
                    task=task_name,
                    subtask='default'
                ))

            print(f"Loaded {len(examples)} examples for {task_name}")
            self.cot_hub_data[task_name] = examples

        total = sum(len(v) for v in self.cot_hub_data.values())
        print(f"Total CoT examples loaded: {total}")

    def select_28_tasks(self) -> None:
        # prefer these harder MMLU subtasks
        mmlu_hard = [
            "college_mathematics", "formal_logic", "electrical_engineering",
            "conceptual_physics", "econometrics", "college_physics", "college_computer_science",
            "high_school_physics", "high_school_mathematics"
        ]

        subtasks = {
            "mmlu": mmlu_hard,
            "bigbench": [
                "analogical_reasoning", "arithmetic", "auto_debugging", "cs_algorithms",
                "date_understanding", "elementary_maths_word_problems", "factual_qa",
            ],
            "commonsense2": [
                "cause_effect", "comparison", "composition", "conceptual_similarity",
                "commonsense_reasoning", "logical_reasoning", "physical_reasoning",
            ],
            "hotpotqa_mc": ["multi_hop", "fact_retrieval", "comparison", "synthesis", "evidence"],
            "arc_challenge": ["science_concepts", "scientific_reasoning", "physical_science"],
            "anli": ["r1", "r2", "r3"],
            "copa": ["causal_choice"],
            "math_gsm8k": ["algebra", "arithmetic", "multi_step"],
            "openbookqa": ["science_facts"],
            "socialiqa": ["social_reasoning"],
            "piqa": ["physical_interaction"],
        }

        pool: List[Tuple[str, str]] = []
        for main, subs in subtasks.items():
            for s in subs:
                pool.append((main, s))

        # ensure at least 28 candidates
        if len(pool) < 28:
            keys = list(subtasks.keys())
            i = 0
            while len(pool) < 28:
                k = keys[i % len(keys)]
                s = subtasks[k][i % len(subtasks[k])]
                pool.append((k, s))
                i += 1

        # prefer unique pairs when possible
        unique = list(dict.fromkeys(pool))
        if len(unique) >= 28:
            self.selected_tasks = random.sample(unique, 28)
        else:
            self.selected_tasks = random.sample(pool, 28)

        print(f"Selected {len(self.selected_tasks)} task-subtask pairs for evaluation")

    def extract_options_from_question(self, question: str) -> Tuple[str, dict]:
        m = re.search(r'Options:\s*(.+)$', question)
        if not m:
            return question, {}
        options_text = m.group(1)
        q_text = question[:m.start()].strip()
        pattern = r'\(([A-Z])\)\s*([^(]+?)(?=\s*\([A-Z]\)|$)'
        option_map: Dict[str, str] = {}
        for mm in re.finditer(pattern, options_text):
            option_map[mm.group(1)] = mm.group(2).strip()
        return q_text, option_map

    def convert_to_binary_choice(self, question: str, correct_answer: str) -> Tuple[str, str]:
        q_text, option_map = self.extract_options_from_question(question)
        correct_text = option_map.get(correct_answer, "")
        incorrects = [v for k, v in option_map.items() if k != correct_answer]
        if incorrects:
            other = random.choice(incorrects)
        else:
            other = "Alternative option"
        opts = [correct_text, other]
        random.shuffle(opts)
        new_correct = "A" if opts[0] == correct_text else "B"
        formatted = f"{q_text} Options: (A) {opts[0]} (B) {opts[1]}"
        return formatted, new_correct

    def select_cross_task_demonstrations(self, target: Tuple[str, str]) -> List[Tuple[CoTExample, str, str]]:
        target_main, target_sub = target
        avail = [t for t in self.selected_tasks if not (t[0] == target_main and t[1] == target_sub)]
        if len(avail) < 3:
            avail = [t for t in self.selected_tasks if t[0] != target_main]
        chosen = random.sample(avail, min(3, len(avail)))
        demos: List[Tuple[CoTExample, str, str]] = []
        flat_examples = [ex for lst in self.cot_hub_data.values() for ex in lst]
        for main, sub in chosen:
            examples = self.cot_hub_data.get(main, [])
            if examples:
                ex = random.choice(examples)
            elif flat_examples:
                ex = random.choice(flat_examples)
            else:
                # placeholder empty example
                ex = CoTExample(question="", cot="", answer="", task=main, subtask=sub)
            ex.subtask = sub
            demos.append((ex, main, sub))
        return demos

    def generate_cross_task_prompt(self, target_q: CoTExample, demos: List[Tuple[CoTExample, str, str]], qid: int) -> Tuple[str, str, List[Dict[str, str]]]:
        formatted_target, true_answer = self.convert_to_binary_choice(target_q.question, target_q.answer)
        parts: List[str] = []
        demo_info: List[Dict[str, str]] = []
        for i, (demo, dt, ds) in enumerate(demos):
            formatted_demo, demo_ans = self.convert_to_binary_choice(demo.question, demo.answer)
            parts.append(f"### Example {i+1}:")
            parts.append(f"Question: {formatted_demo}")
            parts.append(f"Reasoning: {demo.cot}")
            parts.append(f"Answer: {demo_ans}")
            parts.append("")
            demo_info.append({"demo_task": dt, "demo_subtask": ds})
        parts.append("### New Question:")
        parts.append(f"Question: {formatted_target}")
        return "\n".join(parts), true_answer, demo_info

    def generate_all_prompts(self, target_count: int = 1000) -> List[Dict[str, Any]]:
        if not self.selected_tasks:
            raise RuntimeError("Call select_28_tasks() before generating prompts")

        total = target_count
        per_task = total // len(self.selected_tasks)
        remainder = total % len(self.selected_tasks)
        results: List[Dict[str, Any]] = []
        qid = 0
        for i, (main, sub) in enumerate(self.selected_tasks):
            n = per_task + (1 if i < remainder else 0)
            examples = self.cot_hub_data.get(main, [])
            if not examples:
                print(f"Warning: no examples for {main}; skipping {n} samples")
                continue
            for _ in range(n):
                target_q = random.choice(examples)
                demos = self.select_cross_task_demonstrations((main, sub))
                prompt, true_ans, demo_info = self.generate_cross_task_prompt(target_q, demos, qid)
                results.append({
                    "task": main,
                    "subtask": sub,
                    "question_id": qid,
                    "prompt": prompt,
                    "true_answer": true_ans,
                    "demonstrations": demo_info,
                })
                qid += 1
        print(f"Finished generating {len(results)} prompts")
        return results

    def save_prompts(self, prompts: List[Dict[str, Any]], filename: str) -> None:
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(prompts)} prompts to {path}")


def main():
    print("Starting AALC prompt generation (1000 prompts total)")
    gen = PromptGeneratorAALC()
    gen.load_cot_hub_data()
    gen.select_28_tasks()
    prompts = gen.generate_all_prompts(target_count=1000)
    gen.save_prompts(prompts, "prepared_prompts.json")
    print("Done.")


if __name__ == '__main__':
    main()