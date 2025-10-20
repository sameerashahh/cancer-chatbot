#!/usr/bin/env python3
"""
Generate cross-task prompts exactly as described in the FAMICOM paper.
This script creates cross-task question-demonstration pairs for evaluation.
"""

import json
import random
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import re

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

@dataclass
class CoTExample:
    """Represents a Chain-of-Thought example."""
    question: str
    cot: str
    answer: str
    task: str = ""
    subtask: str = ""

class PromptGenerator:
    """Generates cross-task prompts following FAMICOM paper specifications."""
    
    def __init__(self):
        self.cot_hub_data: Dict[str, List[CoTExample]] = {}
        self.selected_tasks: List[str] = []
        
    def load_cot_hub_data(self) -> None:
        """Load Chain-of-Thought demonstrations from CoT Hub JSON files."""
        print("Loading Chain-of-Thought demonstrations from CoT Hub...")
        
        cot_hub_dir = "data/cot_hub"
        task_files = ["mmlu.json", "bigbench.json", "strategyqa.json", "commonsenseqa.json"]
        
        for filename in task_files:
            task_name = filename.replace(".json", "")
            filepath = os.path.join(cot_hub_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    
                # Convert to CoTExample objects
                self.cot_hub_data[task_name] = []
                for item in raw_data:
                    example = CoTExample(
                        question=item["question"],
                        cot=item["cot"],
                        answer=item["answer"],
                        task=task_name,
                        subtask="default"  # We'll assign specific subtasks later
                    )
                    self.cot_hub_data[task_name].append(example)
                    
                print(f"Loaded {len(self.cot_hub_data[task_name])} examples from {task_name}")
            except FileNotFoundError:
                print(f"Warning: Could not find {filepath}")
                self.cot_hub_data[task_name] = []
            except json.JSONDecodeError as e:
                print(f"Error loading {filepath}: {e}")
                self.cot_hub_data[task_name] = []
        
        total_examples = sum(len(examples) for examples in self.cot_hub_data.values())
        print(f"Loaded {total_examples} total CoT examples from {len(self.cot_hub_data)} tasks")
    
    def select_28_tasks(self) -> None:
        """Randomly select 28 tasks from the pool."""
        print("Selecting 28 tasks...")
        
        # Define specific subtasks for each main task type
        subtasks = {
            "mmlu": [
                "abstract_algebra", "anatomy", "astronomy", "business_ethics", 
                "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science",
                "college_mathematics", "college_medicine", "college_physics", "computer_security",
                "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics",
                "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry",
                "high_school_computer_science", "high_school_european_history", "high_school_geography",
                "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics",
                "high_school_physics", "high_school_psychology"
            ],
            "bigbench": [
                "analogical_reasoning", "arithmetic", "auto_debugging", "auto_causal_reasoning",
                "auto_causal_reasoning", "cause_and_effect", "code_line_description", "color",
                "com2sense", "common_morpheme", "conceptual_combinations", "conlang_translation",
                "cryptonite", "cs_algorithms", "date_understanding", "disambiguation_qa",
                "dyck_languages", "elementary_maths_word_problems", "empirical_judgments", "factual_qa"
            ],
            "strategyqa": [
                "causal_reasoning", "commonsense_reasoning", "comparison_reasoning", "counterfactual_reasoning",
                "logical_reasoning", "multi_step_reasoning", "temporal_reasoning", "spatial_reasoning"
            ],
            "commonsenseqa": [
                "cause_effect", "comparison", "composition", "conceptual_similarity",
                "commonsense_reasoning", "logical_reasoning", "physical_reasoning", "social_reasoning"
            ]
        }
        
        # Create a pool of tasks with subtask names
        task_pool = []
        for main_task, subtask_list in subtasks.items():
            for subtask in subtask_list[:7]:  # Take first 7 subtasks per main task
                task_pool.append((main_task, subtask))
        
        # Randomly select 28 tasks
        self.selected_tasks = random.sample(task_pool, 28)
        print(f"Selected {len(self.selected_tasks)} tasks from the pool")
    
    def extract_options_from_question(self, question: str) -> Tuple[str, List[str], str]:
        """Extract question text and options from a formatted question."""
        # Find the options part: "Options: (A) ... (B) ... (C) ... (D) ..."
        options_match = re.search(r'Options:\s*(.+)$', question)
        if not options_match:
            return question, [], ""
        
        options_text = options_match.group(1)
        question_text = question[:options_match.start()].strip()
        
        # Parse options: (A) option1 (B) option2 (C) option3 (D) option4
        option_pattern = r'\(([A-Z])\)\s*([^(]+?)(?=\s*\([A-Z]\)|$)'
        options = []
        option_map = {}
        
        for match in re.finditer(option_pattern, options_text):
            letter = match.group(1)
            option_text = match.group(2).strip()
            options.append(option_text)
            option_map[letter] = option_text
        
        return question_text, options, option_map
    
    def convert_to_binary_choice(self, question: str, correct_answer: str) -> Tuple[str, str]:
        """Convert multiple-choice question to binary choice format."""
        question_text, options, option_map = self.extract_options_from_question(question)
        
        # Get correct option text
        correct_text = option_map.get(correct_answer, "")
        
        # Get all incorrect options
        incorrect_options = [text for letter, text in option_map.items() if letter != correct_answer]
        
        # Randomly select one incorrect option
        if incorrect_options:
            incorrect_text = random.choice(incorrect_options)
        else:
            incorrect_text = "Alternative option"
        
        # Create binary options
        binary_options = [correct_text, incorrect_text]
        
        # Shuffle order
        random.shuffle(binary_options)
        
        # Determine new correct answer
        new_correct = "A" if correct_text == binary_options[0] else "B"
        
        # Format question with binary options
        formatted_question = f"{question_text} Options: (A) {binary_options[0]} (B) {binary_options[1]}"
        
        return formatted_question, new_correct
    
    def select_cross_task_demonstrations(self, target_task: Tuple[str, str], available_subtasks: List[Tuple[str, str]]) -> List[Tuple[CoTExample, str, str]]:
        """Select 3 demonstrations from other task-subtask combinations (not target task-subtask)."""
        target_main_task, target_subtask = target_task
        
        # Get available task-subtask combinations (excluding the target)
        available_combinations = [
            (task, subtask) for task, subtask in available_subtasks 
            if not (task == target_main_task and subtask == target_subtask)
        ]
        
        if len(available_combinations) < 3:
            print(f"Warning: Only {len(available_combinations)} task-subtask combinations available for cross-task demonstrations")
            available_combinations = [(task, subtask) for task, subtask in available_subtasks 
                                    if task != target_main_task]  # Use any different main task if needed
        
        # Randomly select 3 different task-subtask combinations
        selected_combinations = random.sample(available_combinations, min(3, len(available_combinations)))
        
        # From each selected task-subtask combination, randomly choose 1 CoT example
        selected_demos = []
        for task, subtask in selected_combinations:
            examples = self.cot_hub_data[task]
            selected_example = random.choice(examples)
            # Update the example with the specific subtask
            selected_example.subtask = subtask
            selected_demos.append((selected_example, task, subtask))
            
        return selected_demos
    
    def generate_cross_task_prompt(self, target_question: CoTExample, 
                                 demonstrations: List[Tuple[CoTExample, str, str]], 
                                 task_name: Tuple[str, str], question_id: int) -> Tuple[str, str, List[Dict[str, str]]]:
        """Generate cross-task prompt with specified format."""
        
        # Convert target question to binary choice
        formatted_target, true_answer = self.convert_to_binary_choice(
            target_question.question, target_question.answer
        )
        
        # Build prompt and collect demonstration info
        prompt_parts = []
        demo_info = []
        
        for i, (demo, demo_task, demo_subtask) in enumerate(demonstrations):
            # Convert demonstration to binary choice
            formatted_demo, demo_answer = self.convert_to_binary_choice(
                demo.question, demo.answer
            )
            
            prompt_parts.append(f"### Example {i+1}:")
            prompt_parts.append(f"Question: {formatted_demo}")
            prompt_parts.append(f"Reasoning: {demo.cot}")
            prompt_parts.append(f"Answer: {demo_answer}")
            prompt_parts.append("")
            
            # Store demonstration task/subtask info
            demo_info.append({
                "demo_task": demo_task,
                "demo_subtask": demo_subtask
            })
        
        prompt_parts.append("### New Question:")
        prompt_parts.append(f"Question: {formatted_target}")
        
        return "\n".join(prompt_parts), true_answer, demo_info
    
    def generate_all_prompts(self, target_count: int = 1000) -> List[Dict[str, Any]]:
        """Generate all cross-task prompts."""
        print(f"Generating {target_count} cross-task question-demonstration pairs...")
        
        generated_prompts = []
        generated_count = 0
        
        # Calculate questions per task (roughly equal distribution)
        questions_per_task = target_count // len(self.selected_tasks)
        remainder = target_count % len(self.selected_tasks)
        
        for i, (main_task, subtask) in enumerate(self.selected_tasks):
            # Determine how many questions for this task
            task_question_count = questions_per_task
            if i < remainder:  # Distribute remainder across first few tasks
                task_question_count += 1
            
            # Get examples for this main task
            available_examples = self.cot_hub_data[main_task]
            
            if not available_examples:
                print(f"Warning: No examples available for task {main_task}")
                continue
            
            # Generate questions for this task
            for j in range(task_question_count):
                if generated_count >= target_count:
                    break
                
                # Randomly sample a question from this task
                target_question = random.choice(available_examples)
                
                # Select 3 cross-task demonstrations
                demonstrations = self.select_cross_task_demonstrations((main_task, subtask), self.selected_tasks)
                
                # Generate prompt
                prompt, true_answer, demo_info = self.generate_cross_task_prompt(
                    target_question, demonstrations, (main_task, subtask), generated_count
                )
                
                # Create result entry with subtask and demonstration information
                result = {
                    "task": main_task,
                    "subtask": subtask,
                    "question_id": generated_count,
                    "prompt": prompt,
                    "true_answer": true_answer,
                    "demonstrations": demo_info
                }
                
                generated_prompts.append(result)
                generated_count += 1
                
                if generated_count % 100 == 0:
                    print(f"Generated {generated_count} prompts...")
        
        print(f"Generated {len(generated_prompts)} cross-task prompts")
        return generated_prompts
    
    def save_prompts(self, prompts: List[Dict[str, Any]], filename: str) -> None:
        """Save prompts to JSON file."""
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        filepath = os.path.join("data", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(prompts)} prompts to {filepath}")

def main():
    """Main function to generate cross-task prompts."""
    print("Starting cross-task prompt generation...")
    print(f"Using random seed: {RANDOM_SEED}")
    
    # Initialize generator
    generator = PromptGenerator()
    
    # Load data
    generator.load_cot_hub_data()
    
    # Select 28 tasks
    generator.select_28_tasks()
    
    # Generate prompts
    prompts = generator.generate_all_prompts(target_count=1000)
    
    # Save results
    generator.save_prompts(prompts, "prepared_prompts.json")
    
    # Print confirmation
    print("\n" + "="*50)
    print("GENERATION COMPLETE")
    print("="*50)
    print(f"Generated {len(prompts)} cross-task question-demonstration pairs.")
    print("Saved to data/prepared_prompts.json")
    print("="*50)

if __name__ == "__main__":
    main()