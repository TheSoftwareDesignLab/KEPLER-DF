import sys
import time
from datetime import datetime, timezone
from src.core.datatypes import TargetTask
from src.modules.prompt_factory.generator import generate_ollama_semantic_prompt


def main():
    print("==================================================")
    print("Running Ollama Live Integration & Output Parser Test")
    print("==================================================")

    current_unix = int(time.time())
    
    mock_targets = [
        TargetTask(
            task_id="TASK_MOCK_001",
            region_tag="colombia_andina",
            priority=5,
            task_type="point",
            coordinates=[(4.6097, -74.0817)],
            required_sensors=["SAR"],
            release_time=0,
            deadline=current_unix + 5400
        ),
        TargetTask(
            task_id="TASK_MOCK_002",
            region_tag="panama_canal",
            priority=2,
            task_type="polygon",
            coordinates=[(8.5, -80.0), (8.5, -79.0)],
            required_sensors=["VIS"],
            release_time=60,
            deadline=current_unix + 90000
        )
    ]

    system_instruction_input = """Return ONLY one block delimited by triple backticks with the language tag `prompt`.

You are an Earth-observation mission operator talking to an automated planner for a LEO satellite constellation.

Your task is to write a single, natural-sounding English request, as if you were asking the planner for help. The request must:

1. Explain that you need a prioritized, conflict-free plan for captures, uplinks, and downlinks that minimizes idle time, avoids resource conflicts, and matches satellite and ground-station capabilities.

2. Mention EVERY task from the dataset by its location name and sensor type (for example: "Brussels NIR", "SanJose VIS", etc.). Do not invent extra tasks.

3. For each task, explicitly state how much time is left until its deadline by REUSING the exact numeric string in the last column of the dataset (the "remaining" field), which is written like: `Xd Yh Zm`, `Xh Ym` or `Xm`. You MUST keep the numbers and units exactly as they appear in that last column. You may wrap it in short phrases like "with roughly 2h 18m left" but you MUST NOT change the digits or units.

4. Keep the tone conversational and compact, as if you were summarizing the situation to the planner. Integrate the information into flowing narrative sentences instead of bullets or tables.

Tasks dataset:
{tasks_dataset}

Now output ONLY the final user request, inside:
```prompt
...your text here...
"""

    print("\n[OLLAMA INFERENCE START] Streaming responses from local model...")
    print("-" * 50)
    
    try:
        prompts_map = generate_ollama_semantic_prompt(
            targets=mock_targets,
            system_instruction_template=system_instruction_input,
            model_name="llama3.1:8b",
            temperature=0.3
        )
        
        print("-" * 50)
        print("[OLLAMA INFERENCE END]")
        
        failed_sanitization = False

        for task_id, clean_human_request in prompts_map.items():
            print(f"\n--- [FINAL PARSED FILE CONTENT FOR {task_id}] ---")
            print(clean_human_request)
            print("-------------------------------------------------")
            
            if "```" in clean_human_request or "Tasks dataset:" in clean_human_request:
                print(f"[FAIL] Post-processing did not sanitize structural tokens for {task_id}.")
                failed_sanitization = True

        if failed_sanitization:
            sys.exit(1)
        else:
            print("\n[SUCCESS] Integration test passed. Outputs are purely human narrative.")

    except Exception as e:
        print(f"\n[CRASH] Connection or parsing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()