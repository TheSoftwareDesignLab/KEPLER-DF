import json
import pathlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from src.core.datatypes import TargetTask
from .generator import generate_ollama_semantic_prompt, build_single_task_string
from .validation import SemanticValidator

__all__ = ["prompt_factory_main"]


def prompt_factory_main(
    targets: List[TargetTask],
    prompt_config: Optional[Dict[str, Any]],
    output_dir: str = "data",
    model_name: str = "llama3.1:8b",
    temperature: float = 0.3,
    sensor_categories: Optional[Dict[str, str]] = None 
) -> Dict[str, str]:
    if prompt_config is None:
        return {}

    system_instruction = prompt_config.get("system_instruction_template", "")
    if not system_instruction:
        raise ValueError("The YAML configuration is missing the 'system_instruction_template' key.")


    urgency_categories = ["low", "medium", "high"]
    priority_map = {1: "low", 2: "medium", 3: "high"}

    validator = SemanticValidator(
        sensor_categories=sensor_categories,
        urgency_categories=urgency_categories
    )

    prompts_map = generate_ollama_semantic_prompt(
        targets=targets,
        system_instruction_template=system_instruction,
        model_name=model_name,
        temperature=temperature
    )

    dir_path = pathlib.Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    task_lookup = {task.task_id: task for task in targets}
    now_utc = datetime.now(timezone.utc)

    processed_tasks_list = []
    successful_sensor_matches = 0
    successful_urgency_matches = 0

    for task_id, clean_text in prompts_map.items():
        txt_file_path = dir_path / f"ollama_prompt_{task_id}.txt"
        txt_file_path.write_text(clean_text, encoding="utf-8")

        task = task_lookup.get(task_id)
        if task:
            task_string = build_single_task_string(task, now_utc)
            full_prompt_sent = system_instruction.format(tasks_dataset=task_string)
            
          
            raw_sensor = task.required_sensors[0] if task.required_sensors else "VISUAL"
            expected_sensor = "VISUAL" if raw_sensor == "VIS" else raw_sensor
            
            raw_priority = task.priority
            expected_urgency = priority_map.get(raw_priority, "low")

            metrics = validator.validate_generated_request(clean_text)
            
            sensor_match = metrics["predicted_sensor"] == expected_sensor
            urgency_match = metrics["predicted_urgency"] == expected_urgency

            if sensor_match:
                successful_sensor_matches += 1
            if urgency_match:
                successful_urgency_matches += 1

            processed_tasks_list.append({
                "task_id": task_id,
                "prompt_sent": full_prompt_sent,
                "generated_output": clean_text,
                "ground_truth": {
                    "expected_sensor": expected_sensor,
                    "expected_urgency": expected_urgency
                },
                "validation": {
                    "predicted_sensor": metrics["predicted_sensor"],
                    "sensor_similarity": metrics["sensor_similarity"],
                    "sensor_match": sensor_match,
                    "predicted_urgency": metrics["predicted_urgency"],
                    "urgency_similarity": metrics["urgency_similarity"],
                    "urgency_match": urgency_match
                }
            })

    if processed_tasks_list:
        total_tasks_count = len(processed_tasks_list)
        
        scenario_summary = {
            "total_tasks_evaluated": total_tasks_count,
            "global_sensor_accuracy": float(successful_sensor_matches / total_tasks_count),
            "global_urgency_accuracy": float(successful_urgency_matches / total_tasks_count),
            "successful_sensor_matches": successful_sensor_matches,
            "successful_urgency_matches": successful_urgency_matches
        }

        combined_payload = {
            "timestamp_utc": now_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "scenario_metrics": scenario_summary,
            "tasks": processed_tasks_list
        }
        
        json_file_path = dir_path / "ollama_prompts_combined.json"
        json_file_path.write_text(
            json.dumps(combined_payload, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    print(f"\n[SUCCESS] Saved {len(prompts_map)} independent human request TXT files and 1 unified JSON catalog with semantic validation metrics to directory: {dir_path.resolve()}")
    return prompts_map