import os
import json
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional
from src.core.datatypes import TargetTask

__all__ = ["generate_ollama_semantic_prompt", "build_single_task_string"]


def _format_remaining_time(deadline_utc: datetime, now_utc: datetime) -> str:
    delta = deadline_utc - now_utc
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "past due"
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def build_single_task_string(task: TargetTask, now_utc: datetime) -> str:
    local_tz = ZoneInfo("America/Bogota")
    clean_name = task.region_tag.replace("_", " ")
    primary_sensor = task.required_sensors[0] if task.required_sensors else "VIS"
    
    task_deadline_utc = datetime.fromtimestamp(task.deadline, tz=timezone.utc)
    task_deadline_local = task_deadline_utc.astimezone(local_tz)
    
    remaining_str = _format_remaining_time(task_deadline_utc, now_utc)
    
    return f"{clean_name}|{primary_sensor}|{task_deadline_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}|{task_deadline_local.strftime('%Y-%m-%d %H:%M (%Z)')}|{remaining_str}"


def generate_ollama_semantic_prompt(
    targets: List[TargetTask],
    system_instruction_template: str,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.3,
    num_predict: int = 700,
    repeat_penalty: float = 1.05
) -> Dict[str, str]:
    raw_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    clean_url = raw_url.replace("[", "").replace("]", "").split("(")[0].strip()
    target_endpoint = f"{clean_url}/api/generate"
    
    generated_prompts_map = {}
    
    for task in targets:
        now_utc = datetime.now(timezone.utc)
        task_string = build_single_task_string(task, now_utc)
        full_prompt = system_instruction_template.format(tasks_dataset=task_string)
        
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "repeat_penalty": repeat_penalty
            }
        }

        print(f"\n[OLLAMA] Generating prompt for {task.task_id}...")
        response = requests.post(
            target_endpoint,
            json=payload,
            stream=True,
            timeout=180
        )
        response.raise_for_status()

        output_chunks = []
        for line in response.iter_lines():
            if not line:
                continue
            line_object = json.loads(line)
            if "response" not in line_object:
                continue
            
            chunk = line_object["response"]
            print(chunk, end="", flush=True)
            output_chunks.append(chunk)

            if line_object.get("done"):
                break

        print()
        raw_text = "".join(output_chunks)
        
        if "```prompt" in raw_text:
            parsed_text = raw_text.split("```prompt")[1].split("```")[0].strip()
        else:
            parsed_text = raw_text.strip()
            
        generated_prompts_map[task.task_id] = parsed_text
        
    return generated_prompts_map