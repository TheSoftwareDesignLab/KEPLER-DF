import pathlib
from typing import List, Dict, Any, Optional
from src.core.datatypes import TargetTask
from .generator import generate_ollama_semantic_prompt

__all__ = ["prompt_factory_main"]


def prompt_factory_main(
    targets: List[TargetTask],
    prompt_config: Optional[Dict[str, Any]],
    output_dir: str = "data",
    model_name: str = "llama3.1:8b",
    temperature: float = 0.3
) -> Dict[str, str]:
    if prompt_config is None:
        return {}

    system_instruction = prompt_config.get("system_instruction_template", "")
    if not system_instruction:
        raise ValueError("The YAML configuration is missing the 'system_instruction_template' key.")

    prompts_map = generate_ollama_semantic_prompt(
        targets=targets,
        system_instruction_template=system_instruction,
        model_name=model_name,
        temperature=temperature
    )

    dir_path = pathlib.Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    for task_id, clean_text in prompts_map.items():
        file_path = dir_path / f"ollama_prompt_{task_id}.txt"
        file_path.write_text(clean_text, encoding="utf-8")

    print(f"\n[SUCCESS] Saved {len(prompts_map)} independent human request files to directory: {dir_path.resolve()}")
    return prompts_map