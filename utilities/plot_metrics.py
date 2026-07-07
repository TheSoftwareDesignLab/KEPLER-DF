import json
import pathlib
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["generate_metrics_chart"]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"] + plt.rcParams["font.serif"]


def _discover_and_parse_metrics(data_root_str: str = "data") -> Dict[str, Dict[str, List[float]]]:
    root_path = pathlib.Path(data_root_str)
    
    print(f"[DEBUG] Checking data folder at: {root_path.resolve()}")
    
    if not root_path.exists():
        fallback_path = (pathlib.Path(__file__).resolve().parent.parent / "data").resolve()
        print(f"[DEBUG] Path '{data_root_str}' not found. Trying fallback path: {fallback_path}")
        root_path = fallback_path

    model_metrics = {}
    
    print(f"[DEBUG] Final path used for scanning: {root_path.resolve()} (Exists: {root_path.exists()})")

    for const_dir in root_path.glob("constellation_*"):
        if not const_dir.is_dir() or const_dir.name == "constellation_dataset_gemma2_27b_v2":
            print(f"[DEBUG] Skipping or ignoring directory: {const_dir.name}")
            continue

        model_name = const_dir.name
        print(f"[DEBUG] Processing constellation folder: {model_name}")

        if model_name not in model_metrics:
            model_metrics[model_name] = {
                "sensor": [],
                "priority": [],
                "day": [],
                "hour": []
            }

        scenario_count = 0
        for scenario_dir in const_dir.glob("scenario_*"):
            if not scenario_dir.is_dir():
                continue
            
            scenario_count += 1
            target_json = scenario_dir / "ollama_prompts_combined.json"

            if not target_json.exists():
                continue

            try:
                with target_json.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    metrics = data.get("scenario_metrics", {})
                    
                    s_acc = metrics.get("global_sensor_accuracy")
                    p_acc = metrics.get("global_priority_accuracy")
                    d_acc = metrics.get("global_day_accuracy")
                    h_acc = metrics.get("global_hour_accuracy")

                    if all(v is not None for v in [s_acc, p_acc, d_acc, h_acc]):
                        model_metrics[model_name]["sensor"].append(float(s_acc))
                        model_metrics[model_name]["priority"].append(float(p_acc))
                        model_metrics[model_name]["day"].append(float(d_acc))
                        model_metrics[model_name]["hour"].append(float(h_acc))
            except Exception:
                continue
                
        print(f"[DEBUG] Total scenarios successfully evaluated for {model_name}: {scenario_count}")

    return model_metrics


def generate_metrics_chart(data_dir: str = "data", output_image_path: str = "utilities/output/model_accuracy_comparison.png") -> None:
    raw_data = _discover_and_parse_metrics(data_dir)
    
    if not raw_data:
        print("[ERROR] No valid data found in constellation folders.")
        return

    models = sorted(list(raw_data.keys()))
    categories = ["Day", "Hour", "Sensor", "Priority"]
    
    processed_averages = {model: [] for model in models}
    for model in models:
        m_data = raw_data[model]
        processed_averages[model].append(np.mean(m_data["day"]) if m_data["day"] else 0.0)
        processed_averages[model].append(np.mean(m_data["hour"]) if m_data["hour"] else 0.0)
        processed_averages[model].append(np.mean(m_data["sensor"]) if m_data["sensor"] else 0.0)
        processed_averages[model].append(np.mean(m_data["priority"]) if m_data["priority"] else 0.0)

    x = np.arange(len(categories))
    width = 0.22 if len(models) > 1 else 0.4
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    colors = ["#2e86c1", "#27ae60", "#e67e22", "#9b59b6", "#e74c3c"]

    for i, model in enumerate(models):
        offset = width * multiplier
        averages_pct = [val * 100.0 for val in processed_averages[model]]
        
        rects = ax.bar(
            x + offset, 
            averages_pct, 
            width, 
            label=model, 
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.8
        )
        ax.bar_label(rects, padding=4, fmt="%.1f%%", fontsize=9, fontweight="bold")
        multiplier += 1

    ax.set_ylabel("Semantic Matching Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("KEPLER Dataset Factory - LLM Semantic Generation Accuracy by Category", fontsize=13, fontweight="bold", pad=15)
    
    center_offset = (width * (len(models) - 1)) / 2.0 if len(models) > 1 else 0.0
    ax.set_xticks(x + center_offset)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, shadow=False, facecolor="#f8f9f9", edgecolor="#d5dbdb")

    out_path = pathlib.Path(output_image_path)
    if not out_path.is_absolute() and not pathlib.Path("utilities").exists():
        out_path = (pathlib.Path(__file__).resolve().parent / "output" / "model_accuracy_comparison.png").resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"[SUCCESS] Metrics chart generated at: {out_path.resolve()}")


if __name__ == "__main__":
    generate_metrics_chart()