import gradio as gr
import pandas as pd
from src.model_pipeline import preprocess_closet, parse_prompt, generate_outfit, get_season, model
from src.data_validation import validate_closet_data
import requests
import numpy as np
import time
from datetime import datetime
import os
import csv
def log_metrics(inference_time, city, weather, style):
    os.makedirs("monitoring", exist_ok=True)
    log_path = "monitoring/metrics_log.csv"

    header = [
        "timestamp", "city", "temp_f", "condition",
        "is_rain", "is_hot", "is_cold", "style", "inference_time_sec"
    ]

    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:  
            writer.writerow(header)

        writer.writerow([
            datetime.now().isoformat(),
            city,
            weather["temp"],
            weather["condition"],
            weather["is_rain"],
            weather["is_hot"],
            weather["is_cold"],
            style,
            round(inference_time, 4)
        ])

def outfit_ui(city, prompt, file_obj):
    if file_obj:
        df = pd.read_csv(file_obj.name)
    else:
        df = pd.read_csv("data/closet.csv")

    df, _ = validate_closet_data(df)
    df = preprocess_closet(df, model)

    # Weather lookup
    API_KEY = "6f44360e000b4bf3847100146252311"
    r = requests.get(f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}")
    w = r.json()

    temp = w["current"]["temp_f"]
    condition = w["current"]["condition"]["text"].lower()

    weather = {
        "temp": temp,
        "condition": condition,
        "is_rain": any(x in condition for x in ["rain", "storm", "drizzle"]),
        "is_hot": temp > 80,
        "is_cold": temp < 70,
        "need_pants": temp < 56
    }

    style = parse_prompt(prompt)
    season = get_season()

    start = time.time()

    outfit = generate_outfit(df, weather, season, style, model)
    inference_time = time.time() - start
    print(f"Inference Time: {inference_time:.4f} seconds")
    log_metrics(inference_time, city, weather, style)

    result = f"### Weather in {city}: {temp}°F — {condition}\n"
    result += f"**Season:** {season}\n"
    result += f"**Detected Style:** {style}\n\n"

    for cat, item in outfit.items():
        result += f"- **{cat}:** {item}\n"

    return result


def launch_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## ClosetAI — AI Personal Stylist")

        with gr.Row():
            city = gr.Textbox(label="City")
            prompt = gr.Textbox(label="Your plans")
            file_obj = gr.File(label="Upload closet CSV", file_types=[".csv"])

        output = gr.Markdown()
        btn = gr.Button("Generate Outfit")

        btn.click(outfit_ui, inputs=[city, prompt, file_obj], outputs=output)

    demo.launch()
