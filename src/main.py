from src.data_validation import validate_closet_data
from src.model_pipeline import preprocess_closet, generate_outfit, parse_prompt, get_season
from src.ui import launch_ui

import pandas as pd


def run_pipeline(csv_path="data/closet.csv", city="Gainesville", prompt="going to a party"):

    df = pd.read_csv(csv_path)
    df, report = validate_closet_data(df)
    print("\n--- Validation Report ---")
    print(report)

    df = preprocess_closet(df)

    season = get_season()
    style = parse_prompt(prompt)

    weather = {
        "temp": 75,
        "condition": "clear",
        "is_rain": False,
        "is_hot": False,
        "is_cold": False,
        "need_pants": False
    }

    outfit = generate_outfit(df, weather, season, style)
    print("\n--- Outfit Recommendation ---")
    print(outfit)


if __name__ == "__main__":
    launch_ui()
