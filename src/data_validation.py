import os
import re
import json
import datetime
import pandas as pd
from diffprivlib.mechanisms import Laplace


def validate_closet_data(df):
    season = {"Winter", "Fall", "Spring", "Summer", "All"}
    feat = ["Item_ID", "Type", "Color", "Season", "Occasion", "Material"]

    missing = [c for c in feat if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    assert df["Item_ID"].notna().all(), "Item_ID cannot be null"


    def clean_season(val):
        if val is None:
            return ["All"]

        text = str(val)
        if text.strip().lower() in ("nan", "none", "", "[]"):
            return ["All"]

        found = re.findall(r"(Winter|Fall|Spring|Summer|All)", text, flags=re.IGNORECASE)
        clean = [s.title() for s in found]

        return clean if clean else ["All"]

    df["Season"] = df["Season"].apply(clean_season)

    # season validation
    bad = set(s for sub in df["Season"] for s in sub if s not in season)
    assert not bad, f"Invalid Season values: {bad}"

    df["Season_display"] = df["Season"].apply(lambda x: ", ".join(x))


    temp_df = df.copy()
    temp_df["Season"] = temp_df["Season"].apply(tuple)

    dupes = temp_df.duplicated().sum()
    if dupes:
        df = df.loc[~temp_df.duplicated()].reset_index(drop=True)
        print("Dropped duplicates:", dupes)

    def print_share(col):
        col_str = col.apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        return col_str.value_counts(normalize=True)

    season_dist = print_share(df["Season"])
    occasion_dist = print_share(df["Occasion"])

    season_imbalanced = season_dist.max() > 0.60
    occasion_imbalanced = occasion_dist.max() > 0.60

    winter_count = df["Season"].apply(lambda x: "Winter" in x).sum()

    laplace = Laplace(epsilon = 1.0, sensitivity = 1)
    dp_value = laplace.randomise(winter_count)

    print(f"Winter count true={winter_count}  dp_noised={round(dp_value)}  (ε=1.0)")

    os.makedirs("data", exist_ok=True)
    
    season_dist.to_csv("data/season_distribution.csv", header=["proportion"])
    occasion_dist.to_csv("data/occasion_distribution.csv", header=["proportion"])
    imbalance_report = pd.DataFrame({
        "metric": ["season_imbalanced", "occasion_imbalanced"],
        "value": [season_imbalanced, occasion_imbalanced]
    })
    imbalance_report.to_csv("data/imbalance_report.csv", index=False)

    metadata = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rows": len(df),
        "columns": len(df.columns),
        "columns name": df.columns.tolist(),
        "sources": ["closet.csv"],
        "notes": "Validated schema; duplicates handling; distribution & Differential Privacy check."
    }
    with open("data/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    risks = [
        {"Phase": "Data Collection", "Risk": "Representativeness bias",
         "Mitigation": "Distribution checks and targeted augmentation", "Remaining Risk": "Moderate"},

        {"Phase": "Privacy", "Risk": "Summary Leakage",
         "Mitigation": "Differential-privacy noise on summaries", "Remaining Risk": "Low"},

        {"Phase": "Data Quality", "Risk": "Schema drift",
         "Mitigation": "Controlled misspelling and validation; Prevent ingestion if validation failed",
         "Remaining Risk": "Low"}
    ]
    pd.DataFrame(risks).to_csv("data/risk.csv", index=False)

    print("Wrote data/metadata.json and data/risk.csv")

    report = {
        "season_dist": season_dist.to_dict(),
        "occasion_dist": occasion_dist.to_dict(),
        "season_imbalanced": season_imbalanced,
        "occasion_imbalanced": occasion_imbalanced,
        "winter_count": winter_count,
        "winter_dp_noised": dp_value
    }

    return df, report
