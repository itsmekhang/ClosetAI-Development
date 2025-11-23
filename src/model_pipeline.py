import numpy as np
import datetime
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import spacy

model = SentenceTransformer("intfloat/e5-base-v2")
nlp = spacy.load("en_core_web_sm")

def get_season(date=None):
    if date is None:
        date = datetime.datetime.now()  # this ensures date is a datetime object
    month = date.month  # get month safely

    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def parse_prompt(prompt):
    doc = nlp(prompt.lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop]

    style = "Casual"  # default
    if any(t in tokens for t in ["formal",  "fancy", "dressy",  "event"]):
        style = "Formal"
    elif any(t in tokens for t in ["business", "office", "job", "interview"]):
        style = "Business"
    elif any(t in tokens for t in ["party", "night", "club", "event", "fancy"]):
        style = "Party"
    elif any(t in tokens for t in ["workout", "gym", "run", "athletic", "sport", "training", "basketball", "football", "soccer", "baseball"]):
        style = "Sport"
    elif any(t in tokens for t in ["chill", "relax", "casual", "everyday", "streetwear"]):
        style = "Casual"
    elif any(t in tokens for t in ["hot", "sunny", "beach", "ocean", "sand", "swim", "swimming"]):
        style = "Swim"

    return style

import pandas as pd, json, datetime, os
import numpy as np
from sentence_transformers import SentenceTransformer, util
import requests
import spacy
from diffprivlib.mechanisms import Laplace
import random
import os, sys, asyncio
import re



model = SentenceTransformer("intfloat/e5-base-v2")
def preprocess_closet(df, model):
    category_desc = {
        "Top": "shirts, t-shirts, blouses, tank tops, or anything worn on the upper body",
        "Bottom": "pants, jeans, shorts, skirts, joggers or clothing for the lower body",
        "Shoes": "More business casual or formal closed-toe footware",
        "Outerwear": "jackets, coats, blazers, hoodies, or anything worn over other clothing",
        "Accessory": "items that complement outfits such as hats, belts, bags, or sunglasses",
        "Sneakers": "athletic or casual shoes such as Jordans, Air Max, Yeezys, or basketball shoes",
    }
    categories = list(category_desc.keys())
    anchors = list(category_desc.values())
    
    cat_embeds = model.encode(anchors, convert_to_tensor=True, normalize_embeddings=True)
    occasion_desc = {
        "Casual": "comfortable everyday clothing for daily wear, not athletic or formal, includes jeans and t-shirts",
        "Formal": "professional, elegant, dressy clothing like suits or gowns, not sporty",
        "Sport": "clothing made for athletic activity, gym, running, or sports training, flexible fabric, sneakers only, no formal or business clothes",
        "Party": "night out or stylish event clothing, fashionable but not athletic, no sandals or flip flops",
        "Swim": "beach or pool clothing, swimsuits, swim trunks, flip flops, no closed-toe shoes or outerwear",
        "Business": "office or meeting attire, smart and professional, suits, blazers, loafers, not casual sneakers, any pants ",
        "All": "neutral versatile clothing suitable for many occasions; though sneakers cannot be formal or business",
    }
    occasions = list(occasion_desc.keys())
    occasion_embeds = model.encode(
        list(occasion_desc.values()),
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    
    def ai_categorize_text(item_name):
        name = item_name.lower()
        sneaker_keywords = ["air", "jordan", "lebron", "yeezy", "boost", "max", "trainer", "sneaker", "kyrie", "kd", "ultraboost"]
        shoe_keywords = ["shoe", "sandal", "loafer", "boot", "flip", "heel", "dress shoes"]
        swim_keywords = ["swim", "trunk", "trunks", "boardshort", "hawaiian", "bikini", "rashguard"]
    
        if any(word in name for word in swim_keywords):
            if any(k in name for k in ["trunk", "trunks", "boardshort", "bikini"]):
                return "Bottom"
            elif any(k in name for k in ["rashguard", "hawaiian"]):
                return "Top"
            else:
                return "Top"
    
        if any(word in name for word in sneaker_keywords):
            return "Sneakers"
        if any(word in name for word in shoe_keywords):
            return "Shoes"
        
            
        text = f"{item_name}: a piece of clothing"
        item_vec = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(
            model.encode(text, convert_to_tensor=True, normalize_embeddings=True),
            cat_embeds
        )[0].cpu().numpy()
        return categories[int(np.argmax(sims))]
    
    manual_overrides = {
        r"Blazer": "Outerwear",
        r"Suit Jacket": "Outerwear",
        r"Air Force 1": "Sneakers",
        r"Chinos": "Bottom",
        r"Shorts": "Bottom",
        r"Wetsuit": "Swimwear",
        r"Jordan": "Sneakers",
        r"Swim": "Swim"
    }
    def ai_categorize_occasion(item_name, top_k=2, threshold=0.814):
        text = f"{item_name}: type of clothing and what situation it fits best"
        item_vec = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(item_vec, occasion_embeds)[0].cpu().numpy()
#set tuning weight    
        occasion_weights = sims.copy()
    
        if "jeans" in item_name.lower():
            # Jeans → Casual or Business Casual, not Sport
            for i, occ in enumerate(occasions):
                if occ == "Sport":
                    occasion_weights[i] -= 0.5
                if occ in ["Casual", "Business"]:
                    occasion_weights[i] += 0.3
                    
        if any(k in item_name.lower() for k in ["short", "gym", "athletic", "training", "workout", "joggers", r"Sweat", r"Sweatpants", r"ball"]):
            for i, occ in enumerate(occasions):
                if occ == "Sport":
                    occasion_weights[i] += 2.0
                elif occ == "Casual":
                    occasion_weights[i] += 0.3  # small secondary boost
        elif any(k in item_name.lower() for k in ["pant", r"trouser", r"slack", r"chinos"]):
            # Pants → can be Business or Formal
            for i, occ in enumerate(occasions):
                if occ in ["Business", "Formal"]:
                    occasion_weights[i] += 0.5
                if occ == "Casual":
                    occasion_weights[i] -= 0.2
                if occ == "Sport":
                    occasion_weights[i] -= 0.8
    
        if any(k in item_name.lower() for k in ["button", "dress", "collar", "oxford", "blazer"]):
            # Dress shirts are Formal or Business
            for i, occ in enumerate(occasions):
                if occ in ["Business", "Formal"]:
                    occasion_weights[i] += 1.0
                elif occ == "Party":
                    occasion_weights[i] += 0.32
                elif occ == "Casual":
                    occasion_weights[i] += 0.3
                elif occ == "Swim":
                    occasion_weights[i] += 0.4
        if any(k in item_name.lower() for k in ["dress shoes", "loafers"]):
            for i, occ in enumerate(occasions):
                if occ in ["Formal", "Business"]:
                    occasion_weights[i] += 0.6   # strong fit
                elif occ == "Party":
                    occasion_weights[i] += 0.2  # acceptable for upscale parties
                elif occ == "Casual":
                    occasion_weights[i] -= 0.8   # overdressed for casual
                elif occ == "Sport":
                    occasion_weights[i] -= 1   # completely inappropriate
                elif occ == "Swim":
                    occasion_weights[i] -= 1.0   # never for swim/beach
    
        if any(k in item_name.lower() for k in ["hoodie", "t-shirt", "joggers"]):
            # Hoodies, joggers, sweaters are casual
            for i, occ in enumerate(occasions):
                if occ == "Casual":
                    occasion_weights[i] += 0.8
                if occ == "Sport":
                    occasion_weights[i] += 1
                if occ in ["Formal", "Business"]:
                    occasion_weights[i] -= 1
        if any(k in item_name.lower() for k in ["sweat"]):
            for i, occ in enumerate(occasions):
                if occ == "Casual":
                    occasion_weights[i] += 0.8
                if occ == "Sport":
                    occasion_weights[i] += 0.5
            
        if any(word in item_name.lower() for word in ["air", "jordan", "lebron", "yeezy", "boost", "max", "trainer", "run", "sneaker"]):
            for i, occ in enumerate(occasions):
                if occ == "Party":
                    occasion_weights[i] += 0.5
                elif occ == "Casual":
                    occasion_weights[i] += 0.3
                elif occ == "Sport":
                    occasion_weights[i] += 1
                return "Casual, Party, Sport"
    
        # --- Select top occasions ---
        top_indices = occasion_weights.argsort()[::-1][:top_k]
        top_matches = [(occasions[i], occasion_weights[i]) for i in top_indices if occasion_weights[i] >= threshold]
    
        if not top_matches:
            return occasions[int(occasion_weights.argmax())]
        return ", ".join([m[0] for m in top_matches])
    df["Occasion"] = df["Type"].apply(ai_categorize_occasion)
    
    df["Category"] = df["Type"].apply(ai_categorize_text)
    df["Category"] = df.apply(
    lambda x: next(
        (v for k, v in manual_overrides.items() if re.search(k, x["Type"], re.I)),
        x["Category"]
    ),
    axis=1
)
    df.loc[df["Type"].str.contains("Swim", case=False, na=False),
        "Occasion"] = "Swim"
    df.loc[df["Type"].str.contains(r"t-shirt", case=False, na=False),
        "Occasion"] = "Party, Casual, Sport"
    df.loc[df["Type"].str.contains(r"sweat", case=False, na=False),
        "Occasion"] = "Party, Casual, Sport"
    df.loc[df["Type"].str.contains("sports jersey", case=False, na=False),
        "Occasion"] = "Party, Casual, Sport"
    df.loc[df["Category"].str.contains("Sneakers", case=False, na=False),
        "Occasion"] = "Casual, Party, Sport"    
    df.loc[df["Type"].str.contains("Polo shirt|shirt", case=False, na=False),
        "Occasion"] = "Casual, Party, Business"
    df.loc[df["Type"].str.contains("hawaiian", case=False, na=False),
        "Occasion"] = "Casual, Party"
    df.loc[
        df["Type"].str.contains(r"dress|oxford|loafer|derby|wingtip", case=False, na=False),
        "Category"
    ] = "Shoes"
    #print the sorted closet to check for accuracy
    print(df[["Type", "Category", "Occasion"]])
    return df
    pass

def recommend_outfit(df, weather, season, style):
    df = df.copy()
    df["score"] = 0.0

    if weather.get("is_hot", False):
        df.loc[
            ~df["Type"].str.contains("Jacket|Sweater|Coat|Hoodie|Trench", case=False, na=False),
            "score"
        ] += 1
    elif weather.get("is_cold", False):
        df.loc[
            df["Type"].str.contains("Jacket|Sweater|Hoodie|Coat|Trench", case=False, na=False),
            "score"
        ] += 1
        df.loc[
            df["Type"].str.contains("short|trunk|boardshort|swim", case=False, na=False),
            "score"
        ] -= 1
    elif weather.get("is_rain", False):
        df.loc[
            df["Type"].str.contains("Raincoat|Waterproof|Boot", case=False, na=False),
            "score"
        ] += 1
    else:
        df["score"] += 0.5  # neutral weather

    # --- SEASON SCORING ---
    df["Season"] = df["Season"].fillna("").astype(str)
    df.loc[df["Season"].str.contains(season, case=False, na=False), "score"] += 1
    df.loc[df["Season"].str.contains("All", case=False, na=False), "score"] += 0.5

    # --- OCCASION / STYLE SCORING ---
    df["Occasion"] = df["Occasion"].fillna("").astype(str)
    df.loc[df["Occasion"].str.contains(style, case=False, na=False), "score"] += 1.2
    df.loc[df["Occasion"].str.contains("Casual|All", case=False, na=False), "score"] += 0.5

    if style.lower() == "party":
        df.loc[df["Occasion"].str.contains("Casual|Business|Sport|All", case=False, na=False), "score"] += 0.4
    elif style.lower() == "formal":
        df.loc[df["Occasion"].str.contains("Business|All", case=False, na=False), "score"] += 0.4
        df.loc[df["Type"].str.contains("Hoodie|Sweater", case=False, na=False), "score"] -= 0.7

    # --- SWIM FILTER ---
    if style.lower() == "swim":
        df = df[
            df["Occasion"].str.contains("Swim|All", case=False, na=False)
            | df["Type"].str.contains("swim", "trunk", "flip flop", "sandal", "hawaiian", "rashguard", "bikini", case=False, na=False)
        ]
        df = df[
            ~df["Type"].str.contains("jean|chino|pant|slack|trouser|loafer|dress|oxford|boot", case=False, na=False)
        ]
        df.loc[df["Category"].str.lower() == "outerwear", "score"] = -1

    df = df.sort_values("score", ascending=False)
    recommended = df[df["score"] > 0].copy()

    if recommended.empty:
        print("No strong match found — showing top-rated items.")
        recommended = df.sort_values("score", ascending=False).head(5)

    return recommended
    pass
def generate_outfit(df, weather, season, style, model):
    is_rain = weather.get("is_rain", False)
    is_hot = weather.get("is_hot", False)
    is_cold = weather.get("is_cold", False)
    need_pants = weather.get("need_pants", False)
    scored = recommend_outfit(df, weather, season, style).copy()
    if "Category" not in scored.columns:
        print("No category column found.")
        return {}

    # Calculate similarity
    style_vec = model.encode(f"{style} outfit", convert_to_tensor=True, normalize_embeddings=True)
    occ_vecs  = model.encode(scored["Occasion"].fillna("").tolist(), convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(style_vec, occ_vecs)[0].cpu().numpy()
    scored["style_sim"] = np.clip(sims, 0, 0.4)
    scored["total_score"] = scored["score"] + (0.5 * scored["style_sim"])

    style_lower = style.lower()
    is_cold = weather.get("is_cold", False)
    is_hot = weather.get("is_hot", False)
    need_pants = weather.get("need_pants", False)

    if any(w in style_lower for w in ["sport", "gym", "soccer", "basketball", "training", "run", "workout"]):
        scored = scored[~scored["Category"].str.lower().isin(["Shoes"])]
        scored = scored[
            ~scored["Type"].str.contains(
                "button|shirt|blazer|coat|jacket|loafer|dress|oxford|boot|chino|slack|trouser|jeans",
                case=False, na=False
            )
        ]
        scored = scored[
            ~scored["Occasion"].str.contains(
                "Swim|Business|Formal",
                case=False, na=False
            )
        ]

        scored.loc[
            scored["Category"].str.lower().eq("sneakers") |
            scored["Type"].str.contains(r"sneaker|trainer|air|jordan|yeezy|boost|max|athletic|sport shoe", case=False, na=False),
            "total_score"
        ] += 10
        sneaker_keywords = r"sneaker|trainer|running shoe|athletic|sport shoe|air|max|boost|jordan|yeezy"
        nonsport_keywords = r"flip[- ]?flop|sandal|slide|loafer|dress|oxford|boot(?!sneaker)|heel"
        
        # Strong boost for sneakers and athletic shoes
        scored.loc[
            scored["Type"].str.contains(sneaker_keywords, case=False, na=False),
            "total_score"
        ] += 12
        
        # Hard penalty + remove non-sport shoes
        scored.loc[
            scored["Type"].str.contains(nonsport_keywords, case=False, na=False),
            "total_score"
        ] -= 25
        scored = scored[
            ~scored["Type"].str.contains(nonsport_keywords, case=False, na=False)
        ]
        # 3️⃣ Default: athletic shorts
        scored.loc[
            scored["Type"].str.contains("athletic short", case=False, na=False),
            "total_score"
        ] += 4

        if need_pants:
            scored.loc[
                scored["Type"].str.contains(r"jogger|sweatpants|track pant|warmup", case=False, na=False),
                "total_score"
            ] += 10
            # Penalize & remove shorts
            scored.loc[
                scored["Type"].str.contains("short|trunk|boardshort|swim", case=False, na=False),
                "total_score"
            ] -= 10
            scored = scored[
                ~scored["Type"].str.contains("short|trunk|boardshort|swim", case=False, na=False)
            ]

    elif "swim" in style_lower or "beach" in style_lower:
        scored = scored[~scored["Category"].str.lower().isin(["outerwear"])]
        scored = scored[~scored["Type"].str.contains("coat|blazer|jacket|loafer|dress", case=False, na=False)]
        scored.loc[
            scored["Type"].str.contains("flip|sandal|slide|hawaiian", case=False, na=False),
            "total_score"
        ] += 3

    elif "formal" in style_lower or "business" in style_lower:
        scored = scored[~scored["Category"].str.lower().isin(["sneakers"])]
        scored = scored[~scored["Type"].str.lower().isin(["Hoodie"])]
        scored.loc[
        scored["Type"].str.contains("t-shirt|hawaiian", case=False, na=False),
        "total_score"
    ] -= 5
        scored = scored[
        ~scored["Type"].str.contains("flip[- ]?flop|thong sandal", case=False, na=False)
    ]
        scored = scored[
        ~scored["Type"].str.contains("shorts|athletic|jeans", case=False, na=False)
    ]

    elif "casual" in style_lower:
        scored.loc[scored["Category"].str.contains("Sneakers", case=False, na=False), "total_score"] += 2
        scored.loc[scored["Category"].str.contains("Shoes", case=False, na=False), "total_score"] += 0.5
        scored.loc[scored["Type"].str.contains("dress shoes|loafers", case=False, na=False), "total_score"] -= 10
        scored.loc[scored["Type"].str.contains("sneakers", case=False, na=False), "total_score"] += 8
        scored = scored[
        ~scored["Type"].str.contains("flip[- ]?flop|thong sandal", case=False, na=False)
    ]
        scored.loc[scored["Type"].str.contains("shorts|athletic", case=False, na=False), "total_score"] -= 0.8



    elif "party" in style_lower:
        scored.loc[
            scored["Type"].str.contains(r"flip flops", case=False, na=False),
            "total_score"] -= 18
        scored.loc[
            scored["Type"].str.contains(r"Dress ?Shoes", case=False, na=False),
            "total_score"] -= 14
        scored.loc[
            scored["Type"].str.contains(r"loafers|derby|wingtips", case=False, na=False),
            "total_score"] -= 6
        scored.loc[
        scored["Type"].str.contains(
            r"sneaker|trainer|jordan|air\s*force|yeezy|boost|max|casual shoe|lebron",
            case=False, na=False
        ),
        "total_score"
    ] += 25
        scored.loc[scored["Category"].str.contains("Sneakers", case=False, na=False),
        "total_score"] += 6
        scored = scored[
        ~scored["Type"].str.contains(
            "flip[- ]?flops|thong sandals|slides|boots|heels|athletic shorts|sweat pants|swim|trunk",
            case=False, na=False
        )
    ]
        scored = scored[
        ~scored["Type"].str.contains(r"running|shorts|athletic|swim|trunks", case=False, na=False)
    ]
        
    if not is_rain:
        scored.loc[
                scored["Type"].str.contains(r"rain", case=False, na=False),
                "total_score"
            ] -= 10.0
        
    if is_hot:
    #prioritize lightweight tops when it's hot
        scored = scored[~scored["Category"].str.lower().isin(["outerwear"])]
    
        # Big boost for t-shirts, polos, and short sleeves
        scored.loc[
            scored["Type"].str.contains("t-shirt|polo|short sleeve|jersey", case=False, na=False),
            "total_score"
        ] += 8.0
    
        # reduce sweaters, hoodies, or coats score
        scored.loc[
            scored["Type"].str.contains("sweater|hoodie|coat|jacket|trench", case=False, na=False),
            "total_score"
        ] -= 10.0
    
        # Re-grade after weather
        scored = scored.sort_values("total_score", ascending=False)
    if is_cold:
        if "formal" in style_lower or "business" in style_lower:
            scored.loc[
                scored["Type"].str.contains("sweater|hoodie|t-shirt", case=False, na=False),
                "total_score"
            ] -= 10.0
            
        else:
            top_items = scored[
            scored["Category"].str.lower().isin(["top", "outerwear"])
        ].nlargest(10, "total_score")
            
            hoodie_rows = scored["Type"].astype(str).str.contains("hoodie", case=False, na=False)
            sweater_rows = scored["Type"].astype(str).str.contains("sweater", case=False, na=False)
 
            has_hoodie = bool(hoodie_rows.any())
            has_sweater = bool(sweater_rows.any())
            if has_hoodie and has_sweater:
        # Compare top scores for each type
                top_hoodie = float(scored.loc[hoodie_rows, "total_score"].max())
                top_sweater = float(scored.loc[sweater_rows, "total_score"].max())
        
                if top_hoodie >= top_sweater:
                    #if hoodie is selected, remove sweaters
                    scored = scored.loc[~sweater_rows].copy()
                    scored.loc[scored["Type"].str.contains("long sleeve|crewneck|thermal|t-shirt", case=False, na=False),
                "total_score"] +=6
                    
                else:
                    #if sweater is chosen, remove hoodie
                    scored = scored.loc[~hoodie_rows].copy()
                    
            if has_hoodie:
                scored.loc[
                scored["Type"].str.contains("hoodie", case=False, na=False),
                ["Category", "total_score"]
            ] = ["Outerwear", 4]
                scored.loc[
                scored["Type"].str.contains("sweater", case=False, na=False),
                "total_score"
            ] -= 10
                scored.loc[scored["Type"].str.contains("long sleeve|crewneck|thermal", case=False, na=False),
                "total_score"] += 4

            elif has_sweater:
                print("🧶 Cold detected — sweater chosen (no hoodie).")
                scored = scored[~scored["Type"].str.contains("hoodie|jacket|coat|trench", case=False, na=False)]
                scored.loc[
                scored["Type"].str.contains("sweater", case=False, na=False),
                "total_score"
            ] += 4
                scored.loc[scored["Type"].str.contains("long sleeve|crewneck|thermal|t-shirt", case=False, na=False),
                "total_score"] += 7

            else:
                if style_lower in ["Casual", "Sport"] and not is_cold:
                    scored.loc[
                        scored["Type"].str.contains("t-shirt", case=False, na=False),
                        "total_score"
                    ] += 3
                scored.loc[
                scored["Type"].str.contains("long sleeve|crewneck|thermal|t-shirt", case=False, na=False),
                "total_score"
            ] += 1.0
        
# Putting the whole outfit together
    scored["total_score"] = scored["total_score"] + np.random.uniform(-0.7, 0.7, len(scored))
    scored = scored.sort_values("total_score", ascending=False)
    outfit = {}
    top_item = None
    for cat in ["Top", "Bottom", "Outerwear", "Accessory", "Shoes"]:
        subset = scored[scored["Category"].str.lower() == cat.lower()]
        if subset.empty:
            outfit[cat] = ""
            continue
    
        #usually can't wear polo shirts with athletic bottoms
        if cat.lower() == "bottom" and top_item:
                top_lower = top_item.lower()
        
                # Rule 1: Polo / Button-up / Dress shirt → NO athletic bottoms
                if re.search(r"polo|Button|dress shirt|long-sleve", top_lower):
                    subset = subset[
                        ~subset["Type"].str.contains(
                            "athletic shorts|joggers|sweat|gym shorts|training|track|swim|sweatpants",
                            case=False, na=False
                        )
                    ]
                    # score boost for nicer pants
                    subset.loc[
                        subset["Type"].str.contains("chino|slack|trouser|khaki|pant", case=False, na=False),
                        "total_score"
                    ] += 5
        
                # Rule 2: Cannot wear hawaiian shirt with athletic bottoms, only casual shorts/pants
                elif "hawaiian shirt" in top_lower:
                    subset = subset[
                        ~subset["Type"].str.contains(
                            "running|athletic shorts|joggers|sweat|gym short|training|track",
                            case=False, na=False
                        )
                    ]
                    subset.loc[
                        subset["Type"].str.contains("chino|linen|khaki|short", case=False, na=False),
                        "total_score"
                    ] += 2
                    
        top_n = subset.nlargest(5, "total_score")
        chosen = top_n.sample(1).iloc[0]
        item_type = chosen["Type"]      
        outfit[cat] = item_type
        if cat.lower() == "top":
            top_item = item_type
        
        if cat.lower() == "Shoes" and any(
            word in str(item_type).lower() for word in ["trouser", "jean", "chino", "slack"]
        ):
            shoe_subset = scored[
                scored["Type"].str.contains(
                    "shoe|sneaker|boot|loafer|heel|slide|flip|sandal",
                    case=False, na=False
                )
            ]
            if not shoe_subset.empty:
                item_type = shoe_subset.iloc[0]["Type"]
            else:
                item_type = ""
    
        outfit[cat] = item_type
    
        #refresh top type for next iteration (needed for polo rule)
        if cat.lower() == "top":
            top_item = item_type
        
            

    # Fallback for sneakers
    if any(s in style_lower for s in ["casual", "party"]) and outfit.get("Shoes"):
        current_shoe = str(outfit["Shoes"]).lower()
        # Replace formal shoes with sneakers for casual looks
        if any(word in current_shoe for word in ["dress", "oxford", "heel", "loafer"]):
            sneaker_alts = scored[
                scored["Type"].str.contains("sneaker|trainer|athletic|casual shoe", case=False, na=False)
            ]
            if not sneaker_alts.empty:
                outfit["Shoes"] = sneaker_alts.sample(1).iloc[0]["Type"]
    
    #ensure valid shoes are always chosen
    if not outfit.get("Shoes") or not re.search(r"shoe|sneaker|boot|loafer|heel|slide|flip|sandal", str(outfit.get("Shoes", "")).lower()):
        if "sport" in style_lower:
            shoe_candidates = df[
                df["Type"].str.contains("sneaker|trainer|athletic", case=False, na=False)
                | df["Category"].str.lower().isin(["sneakers"])
            ]
        else:
            shoe_candidates = df[
                df["Type"].str.contains("shoe|sneaker|boot|loafer|heel|slide|flip|sandal", case=False, na=False)
                | df["Category"].str.lower().isin(["shoes", "sneakers"])
            ]
    
        # Remove non-footwear items to print out for shoes
        shoe_candidates = shoe_candidates[
            ~shoe_candidates["Type"].str.contains(
                "short|pant|trouser|jean|chino|slack|shirt|polo|hoodie",
                case=False, na=False
            )
        ]
        shoe_candidates = shoe_candidates[
        ~shoe_candidates["Type"].str.contains(
            "short|pant|trouser|jean|chino|slack|shirt|polo|hoodie|flip|slide|sandal",
            case=False, na=False
        )
    ]
    
        if not shoe_candidates.empty:
            outfit["Shoes"] = shoe_candidates.sample(1).iloc[0]["Type"]
        else:
            outfit["Shoes"] = ""

    return outfit
