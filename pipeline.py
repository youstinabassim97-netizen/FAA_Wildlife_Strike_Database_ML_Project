"""
Shared inference pipeline for the FAA Wildlife Strike damage model.

Both app.py (Streamlit UI) and api.py (FastAPI service) import from this
module, so the exact same encoding/scaling logic is used everywhere —
no risk of the API and the UI silently drifting apart.
"""

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Load artifacts once
# ---------------------------------------------------------------------------
MODEL = joblib.load("demo_model.joblib")
MODEL_COLUMNS = joblib.load("demo_model_columns.joblib")
SCALER = joblib.load("model_data_A_v2_imputed_scaler.joblib")
TE_MAPS = joblib.load("model_data_A_v2_imputed_target_encoding_maps.joblib")

# ---------------------------------------------------------------------------
# The exact same encodings used in 01b/02/03_encoding_scaling_v2.py
# ---------------------------------------------------------------------------
SIZE_MAP = {"Missing": 0, "Small": 1, "Medium": 2, "Large": 3}
PHASE_RISK_MAP = {"Missing": 0, "Low": 1, "Medium": 2, "High": 3}
HEIGHT_CAT_MAP = {"Missing": 0, "Ground": 1, "Low": 2, "Medium": 3, "High": 4}
SPEED_CAT_MAP = {"Missing": 0, "Slow": 1, "Moderate": 2, "Fast": 3, "VeryFast": 4}

PHASE_RISK_GROUP_BY_PHASE = {
    "Approach": "Medium", "Arrival": "Low", "Climb": "High", "Departure": "Low",
    "Descent": "High", "En Route": "High", "Landing Roll": "Low", "Local": "Medium",
    "Parked": "Low", "Take-off Run": "Medium", "Taxi": "Low", "Unknown": "High",
}
SEASON_BY_MONTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
                   5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
                   9: "Fall", 10: "Fall", 11: "Fall"}

SCALE_COLS = ["HEIGHT", "SPEED", "DISTANCE", "INCIDENT_YEAR", "HOUR",
              "SEEN_STRUCK_RATIO", "STATE_TE", "SPECIES_TE"]

SEEN_STRUCK_RATIO_DEFAULT = 1.0  # training-set median, used when seen-count is "not reported"

STATE_OPTIONS = sorted(TE_MAPS["STATE"]["mapping"].index.tolist())
SPECIES_OPTIONS = sorted(TE_MAPS["SPECIES"]["mapping"].index.tolist())

DEFAULT_THRESHOLD = 0.225  # Person 3's tuned threshold (best F1 on validation)


def height_category(height: float) -> str:
    if height <= 0:
        return "Ground"
    if height <= 500:
        return "Low"
    if height <= 3000:
        return "Medium"
    return "High"


def speed_category(speed: float) -> str:
    if speed <= 100:
        return "Slow"
    if speed <= 160:
        return "Moderate"
    if speed <= 250:
        return "Fast"
    return "VeryFast"


def target_encode(col_name: str, value: str) -> float:
    m = TE_MAPS[col_name]
    return float(m["mapping"].get(value, m["global_mean"]))


def build_feature_vector(raw: dict) -> pd.DataFrame:
    """Takes a dict of raw/human-readable inputs and returns a single-row
    DataFrame with exactly MODEL_COLUMNS, in order, ready for MODEL.predict."""

    row = {c: 0 for c in MODEL_COLUMNS}

    row["AC_MASS"] = raw["AC_MASS"]
    row["NUM_ENGS"] = raw["NUM_ENGS"]
    row["HEIGHT"] = raw["HEIGHT"]
    row["SPEED"] = raw["SPEED"]
    row["DISTANCE"] = raw["DISTANCE"]
    row["INCIDENT_MONTH"] = raw["INCIDENT_MONTH"]
    row["INCIDENT_YEAR"] = raw["INCIDENT_YEAR"]
    row["NUM_SEEN_ORDINAL"] = raw["NUM_SEEN_ORDINAL"]
    row["NUM_STRUCK_ORDINAL"] = raw["NUM_STRUCK_ORDINAL"]
    row["HOUR"] = raw["HOUR"]

    row["MULTIPLE_BIRDS_STRUCK"] = int(raw["NUM_STRUCK_ORDINAL"] >= 2)
    row["SEEN_COUNT_REPORTED"] = int(raw["seen_reported"])
    if raw["seen_reported"] and raw["NUM_SEEN_ORDINAL"] > 0:
        row["SEEN_STRUCK_RATIO"] = raw["NUM_STRUCK_ORDINAL"] / raw["NUM_SEEN_ORDINAL"]
    else:
        row["SEEN_STRUCK_RATIO"] = SEEN_STRUCK_RATIO_DEFAULT
    row["IS_SINGLE_ENGINE"] = int(raw["NUM_ENGS"] == 1)
    row["SMALL_AIRCRAFT_MASS"] = int(raw["AC_MASS"] in (1, 2))
    row["IS_FOG"] = int(raw["IS_FOG"])
    row["IS_RAIN"] = int(raw["IS_RAIN"])
    row["IS_SNOW"] = int(raw["IS_SNOW"])

    for c in ["HEIGHT_WAS_MISSING", "SPEED_WAS_MISSING", "DISTANCE_WAS_MISSING",
              "HOUR_WAS_MISSING", "SEEN_STRUCK_RATIO_WAS_MISSING"]:
        row[c] = 0

    row["SIZE_ORDINAL"] = SIZE_MAP[raw["SIZE"]]
    phase_risk = PHASE_RISK_GROUP_BY_PHASE.get(raw["PHASE_OF_FLIGHT"], "Missing")
    row["PHASE_RISK_GROUP_ORDINAL"] = PHASE_RISK_MAP[phase_risk]
    row["HEIGHT_CATEGORY_ORDINAL"] = HEIGHT_CAT_MAP[height_category(raw["HEIGHT"])]
    row["SPEED_CATEGORY_ORDINAL"] = SPEED_CAT_MAP[speed_category(raw["SPEED"])]

    def set_onehot(prefix, value):
        col = f"{prefix}_{value}"
        if col in row:
            row[col] = 1
        else:
            missing_col = f"{prefix}_Missing"
            if missing_col in row:
                row[missing_col] = 1

    set_onehot("AC_CLASS", raw["AC_CLASS"])
    set_onehot("TYPE_ENG", raw["TYPE_ENG"])
    set_onehot("PHASE_OF_FLIGHT", raw["PHASE_OF_FLIGHT"])
    set_onehot("TIME_OF_DAY", raw["TIME_OF_DAY"])
    set_onehot("SKY", raw["SKY"])
    set_onehot("FAAREGION", raw["FAAREGION"])
    set_onehot("SEASON", SEASON_BY_MONTH[raw["INCIDENT_MONTH"]])

    row["STATE_TE"] = target_encode("STATE", raw["STATE"])
    row["SPECIES_TE"] = target_encode("SPECIES", raw["SPECIES"])

    df = pd.DataFrame([row], columns=MODEL_COLUMNS)
    df[SCALE_COLS] = SCALER.transform(df[SCALE_COLS])
    return df


def predict(raw: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    X = build_feature_vector(raw)
    proba = float(MODEL.predict_proba(X[MODEL_COLUMNS])[0, 1])
    return {
        "damage_probability": proba,
        "threshold_used": threshold,
        "predicted_damage": proba >= threshold,
    }
