"""
FAA Wildlife Strike — Damage Risk Predictor (Streamlit demo)

Run with:  streamlit run app.py

This app ships with a Logistic Regression model trained on Model A's
prepared features (train_demo_model.py). To use the team's real tuned
model instead (XGBoost / LightGBM / LinearSVC from Person 3), just save
it as demo_model.joblib with the same 80-column input order as
model_data_A_v2_imputed_X_train.csv — nothing else needs to change.

All encoding/scaling logic lives in pipeline.py and is shared with api.py,
so the Streamlit UI and the FastAPI service can never silently drift apart.
"""

import streamlit as st
from pipeline import (
    PHASE_RISK_GROUP_BY_PHASE, STATE_OPTIONS, SPECIES_OPTIONS,
    DEFAULT_THRESHOLD, predict,
)

st.set_page_config(page_title="Wildlife Strike Damage Risk", page_icon="✈️", layout="centered")
st.title("✈️ Wildlife Strike — Aircraft Damage Risk")
st.caption(
    "Demo model: Logistic Regression on Model A features (class-weighted). "
    "Swap in the team's tuned model by replacing demo_model.joblib."
)

with st.form("strike_form"):
    st.subheader("Aircraft")
    c1, c2, c3, c4 = st.columns(4)
    ac_class = c1.selectbox("AC_CLASS", ["A", "B", "C", "J", "Y", "Missing"])
    ac_mass = c2.selectbox("AC_MASS (1=lightest .. 5=heaviest)", [1, 2, 3, 4, 5], index=3)
    type_eng = c3.selectbox("TYPE_ENG", ["A", "B", "C", "D", "E", "F", "Y", "Missing"], index=3)
    num_engs = c4.selectbox("NUM_ENGS", [1, 2, 3, 4], index=1)

    st.subheader("Flight")
    c1, c2, c3 = st.columns(3)
    phase = c1.selectbox("PHASE_OF_FLIGHT", list(PHASE_RISK_GROUP_BY_PHASE.keys()) + ["Missing"])
    height = c2.number_input("HEIGHT (ft AGL)", min_value=0, max_value=32000, value=500)
    speed = c3.number_input("SPEED (knots)", min_value=0, max_value=541, value=140)
    distance = st.number_input("DISTANCE from airport (nautical miles)", min_value=0.0, max_value=200.0, value=1.0)

    st.subheader("Time")
    c1, c2, c3, c4 = st.columns(4)
    month = c1.selectbox("Month", list(range(1, 13)), index=8)
    year = c2.number_input("Year", min_value=1990, max_value=2030, value=2026)
    hour = c3.slider("Hour", 0, 23, 12)
    time_of_day = c4.selectbox("TIME_OF_DAY", ["Day", "Night", "Dawn", "Dusk", "Missing"])

    st.subheader("Weather")
    c1, c2, c3, c4 = st.columns(4)
    sky = c1.selectbox("SKY", ["No Cloud", "Some Cloud", "Overcast", "Missing"])
    is_fog = c2.checkbox("Fog")
    is_rain = c3.checkbox("Rain")
    is_snow = c4.checkbox("Snow")

    st.subheader("Wildlife")
    c1, c2 = st.columns(2)
    species = c1.selectbox("SPECIES", SPECIES_OPTIONS,
                            index=SPECIES_OPTIONS.index("Unknown bird") if "Unknown bird" in SPECIES_OPTIONS else 0)
    size = c2.selectbox("SIZE", ["Small", "Medium", "Large", "Missing"])
    seen_reported = st.checkbox("Number of animals seen WAS reported", value=True)
    c1, c2 = st.columns(2)
    num_seen = c1.slider("NUM_SEEN (ordinal bucket: 0=not reported,1=1,2=2-10,3=11-100,4=100+)", 0, 4, 1)
    num_struck = c2.slider("NUM_STRUCK (ordinal bucket: 1=1,2=2-10,3=11-100,4=100+)", 1, 4, 1)

    st.subheader("Location")
    c1, c2 = st.columns(2)
    state = c1.selectbox("STATE", STATE_OPTIONS)
    faaregion = c2.selectbox("FAAREGION", ["AAL", "ACE", "AEA", "AGL", "ANE", "ANM",
                                            "ASO", "ASW", "AWP", "FGN", "Missing"])

    threshold = st.slider(
        "Decision threshold (Person 3 tuned 0.225 for best F1 — lower = more sensitive)",
        0.0, 1.0, DEFAULT_THRESHOLD, 0.005,
    )
    submitted = st.form_submit_button("Predict damage risk")

if submitted:
    raw = dict(
        AC_CLASS=ac_class, AC_MASS=ac_mass, TYPE_ENG=type_eng, NUM_ENGS=num_engs,
        PHASE_OF_FLIGHT=phase, HEIGHT=height, SPEED=speed, DISTANCE=distance,
        INCIDENT_MONTH=month, INCIDENT_YEAR=year, HOUR=hour, TIME_OF_DAY=time_of_day,
        SKY=sky, IS_FOG=is_fog, IS_RAIN=is_rain, IS_SNOW=is_snow,
        SPECIES=species, SIZE=size, seen_reported=seen_reported,
        NUM_SEEN_ORDINAL=num_seen, NUM_STRUCK_ORDINAL=num_struck,
        STATE=state, FAAREGION=faaregion,
    )
    result = predict(raw, threshold=threshold)
    proba = result["damage_probability"]

    st.divider()
    st.metric("Predicted damage probability", f"{proba:.1%}")
    if result["predicted_damage"]:
        st.error(f"⚠️ Damage predicted (probability {proba:.1%} ≥ threshold {threshold:.1%})")
    else:
        st.success(f"✅ No damage predicted (probability {proba:.1%} < threshold {threshold:.1%})")

    with st.expander("What drove this prediction?"):
        st.write(
            f"- **SIZE** = {size} (strongest single feature in the EDA — "
            f"Large birds showed a ~14x higher damage rate than Small)\n"
            f"- **AC_MASS** = {ac_mass} (lighter aircraft = higher damage rate)\n"
            f"- **PHASE_OF_FLIGHT** = {phase} (risk group: {PHASE_RISK_GROUP_BY_PHASE.get(phase, 'Missing')})\n"
            f"- **NUM_STRUCK** bucket = {num_struck} (more birds struck = higher damage rate)"
        )
