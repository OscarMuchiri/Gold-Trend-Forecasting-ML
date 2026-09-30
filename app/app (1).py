import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Gold Trend Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# LOAD LOCKED MODEL BUNDLE
# ============================================================

@st.cache_resource
def load_model_bundle():
    bundle_path = Path(__file__).parent / "artifacts" / "gold_hybrid_model_bundle.joblib"
    return joblib.load(bundle_path)


bundle = load_model_bundle()


# ============================================================
# EXTRACT LOCKED MODEL COMPONENTS
# ============================================================

logistic_model = bundle["logistic_model"]
logistic_scaler = bundle["logistic_scaler"]
gradient_boosting_model = bundle["gradient_boosting_model"]

feature_columns = bundle["feature_columns"]
feature_labels = bundle.get("feature_labels", {})
feature_statistics = bundle.get("feature_statistics", None)
supporting_indicators = bundle.get("supporting_indicators", [])

logistic_training_mean = bundle["logistic_training_mean"]
logistic_training_std = bundle["logistic_training_std"]
gradient_training_mean = bundle["gradient_training_mean"]
gradient_training_std = bundle["gradient_training_std"]

logistic_weight = bundle["logistic_weight"]
gradient_boosting_weight = bundle["gradient_boosting_weight"]

hybrid_threshold = bundle["hybrid_threshold"]
forecast_horizon = bundle.get("forecast_horizon_trading_days", 20)
selected_model_name = bundle.get("selected_model_name", "Technical Hybrid 70/30")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_label(feature_name):
    """Return a clean display label for each feature."""
    if isinstance(feature_labels, dict) and feature_name in feature_labels:
        return feature_labels[feature_name]
    return feature_name


def get_feature_stat(feature_name, stat_name, default_value):
    """
    Safely retrieve feature statistics for dashboard defaults and ranges.
    Works even if the statistics DataFrame uses different column naming.
    """
    if feature_statistics is None:
        return default_value

    try:
        stats = feature_statistics.copy()

        # If feature name is an index
        if feature_name in stats.index:
            row = stats.loc[feature_name]
            for candidate in [stat_name, stat_name.lower(), stat_name.upper(), stat_name.capitalize()]:
                if candidate in row.index:
                    return float(row[candidate])

        # If feature name is in a column
        possible_feature_cols = ["feature", "Feature", "feature_name", "Feature name", "indicator", "Indicator"]
        for col in possible_feature_cols:
            if col in stats.columns:
                matched = stats[stats[col].astype(str) == str(feature_name)]
                if not matched.empty:
                    row = matched.iloc[0]
                    for candidate in [stat_name, stat_name.lower(), stat_name.upper(), stat_name.capitalize()]:
                        if candidate in row.index:
                            return float(row[candidate])
    except Exception:
        pass

    return default_value


def probability_to_log_odds(probability):
    """Convert probability to log-odds with clipping for numerical safety."""
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def get_up_probability(model, X):
    """
    Get probability for the positive class.
    The positive class is assumed to be class 1 where classes_ exists.
    """
    probabilities = model.predict_proba(X)

    if hasattr(model, "classes_") and 1 in list(model.classes_):
        class_index = list(model.classes_).index(1)
    else:
        class_index = 1

    return probabilities[:, class_index]


def generate_forecast(input_values):
    """Generate locked hybrid forecast from user-entered indicator values."""

    X = pd.DataFrame([input_values], columns=feature_columns)

    # Logistic Regression path
    X_scaled = logistic_scaler.transform(X)
    logistic_score = float(logistic_model.decision_function(X_scaled)[0])

    logistic_normalized_score = (
        (logistic_score - logistic_training_mean) / logistic_training_std
    )

    # Gradient Boosting path
    gradient_probability = float(get_up_probability(gradient_boosting_model, X)[0])
    gradient_log_odds = float(probability_to_log_odds(gradient_probability))

    gradient_normalized_score = (
        (gradient_log_odds - gradient_training_mean) / gradient_training_std
    )

    # Hybrid score
    hybrid_score = (
        logistic_weight * logistic_normalized_score
        + gradient_boosting_weight * gradient_normalized_score
    )

    # Dashboard display labels
    # Equality is assigned to DOWNTREND for the dashboard wording.
    # Exact equality is practically very rare.
    predicted_trend = "UPTREND" if hybrid_score > hybrid_threshold else "DOWNTREND"

    score_margin = hybrid_score - hybrid_threshold

    return {
        "predicted_trend": predicted_trend,
        "logistic_score": logistic_score,
        "logistic_normalized_score": logistic_normalized_score,
        "gradient_probability": gradient_probability,
        "gradient_log_odds": gradient_log_odds,
        "gradient_normalized_score": gradient_normalized_score,
        "hybrid_score": hybrid_score,
        "hybrid_threshold": hybrid_threshold,
        "score_margin": score_margin
    }


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.title("Gold Trend Forecasting Dashboard")
st.caption(
    "Research prototype for the locked 20-trading-day XAU/USD hybrid forecasting model."
)

with st.expander("Model summary", expanded=False):
    st.write(f"**Selected model:** {selected_model_name}")
    st.write(f"**Forecast horizon:** {forecast_horizon} trading days")
    st.write(f"**Hybrid weights:** {logistic_weight:.2f} Logistic Regression + {gradient_boosting_weight:.2f} Gradient Boosting")
    st.write(f"**Locked decision threshold:** {hybrid_threshold:.4f}")
    st.write("**Model input:** 12 technical indicator values")
    st.write("**Model output:** UPTREND or DOWNTREND")


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Enter technical indicator values")

st.caption(
    "Enter the latest values for the 12 technical indicators. "
    "The dashboard uses the locked feature order from the exported model bundle."
)

input_values = {}
range_warnings = []

cols = st.columns(3)

for i, feature in enumerate(feature_columns):
    with cols[i % 3]:
        min_value = get_feature_stat(feature, "min", None)
        max_value = get_feature_stat(feature, "max", None)
        mean_value = get_feature_stat(feature, "mean", 0.0)
        median_value = get_feature_stat(feature, "median", mean_value)

        default_value = median_value if median_value is not None else mean_value
        label = safe_label(feature)

        entered_value = st.number_input(
            label=f"{feature}",
            value=float(default_value),
            format="%.6f",
            help=str(label)
        )

        input_values[feature] = entered_value

        if min_value is not None and max_value is not None:
            if entered_value < min_value or entered_value > max_value:
                range_warnings.append(
                    f"{feature}: entered value {entered_value:.4f} is outside training range "
                    f"[{min_value:.4f}, {max_value:.4f}]"
                )


# ============================================================
# FORECAST SECTION
# ============================================================

st.divider()

if st.button("Generate forecast", type="primary"):
    result = generate_forecast(input_values)

    st.header("Forecast result")

    if result["predicted_trend"] == "UPTREND":
        st.success(f"Predicted trend after {forecast_horizon} trading days: UPTREND")
    else:
        st.warning(f"Predicted trend after {forecast_horizon} trading days: DOWNTREND")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted trend", result["predicted_trend"])

    with col2:
        st.metric("Hybrid prediction score", f"{result['hybrid_score']:.4f}")

    with col3:
        st.metric(
            "Locked decision threshold",
            f"{result['hybrid_threshold']:.4f}",
            delta=f"{result['score_margin']:+.4f} score margin"
        )

    st.caption(
        "The hybrid score is a standardized decision score, not a calibrated probability. "
        "A score above the locked threshold produces an Uptrend prediction. "
        "A score equal to or below the threshold produces a Downtrend prediction."
    )

    with st.expander("Component scores", expanded=False):
        st.write(f"**Logistic raw decision score:** {result['logistic_score']:.6f}")
        st.write(f"**Logistic normalized score:** {result['logistic_normalized_score']:.6f}")
        st.write(f"**Gradient Boosting Up probability:** {result['gradient_probability']:.6f}")
        st.write(f"**Gradient Boosting log-odds:** {result['gradient_log_odds']:.6f}")
        st.write(f"**Gradient Boosting normalized score:** {result['gradient_normalized_score']:.6f}")

    if supporting_indicators:
        st.subheader("Supporting indicators")
        supporting_data = []
        for indicator in supporting_indicators:
            if indicator in input_values:
                supporting_data.append(
                    {
                        "Indicator": indicator,
                        "Entered value": input_values[indicator]
                    }
                )

        if supporting_data:
            st.dataframe(pd.DataFrame(supporting_data), use_container_width=True)

    if range_warnings:
        st.subheader("Input range warnings")
        for warning in range_warnings:
            st.warning(warning)
    else:
        st.info("All entered values are within the available training ranges where ranges were provided.")

    st.caption(
        "Academic note: DOWNTREND is the dashboard display label for the model's non-upward class. "
        "In the original target construction, this class means the future close is equal to or below "
        "the current close after the forecast horizon."
    )

else:
    st.info("Enter the indicator values and click 'Generate forecast' to view the predicted trend.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "This dashboard is a proof-of-concept academic research prototype. "
    "It does not provide investment advice, execute trades, or guarantee future market outcomes."
)
