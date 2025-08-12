"""Streamlit dashboard for visualizing HAR training runs.

The dashboard pulls metrics from Weights & Biases when a valid
``WANDB_API_KEY`` exists in the environment (optionally loaded from a
``.env`` file). If no API key is available, the dashboard falls back to a
local CSV file in ``data/``.

Users can filter results by model type and sensor configuration and view
overall accuracy, F1 score, and generalization gap comparisons along with
tables summarizing the best-performing models.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import wandb
from dotenv import load_dotenv

# Load values from a `.env` file if present
load_dotenv()

ENTITY = os.getenv("WANDB_ENTITY", "")
PROJECT = "sensor-fusion-har"

LOCAL_CSV_CANDIDATES = [
    Path("data/training_data_har.csv"),
    Path("data/training_results_har.csv"),
]


@st.cache_data
def load_run_data() -> pd.DataFrame:
    """Return run metrics from W&B or a local CSV fallback."""

    if os.getenv("WANDB_API_KEY"):
        try:
            api = wandb.Api()
            runs = api.runs(f"{ENTITY}/{PROJECT}") if ENTITY else api.runs(PROJECT)
            records = []
            for run in runs:
                records.append(
                    {
                        "model": run.config.get("model"),
                        "sensor_config": run.config.get("sensor_config"),
                        "train_accuracy": run.summary.get("train_accuracy"),
                        "test_accuracy": run.summary.get("test_accuracy"),
                        "train_f1": run.summary.get("train_f1"),
                        "test_f1": run.summary.get("test_f1"),
                        "generalization_gap": run.summary.get("generalization_gap"),
                        "runtime": run.summary.get("_runtime")
                        or run.summary.get("runtime"),
                    }
                )

            df = pd.DataFrame(records)
            if not df.empty:
                return df
        except Exception:
            pass

    for candidate in LOCAL_CSV_CANDIDATES:
        if candidate.exists():
            df = pd.read_csv(candidate)
            if "generalization_gap" not in df.columns:
                if {
                    "train_accuracy",
                    "test_accuracy",
                }.issubset(df.columns):
                    df["generalization_gap"] = (
                        df["train_accuracy"] - df["test_accuracy"]
                    )
            if "Runtime" in df.columns and "runtime" not in df.columns:
                df["runtime"] = df["Runtime"]
            return df

    return pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="Sensor Fusion HAR Dashboard", layout="wide")
    st.title("Sensor Fusion HAR Dashboard")
    st.markdown("### Authored By: Srihari Raman")
    st.write(
        "This dashboard offers an interactive overview of sensor fusion models trained on wearable"
        " sensor data for human activity recognition. Metrics are sourced"
        " from Weights & Biases runs or local CSV logs."
    )

    with st.spinner("Loading run data..."):
        df = load_run_data()
        if df.empty:
            st.info("No run data available.")
            return

    models = df["model"].dropna().unique().tolist()
    sensors = df["sensor_config"].dropna().unique().tolist()

    with st.sidebar:
        st.header("Filters")
        st.write("Select models and sensor setups to explore performance.")
        selected_models = st.multiselect("Model", models, default=models)
        selected_sensors = st.multiselect(
            "Sensor Config", sensors, default=sensors
        )
        group_by = st.selectbox(
            "Group metrics by", ["model", "sensor_config"], index=1
        )

    df_filtered = df[
        df["model"].isin(selected_models)
        & df["sensor_config"].isin(selected_sensors)
    ]

    if df_filtered.empty:
        st.warning("No data after applying filters.")
        return

    # ----- Top-level metrics -----
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Models", int(df_filtered["model"].nunique()))
    col2.metric("Total Runs", int(len(df_filtered)))
    if "runtime" in df_filtered.columns and not df_filtered["runtime"].dropna().empty:
        col3.metric(
            "Avg Runtime (s)", f"{df_filtered['runtime'].mean():.1f}"
        )
    else:
        col3.metric("Avg Runtime (s)", "N/A")

    group_label = group_by.replace("_", " ").title()

    # ----- Row 1: accuracy and F1 -----
    r1c1, r1c2 = st.columns(2)

    acc_summary = (
        df_filtered.groupby(group_by)["test_accuracy"].mean().reset_index()
    )
    fig_acc = px.bar(
        acc_summary,
        x=group_by,
        y="test_accuracy",
        color=group_by,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"Average Test Accuracy by {group_label}",
    )
    r1c1.plotly_chart(fig_acc, use_container_width=True)

    f1_summary = (
        df_filtered.groupby(group_by)["test_f1"].mean().reset_index()
    )
    fig_f1 = px.bar(
        f1_summary,
        x=group_by,
        y="test_f1",
        color=group_by,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"Average Test Weighted F1 by {group_label}",
    )
    r1c2.plotly_chart(fig_f1, use_container_width=True)

    # ----- Row 2: generalization gap and fused vs non-fused difference -----
    r2c1, r2c2 = st.columns(2)

    gap_summary = (
        df_filtered.groupby(group_by)["generalization_gap"]
        .mean()
        .abs()
        .reset_index()
    )
    fig_gap = px.bar(
        gap_summary,
        x=group_by,
        y="generalization_gap",
        color=group_by,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"Average Generalization Gap by {group_label}",
    )
    r2c1.plotly_chart(fig_gap, use_container_width=True)

    if (
        "fused" in df_filtered["sensor_config"].unique()
        and (df_filtered["sensor_config"] != "fused").any()
    ):
        fused_gap = (
            df_filtered[df_filtered["sensor_config"] == "fused"]
            .groupby("model")["generalization_gap"]
            .mean()
        )
        non_fused_gap = (
            df_filtered[df_filtered["sensor_config"] != "fused"]
            .groupby("model")["generalization_gap"]
            .mean()
        )
        gap_diff = (
            fused_gap - non_fused_gap
        ).dropna().reset_index(name="gap_difference")
        if not gap_diff.empty:
            fig_diff = px.bar(
                gap_diff,
                x="model",
                y="gap_difference",
                color="model",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Fused vs Non-Fused Gap Difference",
            )
            r2c2.plotly_chart(fig_diff, use_container_width=True)
    else:
        r2c2.empty()

    # ----- Best model setup by generalization gap -----
    best_setup = (
        df_filtered.loc[
            df_filtered.groupby("model")["generalization_gap"].idxmin(),
            [
                "model",
                "sensor_config",
                "test_accuracy",
                "test_f1",
                "generalization_gap",
            ],
        ]
        .sort_values("test_f1", ascending=False)
        .reset_index(drop=True)
    )
    best_setup = best_setup.rename(
        columns={
            "model": "Model",
            "sensor_config": "Sensor Config",
            "test_accuracy": "Test Accuracy",
            "test_f1": "Test F1",
            "generalization_gap": "Gen Gap",
        }
    ).round(3)
    st.subheader("Best Model Setup by F1 Score")
    st.dataframe(best_setup)


if __name__ == "__main__":
    main()