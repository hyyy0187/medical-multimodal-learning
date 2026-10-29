
# %% Part 1: Generate synthetic survival data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

# Support both `python tutorials/15_day8_survival_demo.py` and VS Code cells.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment import load_config

try:
    from IPython.display import display
except ImportError:
    display = print

cfg = load_config()
survival_cfg = cfg["survival"]
rng = np.random.default_rng(cfg["seed"])

N = survival_cfg["n_patients"]
HORIZON = survival_cfg["horizon_months"]
horizon_label = f"{HORIZON:g}"
horizon_key = horizon_label.replace(".", "_")
status_column = f"status_at_{horizon_key}_months"
risk_column = f"risk_{horizon_key}m"

# Synthetic baseline patient features
age = np.clip(rng.normal(60, 12, N), 18, 90)
severity = rng.binomial(1, 0.25, N)

# Artificial image and text signals
image_signal = rng.normal(size=N)
text_signal = rng.normal(size=N)

# Artificial log-risk function
log_risk = (
    0.025 * (age - 60)
    + 0.7 * severity
    + 0.4 * image_signal
    + 0.3 * text_signal
)

# Exponential event-time distribution
hazard_per_month = 0.035 * np.exp(log_risk)

event_time = rng.exponential(
    scale=1 / hazard_per_month
)

# Some patients stop being observed early
censor_time = rng.uniform(2, 24, N)

# Observed duration and event indicator
observed_time = np.minimum(
    event_time,
    censor_time
)

event = (
    event_time <= censor_time
).astype(int)

df = pd.DataFrame({
    "patient_id": [f"P{i:04d}" for i in range(N)],
    "age": age,
    "severity": severity,
    "image_signal": image_signal,
    "text_signal": text_signal,
    "time_months": observed_time,
    "event": event
})

print(df.head(10).round(3))

print("\nPatients:", len(df))
print("Observed events:", int(df["event"].sum()))
print("Censored:", int((df["event"] == 0).sum()))


# %% Part 2: Event status at a prediction horizon

conditions = [
    (df["event"] == 1) &
    (df["time_months"] <= HORIZON),

    df["time_months"] >= HORIZON
]

choices = [
    f"Event within {horizon_label} months",
    f"Event-free through {horizon_label} months"
]

df[status_column] = np.select(
    conditions,
    choices,
    default=f"Censored before {horizon_label} months"
)

print(
    df[status_column]
    .value_counts()
)

display(
    df[
        [
            "patient_id",
            "time_months",
            "event",
            status_column
        ]
    ].head(15)
)


# %% Part 3: Kaplan-Meier survival curve

kmf = KaplanMeierFitter()

kmf.fit(
    durations=df["time_months"],
    event_observed=df["event"],
    label="Synthetic cohort"
)

plt.figure(figsize=(7, 4.5))

kmf.plot_survival_function(
    show_censors=True
)

plt.xlabel("Follow-up time (months)")
plt.ylabel("Event-free survival probability")
plt.title("Kaplan-Meier Survival Curve")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(
    f"Estimated {horizon_label}-month event-free probability:",
    round(float(kmf.predict(HORIZON)), 4)
)
km_survival = float(kmf.predict(HORIZON))
km_risk = 1 - km_survival

# Naive observed event proportion
naive_risk = (
    (df["event"] == 1) &
    (df["time_months"] <= HORIZON)
).mean()

print(f"KM {horizon_label}-month survival: {km_survival:.4f}")
print(f"KM {horizon_label}-month event risk: {km_risk:.4f}")
print(f"Naive event proportion: {naive_risk:.4f}")

# %% Part 4: Fit Cox proportional hazards model

FEATURES = [
    "age",
    "severity",
    "image_signal",
    "text_signal"
]

# Random split is acceptable here for independent
# synthetic patients; real clinical data need a
# patient-level and preferably temporal split.
indices = rng.permutation(len(df))

n_train = int(len(df) * survival_cfg["train_fraction"])

train_df = df.iloc[
    indices[:n_train]
].reset_index(drop=True)

test_df = df.iloc[
    indices[n_train:]
].reset_index(drop=True)

train_cox = train_df[
    FEATURES + ["time_months", "event"]
].copy()

cox = CoxPHFitter(
    penalizer=survival_cfg["penalizer"]
)

cox.fit(
    train_cox,
    duration_col="time_months",
    event_col="event"
)

print("Cox model coefficients:")

display(
    cox.summary[
        ["coef", "exp(coef)", "p"]
    ].round(4)
)

# Higher partial hazard = higher predicted risk
risk_scores = cox.predict_partial_hazard(
    test_df[FEATURES]
).to_numpy().ravel()

test_cindex = concordance_index(
    test_df["time_months"],
    -risk_scores,
    test_df["event"]
)

print("Test C-index:", round(test_cindex, 4))


# %% Part 5: Predict absolute risk

survival_at_horizon = cox.predict_survival_function(
    test_df[FEATURES],
    times=[HORIZON]
)

predicted_survival = (
    survival_at_horizon.iloc[0]
    .to_numpy()
)

predicted_risk = 1 - predicted_survival

test_df[risk_column] = predicted_risk

print(f"Predicted {horizon_label}-month event risk:")

display(
    test_df[
        [
            "patient_id",
            "age",
            "severity",
            risk_column
        ]
    ].head(10).round(4)
)



# %% Part 6: Simple survival calibration plot

test_df["risk_group"] = pd.qcut(
    test_df[risk_column],
    q=survival_cfg["calibration_groups"],
    labels=False,
    duplicates="drop"
)

calibration_rows = []

for group_id, group_df in test_df.groupby(
    "risk_group"
):

    group_kmf = KaplanMeierFitter()

    group_kmf.fit(
        durations=group_df["time_months"],
        event_observed=group_df["event"]
    )

    observed_risk = (
        1 - float(group_kmf.predict(HORIZON))
    )

    mean_predicted_risk = (
        group_df[risk_column].mean()
    )

    calibration_rows.append({
        "group": int(group_id),
        "N": len(group_df),
        "predicted": mean_predicted_risk,
        "observed_KM": observed_risk
    })

calibration_df = pd.DataFrame(
    calibration_rows
)

display(calibration_df.round(4))

plt.figure(figsize=(5.5, 5))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.scatter(
    calibration_df["predicted"],
    calibration_df["observed_KM"],
    s=65,
    label="Synthetic test data"
)

for row in calibration_df.itertuples():
    plt.annotate(
        f"G{row.group + 1}",
        (row.predicted, row.observed_KM),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel(f"Mean predicted {horizon_label}-month risk")
plt.ylabel(f"KM-estimated observed {horizon_label}-month risk")
plt.title("Survival Model Calibration")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
