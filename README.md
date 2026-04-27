# ✈️ Flight Delay Prediction at Scale
### Multiclass departure delay classification using time-series-aware ML on 15M+ rows of US flight data

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.x-orange?logo=apache-spark)](https://spark.apache.org)
[![Databricks](https://img.shields.io/badge/Platform-Databricks-red?logo=databricks)](https://databricks.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Project Title

**Weathering the Delays: Scalable Flight Departure Delay Prediction with Time-Series Random Forest Modeling in PySpark**

*UC Berkeley MIDS · Machine Learning at Scale (DS261) · 6-person team*

---

## Problem

Flight delays cost the US airline industry an estimated **$33B annually** and generate some of the highest-friction moments in consumer travel. Despite the scale of the problem, passengers typically receive no predictive delay information — only reactive notifications after disruptions have already cascaded.

**The goal:** Build a production-grade ML classification system that predicts a flight's departure delay category at least **2 hours before scheduled departure**, enabling proactive passenger notifications and smarter travel planning.

The problem was deliberately framed as **multiclass classification** across 13 delay severity buckets (≤15 min → 180+ min) rather than regression, so model outputs map directly to actionable passenger alerts — e.g., "expect a 30–60 minute delay" — rather than abstract delay-minute estimates.

| Delay Group | Interval |
|:-----------:|:--------:|
| 0 | ≤ 15 min (on-time) |
| 1 | 15–30 min |
| 2 | 30–45 min |
| ... | ... |
| 12 | > 180 min |

> **Why it's hard:** Over 80% of flights fall in Group 0, creating severe class imbalance across 13 targets. Compounded with 15M rows, temporal dependencies, and strict leakage constraints, this is a non-trivial production ML problem.

---

## Dataset

**Source:** Pre-joined OTPW (On-Time Performance + Weather) dataset from two authoritative US government repositories:

- **Flight/ATP data** — [US DOT TranStats](https://www.transtats.bts.gov): on-time performance for all domestic US flights (carrier, origin, destination, scheduled/actual departure times, delay causes, distance)
- **Weather data** — [NOAA Local Climatological Data](https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data): temperature, wind speed/direction, precipitation, and visibility at weather stations co-located with origin airports
- **Stations data** — Weather station metadata and airport proximity mappings
- **Airport codes** — IATA/ICAO codes and geographic coordinates

| Dimension | Value |
|-----------|-------|
| Date range | 2015–2021 |
| Raw rows | ~15 million |
| Rows after cleaning | ~4.4 million (12-month slice) |
| Columns (pre-selection) | 130+ |
| Final features | 37 |
| Blind test set | Full year 2019 (never seen during training) |
| Training set | 2015–2018 |

**Data quality notes:**
- ~24% of rows removed post-deduplication and missing-value audit — imputation was deliberately avoided to preserve signal integrity
- Multiple columns with 100% or >20% missingness were identified and dropped
- Target variable (`DEP_DELAY_GROUP`) exhibited severe long-tail imbalance requiring explicit sampling strategy

---

## Approach

### End-to-End Modular PySpark Pipeline

The pipeline was designed for reproducibility, scalability, and leakage prevention across every stage.

```
Raw OTPW Parquet
      │
      ▼
  Data Cleaning & EDA
  (dedup, missingness audit, class distribution)
      │
      ▼
  Feature Engineering
  (time-based, event-based, graph-based)
      │
      ▼
  Feature Selection
  (RF importance: 130+ → 37 features)
      │
      ▼
  Time-Series Train/Val/Test Split
  (block-window, 2-hour pre-departure cutoff)
      │
      ├──► Logistic Regression (baseline)
      ├──► Random Forest (primary)
      └──► MLP Neural Network (deep learning)
              │
              ▼
          Optuna Hyperparameter Tuning
          (block-window CV, quarterly folds)
              │
              ▼
          Final Evaluation on 2019 Blind Test Set
```

### Feature Engineering

Three categories of features were engineered to capture delay risk signals unavailable in raw flight records:

**Time-based (rolling-window)**
- Number of outgoing flights at origin airport in the 26 hours prior to departure
- Number of severely delayed flights (>15 min) at origin airport in the same window
- Average delay minutes segmented by airport, airline, and airport × airline combined
- Time-of-day bucket: Morning / Afternoon / Evening / Night

> The lookback window anchors on `two_hours_prior_depart_UTC` to guarantee all features are available at inference time — no leakage.

**Event-based**
- Binary holiday indicator: whether the scheduled departure falls on a major US holiday

**Graph-based**
- PageRank score per airport (computed on 2015–2018 training data only), capturing airport hub connectivity and network-level congestion risk

### Leakage Prevention

Leakage is the most common silent failure in time-series ML. Three explicit controls were enforced:

1. **Strict 2-hour pre-departure feature cutoff** — all features derived from the window ending at `two_hours_prior_depart_UTC`; any field requiring information from after that moment was removed
2. **Removal of leaky raw columns** — wheels-out time, taxi-out duration, and any downstream delay-cause fields were explicitly excluded
3. **Time-based train/test splits** — no random k-fold; all splits respect chronological ordering to prevent future information leaking into training folds

### Time-Series Cross-Validation

Standard k-fold cross-validation is inappropriate for time-series data. This project implemented **block-window CV**:

- **Hyperparameter tuning:** Quarterly blocks over the 12-month dataset (train on Q1 → test on Q2, shift forward)
- **Final training:** Yearly blocks over 2015–2018
- **Blind evaluation:** Full year 2019, held out entirely during all development

Undersampling was applied **within each fold** (not globally) to handle class imbalance without leaking future distribution information into earlier training windows.

### Modeling

| Model | Role | Key Design Choices |
|-------|------|-------------------|
| Logistic Regression | Baseline | Interpretable; exposed feature importance via coefficient magnitude |
| Random Forest | Primary | Bagging ensemble; no scaling/OHE required; captures non-linear feature interactions; parallelized across Spark workers |
| MLP Neural Network | Deep learning | Multiple architectures tested (1 hidden layer vs. 2 hidden layers); PySpark MLlib `MultilayerPerceptronClassifier` |

**Random Forest preprocessing:** StringIndexer for categorical features → VectorAssembler (18 categorical + 19 numerical features). No one-hot encoding or numerical scaling required.

### Hyperparameter Tuning

Grid search was infeasible at this scale (timed out after 8+ hours). **Optuna** (Bayesian optimization) completed 7 trials in ~2 hours, tuning:
- `maxDepth`
- `numTrees`
- Feature subsampling ratio

Best parameters from 12-month CV were used to train the final model on the full 2015–2018 dataset.

### Performance Engineering

Distributed Spark pipelines fail silently in two ways: lazy evaluation recomputes the entire DAG on every action, and null propagation from transformation overwrites is invisible until model training. Solutions applied:

- **Strategic checkpointing** at key pipeline stages (post-feature-engineering, post-join, post-sampling) to break Spark's computation graph and prevent redundant recomputation
- **In-memory caching** (`persist`) for DataFrames accessed multiple times (training folds, feature vectors)
- **Delta Lake persistence** for intermediate outputs, enabling cross-session reuse without re-running upstream stages
- **Repartitioning** before model training to balance executor load
- **Result:** ~40% reduction in pipeline runtime (~2 hours saved per full run)

---

## Results

| Metric | Baseline LR (raw features) | Final LR (engineered features) | Final Random Forest |
|--------|:--------------------------:|:------------------------------:|:-------------------:|
| Test F1 (weighted) | 0.0002 | 0.73 | ~0.82 |
| Weighted Recall | — | 0.70 | — |
| vs. Baseline | — | +365× | +12% vs. final LR |

**Top 5 features by Random Forest importance:**

| Rank | Feature | Type |
|:----:|---------|------|
| 1 | `avg_delay_byairport_airline` | Engineered (rolling window) |
| 2 | `DEP_TIME_HH` | Raw (hour of departure) |
| 3 | `avg_delay_by_airline` | Engineered (rolling window) |
| 4 | `Time_of_Day` | Engineered (time bucket) |
| 5 | `avg_delay_by_airport` | Engineered (rolling window) |

> 4 of the top 5 features are engineered — confirming that time-based congestion signals are the primary driver of delay predictability, not raw flight metadata.

**Tuning efficiency:**

| Method | Time | Trials |
|--------|:----:|:------:|
| Grid search | >8 hrs (timed out) | — |
| Optuna (Bayesian) | ~2 hrs | 7 |

---

## Key Insights

**1. Engineered features are the model.**
The baseline logistic regression with only raw temporal and distance features achieved F1 = 0.0002 — statistically indistinguishable from random. Adding rolling-window airport/airline congestion features pushed performance to F1 = 0.73+, a 365× improvement with no change in model architecture. Feature engineering, not model selection, was the dominant performance lever.

**2. Time-based CV is not optional.**
Standard k-fold cross-validation on time-series data inflates performance estimates by allowing the model to "learn from the future." Block-window CV with strict temporal ordering produced materially more conservative — and realistic — performance estimates. This matters for any production deployment where the model will be evaluated on data it genuinely hasn't seen.

**3. Distributed ML performance is an engineering problem.**
At 15M rows with iterative modeling, pipeline runtime was a binding constraint throughout the project. Spark's lazy evaluation model caused silent recomputation of the full DAG on every action. Strategic checkpointing and Delta Lake persistence reduced per-run time by ~40%. In large-scale ML, engineering discipline around data materialization often matters more than algorithm selection.

**4. Sampling strategy shapes correctness and data leakage, not just performance.**
Class weighting was attempted first but made training infeasible at scale. Undersampling was adopted instead — but it had to be applied per CV fold, not globally, to avoid distributional leakage between folds. Getting the sampling logic correct is a prerequisite for valid evaluation, not just a tuning decision.

**5. Metric choice must reflect the end user.**
Accuracy is misleading when 80% of examples share one class. Weighted F1 was selected as the primary metric. The team also explicitly evaluated the F1 vs. F2 trade-off: for passenger notification use cases, recall (catching real delays) is more valuable than precision (avoiding false alarms), which would favor F2. For airline operational scheduling, the trade-off inverts. A single model, two different evaluation criteria — the metric must follow the use case.

**6. Bayesian search is the practical default at scale.**
Optuna's probabilistic hyperparameter search covered meaningful trial space in 2 hours. Grid search failed entirely at this data volume. For production ML systems on large datasets, exhaustive search is not a viable strategy — Bayesian optimization should be the starting point.

---

## Business Impact

This system was designed for **passenger-centered deployment** with secondary utility for airline and airport operations:

| Use Case | Impact |
|----------|--------|
| Passenger alerts | 2–4 hour advance delay category notifications, enabling proactive re-booking, connection management, and travel adjustments |
| Travel app integration | Pipeline outputs are structured delay buckets — directly consumable by airline apps, third-party travel platforms, and push notification systems |
| Operational planning | With metric reconfiguration (precision-weighted), the same model supports airline scheduling, gate assignment, and crew coordination |

**Precision vs. recall trade-off, operationalized:**

| Target user | Metric preference | Reason |
|-------------|:-----------------:|--------|
| Passenger notifications | Higher recall (F2) | A missed delay hurts more than a false alarm |
| Airline operations | Higher precision (F1) | False scheduling adjustments carry real costs |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Compute & Platform** | Databricks (Azure), Apache Spark 3.x |
| **ML Framework** | PySpark MLlib, `MultilayerPerceptronClassifier` |
| **Languages** | Python 3.10, Spark SQL |
| **Feature Engineering** | PySpark SQL window functions, GraphX (PageRank), custom rolling-window UDFs |
| **Preprocessing** | `StringIndexer`, `VectorAssembler`, `OneHotEncoder` |
| **Modeling** | Logistic Regression, Random Forest (`RandomForestClassifier`), MLP Neural Network |
| **Hyperparameter Tuning** | Optuna (Bayesian optimization) |
| **Storage & Checkpointing** | Delta Lake, DBFS, Azure Blob Storage |
| **Data Sources** | US DOT TranStats, NOAA LCD, Datahub Airport Codes |
| **Visualization** | Plotly Express, Matplotlib, Seaborn |
| **Collaboration** | Databricks Notebooks, GitHub |

---

*Built as part of UC Berkeley's Master of Information and Data Science (MIDS) program, Machine Learning at Scale (DS261).*
