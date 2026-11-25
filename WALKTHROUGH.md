# Walkthrough

A plain-English guide to what this project does, why each piece matters, and how to read the results.

## The Problem

Banks use statistical models to predict which loans will default. These models are trained on historical data — but the world changes. New regulations appear, economic cycles shift, and the population of borrowers evolves. A model trained in the early 1990s has never seen post-2000 interest rate data, the 2007 financial crisis, or post-pandemic lending patterns.

**Model drift** is what happens when the data the model sees in production no longer looks like the data it was trained on. This project measures that drift, detects when it becomes dangerous, and compares strategies for keeping the model current.

## The Data

We use **SBA 7(a) loans** — small business loans partially guaranteed by the US Small Business Administration. The dataset covers FY1991–FY2021 with ~1.5 million labelled loans. A loan "defaults" if it is charged off within 5 years of approval.

This is real lending data, not a toy dataset. Default rates range from 2.6% (FY2020–2021) to 27.5% (FY2007, during the financial crisis).

## The Pipeline

### Stage 0–1: Setup, Labels, and Splits

We split 31 years of data into three non-overlapping windows:

- **Reference (FY1991–1995)**: the model is trained here
- **Monitoring (FY1996–2018)**: the model is scored here, one year at a time
- **Final evaluation (FY2019–2021)**: held back and scored exactly once at the end

The split is fixed before seeing any results. Final evaluation years are never used to choose anything — they exist only to report the honest, out-of-sample performance of whatever model each strategy ends up with.

### Stage 2: Features and Baseline Models

We engineer 17 features from the loan application data: loan amount, SBA guarantee percentage, interest rate, term, jobs supported, business type, industry sector, and others. Two models are trained:

- **Logistic Regression** (AUC 0.87): transparent, interpretable
- **Gradient Boosting Machine** (AUC 0.94): higher accuracy, used as the main model

Both are trained only on FY1991–1995 data. The GBM's AUC of 0.94 on FY1996 means it correctly ranks 94% of default/non-default pairs — a strong baseline.

### Stage 3: Drift Detection

Starting from FY1996, we "replay" each year through the drift detector. For each year, we ask:

1. **Score drift**: Has the distribution of model predictions shifted? (Measured by Population Stability Index, or PSI)
2. **Feature drift**: Have individual input features changed? (PSI per feature, plus KS and chi-square tests)
3. **Performance**: Has the model's accuracy degraded? (AUC compared to baseline)

Each year gets an alert level:

| Alert | Score PSI | Meaning |
|-------|-----------|---------|
| Green | < 0.10 | Minor drift, model is fine |
| Amber | 0.10–0.25 | Moderate drift, investigate |
| Red | > 0.25 | Significant drift, consider retraining |

**Result**: All 23 monitoring years are red. The biggest driver is `interest_rate_missing` — interest rate data was 100% missing for every loan up to FY2008, 11% missing in FY2009, and fully populated from FY2010 onward. This structural data change, combined with the financial crisis and shifting lending standards, means the model is always operating in a fundamentally different environment than it was trained on.

### Stage 4: Retraining Strategies

We compare three approaches:

**Never retrain**: Keep the original FY1991–1995 model forever. AUC degrades from 0.94 (FY1996) to 0.60 (FY2010). The model still works in some years because the features it learned (loan amount, guarantee percentage) remain somewhat predictive, but it misses the new patterns in interest rate data and crisis-era defaults.

**Scheduled retrain**: Every year, train a new "challenger" model on the most recent 3 years of data with known labels. Compare it to the current "champion" on a held-out year. Promote the challenger only if it's clearly better (AUC improvement > 0.005, confirmed by bootstrap confidence intervals).

**Drift-triggered retrain**: Same as scheduled, but only attempt retraining when the drift alert is red.

#### The Label Delay Problem

A critical real-world constraint: we can't label a loan as "defaulted" or "survived" until 5 years have passed. At year 2001, we know the outcomes for loans approved through FY1995 — but not FY1996–2001. This means:

- First retrain possible: FY2000 (when FY1991–1994 labels are known, giving 3 training years + 1 comparison year)
- The model must endure 4 years of drift (FY1996–1999) before retraining is even possible
- Each retrain uses data that is 6+ years old relative to the current monitoring year

#### Results

| Strategy | Retrains | Avg Monitoring AUC | Final Eval AUC |
|----------|----------|---------------------|----------------|
| Never | 0 | 0.751 | 0.744 |
| Scheduled | 13 | 0.825 | 0.925 |
| Drift-triggered | 13 | 0.825 | 0.925 |

Scheduled and drift-triggered are identical because all periods are red. The practical insight: for datasets with persistent structural drift, a simple yearly schedule works as well as a sophisticated drift trigger. Drift-triggered retraining would add value in datasets where drift is episodic — quiet years followed by sudden shifts.

**Key moment — the financial crisis**: Even with retraining, AUC dips to 0.63–0.71 during FY2007–2010. The rolling window trains on pre-crisis data (FY1997–1999), which doesn't prepare the model for 27% default rates. The model catches up after FY2011, when crisis-era data enters the training window.

**6 rejections**: In 6 of 19 retrain attempts, the challenger model was rejected — it didn't beat the champion by the required 0.005 AUC margin. This conservative threshold prevents unnecessary model churn.

### Stage 5: Dashboard

The Streamlit dashboard visualises all of this in five interactive tabs:

- **Overview**: KPI cards showing 23 periods, 23 red alerts, 1.19M loans scored. Alert timeline as a colour-coded bar chart. Default rate over time with financial crisis annotations.
- **Drift Analysis**: Score PSI over time with green/amber/red bands. Feature PSI heatmap showing which features drive drift and when. Baseline AUC degradation.
- **Model Performance**: Three-line AUC chart comparing strategies. Table of all retrain events with bootstrap CIs.
- **Strategy Comparison**: Summary table, final evaluation results, AUC lift bar chart.
- **About**: Methodology and alert thresholds.

## How to Read the Dashboard

**Overview tab**: Start here. The KPI cards give the headline numbers. The alert timeline is all red — that's the honest result, not a bug. The default rate chart shows the crisis spike (27.5% in FY2007) and post-crisis normalisation (3–5% from FY2010).

**Drift Analysis tab**: The PSI chart shows drift building slowly (FY1996–2005), then accelerating through the crisis and peaking at PSI 1.1 by FY2012. The heatmap reveals the structural cause — interest rate features dominate from FY2009.

**Model Performance tab**: This is the key chart. The gray "Never Retrain" line falls and stays low. The blue/green retrained lines recover to 0.95+ by FY2013, once the rolling window catches up. During the crisis (FY2007–2010), even retrained models struggle — a universal limitation when the training data doesn't yet include the current regime.

**Strategy Comparison tab**: The lift chart shows the value of retraining growing over time, peaking at +0.24 AUC by FY2015. The final evaluation confirms the advantage holds on unseen data.

## Caveats

1. **Retrospective analysis**: We compute AUC using labels that are known in hindsight. In production, AUC would only be computable after the 5-year outcome window. Drift metrics (PSI, KS, chi-square) are computable in real-time since they don't need labels.

2. **Single dataset**: Only SBA data is fully run. Freddie Mac requires separate data download. UCI is a benchmark only. Results on mortgage data may differ — different features, shorter outcome windows, different drift patterns.

3. **Fixed hyperparameters for retrain**: Retrained models reuse the baseline's hyperparameters rather than re-tuning. With a 3-year rolling window, there are only 2 CV folds — too few for meaningful hyperparameter search.

4. **All-red alerts**: The all-red finding is genuine but limits the comparison — drift-triggered adds no value over scheduled when every period is red. A dataset with episodic drift would better demonstrate the value of drift-triggered retraining.
