# Modelling Decisions

Every design choice and the reason behind it.

## Stage 0: Setup and Data

### Colour palette
**Decision:** Reuse the colour palette from the sibling `machine-failure-prediction` project (slate-900 primary `#1e293b`, soft white background `#ffffff`, slate-50 card background `#f8fafc`, slate-900 text `#0f172a`).
**Reason:** The spec asks for portfolio consistency. The sibling project is present at `../machine-failure-prediction`.

### Freddie Mac missing-value codes
**Decision:** Credit score 9999 → missing; DTI 999 → missing; Number of borrowers 99 → missing. All blanks in pipe-delimited files → NaN. MI percentage 0 means no mortgage insurance (kept as 0, not missing).
**Reason:** These are the documented sentinel values in the Freddie Mac file layout. Confirmed from the official file layout XLSX (July 2026 release).

### Freddie Mac zero balance codes for default
**Decision:** Codes 02 (Third Party Sale), 03 (Short Sale / Charge-Off), 09 (REO Disposition), 15 (Note Sale) are credit events (defaults). Code 01 (Prepaid or Matured) is voluntary. Code 06/96 (Repurchase / Defect Settlement) is excluded from default — it reflects a buyback for underwriting defects, not a borrower credit event.
**Reason:** Freddie Mac user guide and release notes. Code 06 was renamed to 96 in later releases; both are excluded from the default definition as they are seller/servicer events.

### SBA status normalisation
**Decision:** Normalise all status spellings to: PIF, CHGOFF, CANCLD, COMMIT, EXEMPT. Map "PAID IN FULL" → PIF, "CHARGED OFF"/"CHARGE OFF" → CHGOFF, "CANCELLED"/"CANCELED" → CANCLD, "COMMITTED"/"UNDISBURSED"/"NOT FUNDED" → COMMIT.
**Reason:** The SBA data dictionary shows these canonical codes but historical files have inconsistent spellings.

### SBA exclusions
**Decision:** Exclude CANCLD (cancelled) and COMMIT (undisbursed/committed) loans from modelling.
**Reason:** Cancelled loans were never funded. Committed/undisbursed loans have no disbursement and cannot default. Including them would bias the default rate downward.

### UCI fair-lending practice
**Decision:** Exclude SEX and MARRIAGE from model inputs. Keep AGE as a monitored feature only (tracked for drift but not used as a model input).
**Reason:** Fair-lending practice: protected characteristics should not be used for credit decisions. Age is monitored to detect population shifts but excluded from the model to avoid age discrimination.

### Line length
**Decision:** 99 characters, applied project-wide via ruff.
**Reason:** Fits comfortably on a 13-inch screen with a sidebar and line numbers visible.

### Chunk size for performance files
**Decision:** 500,000 rows per chunk when reading Freddie Mac performance files.
**Reason:** Performance files can be hundreds of MB. 500K rows per chunk keeps memory under ~200 MB per chunk while minimising I/O overhead.

## Stage 1: Labels, Periods and Splits

### SBA default definition
**Decision:** Default = charged off (CHGOFF) within 5 years of approval date. Non-default = paid in full (PIF), or EXEMPT with 5+ years elapsed since approval. Charged off after 5 years = non-default for the window period.
**Reason:** The 5-year outcome window is standard for SBA portfolio analysis. Late charge-offs beyond the window are treated as non-defaults to maintain a clean time-ordered evaluation.

### SBA EXEMPT handling
**Decision:** EXEMPT loans where 5+ years have passed since approval are treated as non-defaults ("survived"). EXEMPT loans with less than 5 years are censored (excluded from modelling for that window).
**Reason:** EXEMPT status means active or undisclosed. If enough time has passed without a charge-off, the loan has effectively survived. Recent EXEMPT loans are censored because their outcome is unknown.

### SBA as-of date
**Decision:** 2026-09-30, derived from the FOIA data filenames (asof_260930).
**Reason:** This is the data extract date per the SBA FOIA file naming convention.

### Freddie Mac default definition
**Decision:** Default = within 24 months of first payment date, reaches 90+ days delinquent OR exits with zero balance code 02/03/09/15. Prepaid or left before 24 months without default = non-default.
**Reason:** Industry standard for mortgage default studies. The 24-month window captures the peak default-hazard period.

### Freddie Mac pandemic forbearance
**Decision:** Count but do not alter the default definition. Report the count of loans with disaster-related delinquency or borrower assistance flags.
**Reason:** Per spec: "Keep the main definition unchanged, but report these counts."

### SBA split (fixed)
**Decision:**
- **Reference (5 years):** FY1991–FY1995
- **Monitoring (23 years):** FY1996–FY2018
- **Final evaluation (3 years):** FY2019–FY2021

31 usable years total (FY1991–FY2021). FY2022–FY2026 are not usable because their 5-year outcome window has not elapsed by 2026-09-30.

**Reason:** Follows the spec rule: first 5 usable years for reference, last 3 for final evaluation, middle for monitoring. This gives a long monitoring window that spans the 2007-2009 financial crisis and 2020 pandemic — ideal for drift detection.

### Freddie Mac split (deferred)
**Decision:** Split years will be computed when data is downloaded, using the same rule (first 5 usable, last 3 final eval, middle monitoring).
**Reason:** Freddie Mac data not yet available. The split function is written and tested.

### UCI split
**Decision:** 40% reference set + 20 simulated periods for synthetic drift benchmark. No time-ordered split (all data is from one 6-month window).
**Reason:** Per spec: UCI is used only for the detector benchmark, not the main replay.

### SBA label counts
- Total: 1,976,761
- Excluded: 267,150 (CANCLD: 245,627 + COMMIT: 21,523)
- Resolved (labelled): 1,498,288
- Defaults: 136,583 (9.12% of resolved)
- Non-defaults: 1,361,705
- Censored: 211,323 (recent EXEMPT loans)

## Stage 2: Features and Baseline Models

### SBA feature set
**Decision:** 8 numeric features (GrossApproval, SBAGuaranteedApproval, TermInMonths, InitialInterestRate, JobsSupported, sba_guaranteed_pct, interest_rate_missing, sold_sec_mrkt_missing) + 9 categorical features (BusinessType, ProjectState, RevolverStatus, BusinessAge, ProcessingMethod, CollateralInd, SoldSecMrktInd, FixedorVariableInterestInd, naics_sector). 17 total.
**Reason:** All known at approval time. Derived features add information: SBA guarantee share and 2-digit NAICS sector. Missing indicators for InitialInterestRate (100% missing in early years) and SoldSecMrktInd (75% missing overall).

### NaN handling
**Decision:** Fill NaN in numeric features with 0 after creating missing indicators. LR pipeline uses median imputation + StandardScaler; GBM handles NaN natively via HistGradientBoosting but all-NaN columns (like InitialInterestRate in FY1991-1999) need the 0 fill to prevent binning failure.
**Reason:** InitialInterestRate is 100% missing in SBA files before FY2009. The missing indicator (interest_rate_missing) carries all the information; the 0 fill is a no-information default.

### Rare category grouping
**Decision:** Categories below 0.5% of training rows grouped to "Other". Fitted on training window only. Unseen categories at scoring time → "Other".
**Reason:** Per spec. Prevents one-hot explosion and handles population shift in rare categories.

### Tuning grids (fast)
**Decision:** Used fast tuning grids for initial baseline (LR: C in {0.01, 1.0}; GBM: 4 combos). Full grids available for final models.
**Reason:** 16GB RAM constraint. Fast grids keep the initial training under 30s while still exploring the key hyperparameter dimensions.

### SBA baseline results
**Decision:** GBM selected as the main model. LR retained as the transparent baseline.
- LR: CV AUC 0.8652, out-of-sample AUC 0.8784 (FY1996)
- GBM: CV AUC 0.9501, out-of-sample AUC 0.9440 (FY1996)
- Calibration within 1pp for both models on FY1996

**Reason:** GBM outperforms LR by ~7pp AUC. Both well-calibrated (LR: pred 7.91% vs actual 8.13%; GBM: pred 7.54% vs actual 8.13%). No recalibration needed.

### Cost-sensitive thresholds (SBA, FY1996, 5:1 cost ratio)
**Decision:** LR threshold 0.14, GBM threshold 0.15. Also reporting 3:1 (LR=0.18, GBM=0.24) and 10:1 (LR=0.09, GBM=0.11).
**Reason:** Assumes a missed default costs 5x as much as wrongly declining a good loan. These are clearly labelled as assumptions per spec.

## Stage 3: Drift Detection

### Drift metrics
**Decision:** Three drift metrics per feature: PSI (Population Stability Index), KS statistic (numeric only), and chi-square test (categorical only). Score drift measured with equal-width PSI on [0,1].
**Reason:** PSI is the standard drift metric in credit risk. KS and chi-square provide complementary statistical tests. PSI uses quantile-based bins for numerics (adapts to skewed distributions) and value-based bins for discrete features (detects binary indicator shifts).

### PSI for discrete features
**Decision:** When a numeric feature has ≤10 unique values in the reference (binary indicators, small integers), use value-based binning instead of quantile-based binning.
**Reason:** Quantile bins collapse on constant features (e.g. `interest_rate_missing` = 1 for all reference loans). Value-based binning correctly detects the shift when the feature later takes new values.

### Alert classification
**Decision:** Three-level alerts (green/amber/red) driven by the worst of:
1. Score PSI: green < 0.10, amber 0.10–0.25, red > 0.25
2. Feature count: if ≥3 features are red (PSI > 0.25) → overall red; if ≥3 amber → overall amber
3. AUC drop: if AUC drops ≥ 0.02 from reference → amber

**Reason:** Matches the thresholds in config. Score PSI catches model-level drift; feature count catches broad population shifts; AUC drop catches performance degradation that drift metrics might miss.

### Index reset in train_baseline
**Decision:** Reset DataFrame indices after filtering to reference years, so that time-ordered CV fold indices (which use `.iloc`) work correctly on non-contiguous indices.
**Reason:** When the full labelled DataFrame is filtered to reference years only, the original index is non-contiguous. The CV fold function returns positional indices via `.index.to_numpy()` that must match `.iloc` semantics.

### SBA drift replay results
**Decision:** All 23 monitoring years (FY1996–FY2018) flagged as red alerts. This is correct — the GBM baseline trained on FY1991–1995 becomes increasingly outdated.
**Key findings:**
- **Dominant drift driver:** `interest_rate_missing` (PSI up to 18.4). Interest rate was 100% missing for every loan through FY2008, 11% missing in FY2009, and fully populated from FY2010 onward. `FixedorVariableInterestInd` (PSI 18.0) and `InitialInterestRate` (PSI 15.1) show the same pattern.
- **Score PSI timeline:** Starts at 0.003 (FY1996, green), rises to 0.12 (FY2005, amber), peaks at 1.1 (FY2012, red), then gradually declines.
- **Crisis spike:** Default rates jump from 6–8% (FY1996–2004) to 22–28% (FY2006–2008), then drop to 3–5% (FY2010–2018). The 2007–2008 financial crisis is clearly visible.
- **Model degradation:** GBM AUC falls from 0.9440 (FY1996) to 0.60–0.73 for FY1999–2018. The model loses about 20pp AUC over the monitoring window.
- **Post-crisis regime:** After 2010, default rates are ~3–5% (lower than reference 6.4%), but the model still performs poorly (AUC 0.60–0.73) because the population has fundamentally changed.

**Reason:** This is exactly the motivation for drift-triggered retraining. A model trained on FY1991–1995 cannot serve a population that changed structurally (new data fields, different lending standards, macroeconomic shifts). Stage 4 will show how retraining strategies recover performance.

### Retrospective labels in replay
**Decision:** Computed AUC for all monitoring years using retrospective labels (all outcomes known by 2026-09-30). In a production system, AUC would only be computable after the 5-year outcome window for each year.
**Reason:** This is a retrospective analysis. The drift metrics (PSI, KS, chi-square) are computable in real-time since they don't need labels. AUC is a bonus metric that validates whether drift corresponds to performance degradation.

## Stage 4: Retraining Strategies

### Champion/challenger comparison
**Decision:** At each retraining opportunity, train a challenger GBM on a 3-year rolling window (ROLLING_WINDOW_YEARS=3), hold out the most recent known year for comparison, and bootstrap the AUC difference (N_BOOTSTRAP_FAST=200). Promote the challenger only if its AUC exceeds the champion's by PROMOTION_AUC_THRESHOLD (0.005).
**Reason:** Prevents model churn from noisy AUC comparisons. The bootstrap CI gives confidence that the improvement is real, not just sampling noise.

### Retraining uses fixed hyperparameters
**Decision:** Retrained GBMs reuse the best hyperparameters from the baseline grid search rather than re-tuning. Only the data changes.
**Reason:** With a 3-year rolling window (often 100K–200K rows), there are only 2 expanding-window CV folds — too few for meaningful hyperparameter search. Reusing baseline params avoids overfitting on a thin CV and keeps runtime manageable (7 minutes total vs estimated 30+ with per-retrain tuning).

### Label delay constrains first retrain
**Decision:** First retrain possible at FY2000. With SBA's 5-year outcome window, at replay year Y the most recent known labels are from FY ≤ Y−6. Retraining requires ROLLING_WINDOW_YEARS + 1 = 4 known years (3 for training, 1 for comparison). At Y=2000, known years are FY1991–1994 — exactly 4.
**Reason:** No-look-ahead rule. At FY1996–1999, too few known label years exist for a meaningful retrain with a held-out comparison year.

### Scheduled vs drift-triggered identical for SBA
**Decision:** Both strategies produce identical results: 13 promotions out of 19 attempts, mean monitoring AUC 0.8246, final evaluation AUC 0.9251. This is because all 23 monitoring periods have red drift alerts — drift-triggered retrains at every opportunity, same as scheduled.
**Key findings:**
- **Never retrain:** Mean monitoring AUC 0.7513, final AUC 0.7441. The FY1991–1995 model degrades from 0.9440 (FY1996) to 0.60 (FY2010) but partially recovers to 0.76 by FY2018 as population stability returns.
- **With retraining:** Mean monitoring AUC improves by +7.3pp. Early years (FY1996–2001) remain on the baseline due to label delay. From FY2002 onward, the rolling window catches up and AUC recovers to 0.90+. Final evaluation years (FY2019–2021) achieve 0.92 AUC on average.
- **Crisis resilience:** Even with retraining, FY2006–2010 AUCs are 0.63–0.80. The rolling window trains on pre-crisis data (FY1997–1999) during the crisis, limiting recovery. The model catches up after FY2011 when crisis-era data enters the training window.
- **6 rejections:** Champions were kept at FY2000, FY2001, FY2007, FY2008, FY2015, FY2017 — cases where the challenger didn't beat the champion by the promotion threshold.

**Reason:** Honest results. With pervasive drift (structural data change when interest rates began to be recorded in FY2009), the drift trigger fires every year. The practical insight: for datasets with persistent structural drift, scheduled retraining is sufficient — the drift trigger adds no value over a fixed schedule. Drift-triggered retraining would add value in datasets where drift is episodic.

### Final evaluation scored once
**Decision:** FY2019–2021 scored exactly once with each strategy's final champion model (gbm_FY2009-2011 for the retrained strategies, the original baseline for "never"). Results never used to choose anything.
**Reason:** Per spec: "Final evaluation years scored once, never used to choose anything."
