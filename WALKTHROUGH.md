# Walkthrough

Everything this project does, in order, in plain words first and then in technical detail. The last part, [Questions an interviewer will ask](#13-questions-an-interviewer-will-ask), works through the questions this project tends to attract and answers each one with the numbers from this repository.

If you want the short version, read the [README](README.md). If you want the reasoning behind a specific choice, read [docs/decisions.md](docs/decisions.md). This file sits in between and is meant to be read start to finish.

## Contents

[1. What this project is](#1-what-this-project-is) · [2. The problem](#2-the-problem) · [3. The three datasets](#3-the-three-datasets) · [4. From raw records to a label](#4-from-raw-records-to-a-label) · [5. Splitting time](#5-splitting-time) · [6. Features](#6-features) · [7. The baseline models](#7-the-baseline-models) · [8. Measuring drift](#8-measuring-drift) · [9. The replay](#9-the-replay) · [10. Retraining](#10-retraining) · [11. Results](#11-results) · [12. Reading the dashboard](#12-reading-the-dashboard) · [13. Questions an interviewer will ask](#13-questions-an-interviewer-will-ask) · [14. Limitations](#14-limitations-and-what-i-would-do-next) · [15. Running it](#15-running-it-yourself) · [16. Where things live](#16-where-things-live-in-the-code)

## 1. What this project is

A credit risk model is trained once and then used for years. This project asks a question that most portfolio projects skip: what actually happens to that model over time, and what is each possible response worth?

To answer it, a probability of default model is trained on the earliest years of real lending data, then replayed forward one year at a time through decades of later loans. At each step the system measures how far the new data has moved from the training data, raises an alert, and decides whether to retrain. Three policies are compared on the same timeline: never retrain, retrain every year, and retrain only when an alert turns red.

The output is a Streamlit dashboard and a set of JSON result files. Nothing is simulated except one deliberate benchmark, described in section 4.

## 2. The problem

A lender approves loans using a model trained on the past. Then the world moves. Interest rates change, a recession arrives, lending standards tighten, the paperwork itself gets redesigned. The model does not know any of this. It keeps returning scores that look just as confident as they did on day one.

The gap between the data a model was trained on and the data it now sees is called **drift**. Drift is dangerous precisely because it is quiet. The model does not crash or throw an error. It simply gets worse at the one thing it was bought to do, and the first hard evidence usually arrives as losses, long after the drift began.

Two kinds matter here:

- **Data drift**, where the inputs change. Borrowers get older, loan sizes grow, a field that was always blank starts being filled in.
- **Concept drift**, where the relationship between inputs and outcome changes. A credit score of 700 meant something different in 2006 than it did in 2010.

PSI and the other tests in this project detect data drift directly. Concept drift shows up indirectly, as the model's accuracy falling while the inputs look stable.

## 3. The three datasets

| Dataset | What it is | Period | Size | Role here |
|---|---|---|---|---|
| SBA 7(a) FOIA | US small business loans partly guaranteed by the Small Business Administration | FY1991 to FY2021 | 1,976,761 records, 1,498,288 usable | Primary. The full drift replay runs on this. |
| Freddie Mac Single Family Loan Level | US residential mortgages, the public sample files | 1999 to 2023 | 1.25M sample loans | Second real dataset. Confirms the findings are not an artefact of one book. |
| UCI Default of Credit Card Clients | 30,000 Taiwanese credit card customers, 2005 | one 6 month window | 30,000 | Controlled benchmark. Drift is injected deliberately so the detectors can be tested against a known answer. |

Using two real datasets rather than one is deliberate. The SBA result turns out to be extreme, with drift in every single period. Without a second dataset there would be no way to tell whether that says something about drift in general or something about SBA specifically. It turns out to be the latter, and that comparison is the most interesting finding in the project.

Raw data is never committed. Freddie Mac's terms allow internal use only and forbid redistribution, so the repository holds code and aggregated JSON results and nothing else.

## 4. From raw records to a label

A model needs a yes or no outcome for every loan. Getting that right is most of the work, and it is where quiet mistakes do the most damage.

### SBA

A loan counts as a **default** if it was charged off within five years of approval. It counts as a **non default** if it was paid in full, or if it is marked exempt and five or more years have passed without a charge off.

Three groups are removed rather than labelled:

- **Cancelled** loans, 245,627 of them. They were never funded, so they could not default. Leaving them in would drag the measured default rate down.
- **Committed or undisbursed** loans, 21,523. Same reasoning.
- **Censored** loans, 211,323. These are recent exempt loans where the five year window has not finished, so the outcome is genuinely unknown. Treating an unknown as a non default would be the single easiest way to make this project quietly wrong.

That leaves **1,498,288 labelled loans, of which 136,583 defaulted, a rate of 9.12%**.

A charge off that happens after the five year window is treated as a non default for that window. This is a choice, not a law: the window has to end somewhere, and moving it would change the numbers.

### Freddie Mac

A loan counts as a default if, within 24 months of the first payment date, it either reaches 90 or more days delinquent or exits with a zero balance code of 02, 03, 09 or 15. Those four codes are third party sale, short sale or charge off, REO disposition, and note sale, all credit events.

Code 01, prepaid or matured, is a voluntary exit and is not a default. Codes 06 and 96, repurchase and defect settlement, are excluded entirely: they mean the loan was bought back because of an underwriting defect, which is a problem between the seller and Freddie Mac rather than a borrower failing to pay.

Pandemic forbearance is counted and reported but does not change the definition.

### UCI

This dataset has no time dimension, so it cannot be replayed. Instead it is used as a test rig. 12,000 records form a reference set, a model is trained on them, and then 20 synthetic periods are drawn from the remaining 18,000 with drift injected at a controlled intensity, running through four phases of five periods each: mild, moderate, severe and recovery.

Because the true amount of drift in each period is known by construction, this is the only place in the project where detector behaviour can be checked against ground truth.

Two features are deliberately dropped from the model inputs, `SEX` and `MARRIAGE`, because using protected characteristics in a credit decision is not acceptable practice. `AGE` is kept but monitored only: it is watched for population shift and never fed to the model.

## 5. Splitting time

One rule, applied identically to both real datasets:

- the **first five usable years** train the baseline
- the **last three usable years** are held back and scored exactly once at the very end
- **everything in between** is the monitoring window

| | Reference | Monitoring | Final evaluation |
|---|---|---|---|
| SBA | FY1991 to FY1995 | FY1996 to FY2018 (23 years) | FY2019 to FY2021 |
| Freddie Mac | 1999 to 2003 | 2004 to 2020 (17 years) | 2021 to 2023 |

"Usable" means the outcome window has finished. For SBA that rules out FY2022 onward, because as of the data extract date of 30 September 2026 those loans have not had their full five years.

The split is fixed in code before any results exist, and the final evaluation years are never used to pick a model, tune a threshold or choose a strategy. They exist to answer one question once: given whatever model each policy ended up with, how good is it on years nobody looked at?

This matters more than it sounds. If you tune anything against your test set, even by glancing at it and trying something else, your reported number stops being an estimate of future performance and becomes a description of the past.

## 6. Features

17 features for SBA, all of them known at the moment the loan is approved. Nothing that becomes known later can be used, because in production it would not exist yet.

**8 numeric:** gross approval amount, SBA guaranteed amount, term in months, initial interest rate, jobs supported, SBA guarantee share (derived), and two missing indicators for interest rate and secondary market flag.

**9 categorical:** business type, project state, revolver status, business age, processing method, collateral indicator, secondary market indicator, fixed or variable rate indicator, and NAICS sector (the first two digits of the industry code, derived).

Three details worth knowing:

- **Missing indicators carry real information.** `interest_rate_missing` turns out to be the single largest drift signal in the whole project. The fact that a field is blank is itself data.
- **Rare categories are grouped.** Any category below 0.5% of the training rows becomes "Other", fitted on the training window only. A category never seen in training also becomes "Other" at scoring time, which is what stops the pipeline breaking when a new state or industry code appears.
- **Numeric blanks are filled with 0 after the indicator is created.** The indicator holds the information; the 0 is just a placeholder that keeps the gradient boosting binning from failing on a column that is entirely blank.

Freddie Mac uses 21 features on the same principle: 10 numeric (credit score, original and combined loan to value, debt to income, interest rate, unpaid balance, loan term, mortgage insurance percentage, number of units, number of borrowers), 5 missing indicators, and 6 categorical fields (first time homebuyer, occupancy status, origination channel, property state, property type, loan purpose).

## 7. The baseline models

Two models are trained on the reference years, with time ordered cross validation, meaning every validation fold is later in time than the data used to fit it. Shuffled cross validation would let the model learn from the future and would flatter the result.

**SBA, scored on FY1996, the first monitoring year:**

| Model | Cross validated AUC | AUC on FY1996 | Predicted default rate | Actual |
|---|---|---|---|---|
| Logistic Regression | 0.8652 | 0.8784 | 7.91% | 8.13% |
| Gradient Boosting | 0.9501 | **0.9440** | 7.54% | 8.13% |

Gradient boosting wins by about 7 percentage points of AUC and becomes the model used throughout. Logistic regression is kept as a transparent reference point, which matters in credit risk where a model often has to be explainable to a regulator.

Both are well calibrated out of the box, within roughly one percentage point of the true rate, so no recalibration step was added.

A cost sensitive decision threshold is also computed, on the stated assumption that a missed default costs five times as much as wrongly declining a good loan: 0.15 for the gradient boosting model, 0.14 for logistic regression. That five to one ratio is an assumption, clearly labelled as such, not something measured from the data.

## 8. Measuring drift

Three tests run at every monitoring period.

### Population Stability Index

PSI is the standard drift measure in credit risk. The idea is simple: chop the reference distribution into 10 bins, see what share of the reference fell into each, see what share of the new data falls into each, and add up the differences.

A PSI of 0 means the two distributions are identical. The conventional reading is below 0.10 nothing is happening, 0.10 to 0.25 something is happening, above 0.25 act.

Two implementation details that matter:

- **Numeric features use quantile bins**, so skewed distributions like loan amount do not end up with nine empty bins and one full one.
- **Features with 10 or fewer distinct values use value based bins.** This sounds minor and is not. `interest_rate_missing` is 1 for every single reference loan. Quantile binning collapses on a constant column and reports no drift forever, silently missing the largest real signal in the dataset.

### Kolmogorov Smirnov and chi-square

PSI is a summary, and summaries hide things. KS is run on numeric features and reports the largest gap between the two cumulative distributions. Chi-square is run on categorical features and tests whether the category mix has changed more than chance would explain. Together they catch shapes of change that a binned summary can smooth over.

### Turning three numbers into one alert

A period is as bad as its worst signal:

| Trigger | Amber | Red |
|---|---|---|
| Score PSI | 0.10 to 0.25 | above 0.25 |
| Number of red features | 3 or more amber | 3 or more red |
| AUC drop from reference | 0.02 or more | n/a |

The score PSI catches drift in what the model outputs. The feature count catches broad shifts across many inputs. The AUC check catches performance falling even when the inputs look calm, which is the fingerprint of concept drift.

Having three independent triggers rather than one is what makes the next finding possible.

## 9. The replay

Starting at the first monitoring year, the baseline model scores that year's loans, the three drift tests run against the reference distribution, an alert is raised, and the process moves to the next year. The model is never updated during the replay itself; updating is the job of the retraining policies in section 10.

On SBA, **all 23 monitoring years come back red**. That looks like a bug and is not. What is interesting is *why*, and it changes over time:

| Years | Score PSI | What actually triggered red |
|---|---|---|
| FY1996 to FY2004 | green, 0.003 rising to 0.047 | feature drift only, 4 to 10 features red |
| FY2005 | amber, 0.123 | feature drift, now joined by score drift |
| FY2006 onward | red, 0.344 rising to a peak of 1.105 in FY2012 | all three triggers |

**For nine years the model's output distribution looked completely stable while its inputs had already moved a long way.** In FY1996 the score PSI is 0.0033, about as calm as a number can be, yet four features are already red and `sba_guaranteed_pct` has a PSI of 4.84. Anyone monitoring only the model's score would have seen nothing wrong until FY2005. Anyone monitoring the inputs would have seen it in year one.

That is the single most useful lesson in this project, and it is why the alert logic has three triggers instead of one.

Freddie Mac behaves differently and more normally: **1 green, 3 amber, 13 red** across 17 years, with the calm years at the start. Its score PSI never leaves green at all, peaking at 0.058. Every red period there is driven purely by feature drift, with origination channel and original interest rate in the top three drifters in all 13.

## 10. Retraining

### The label delay problem

You cannot retrain on loans whose outcome you do not know yet, and with a five year outcome window you do not know an SBA loan's outcome for five years.

Concretely: to retrain at year Y you need four years of known labels, three to train on and one held out to compare against. With the five year window, at year Y the most recent known year is Y minus 6. So the earliest possible retrain is **FY2000**, and even then it trains on FY1991 to FY1994 data, which is six years stale before it is deployed.

This is not a limitation of the project, it is the actual constraint every credit risk team lives with, and ignoring it is the most common way these analyses end up over optimistic. The practical consequence is that the model has to survive FY1996 to FY1999 on its own, with no possibility of help.

Freddie Mac's 24 month window makes this much less painful, which is part of why its results are less dramatic.

### Champion and challenger

Retraining does not mean replacing. At each opportunity:

1. Train a **challenger** on a rolling 3 year window of the most recent data with known labels.
2. Score both the current **champion** and the challenger on a held out year neither was trained on.
3. Bootstrap the AUC difference 200 times to get a confidence interval.
4. Promote the challenger only if it beats the champion by at least **0.005 AUC**.

Step 4 is the point. Without a promotion threshold you swap models on noise, and every swap in a real organisation costs validation, documentation and sign off. On SBA this rule rejected the challenger 6 times out of 19.

The retrained models reuse the baseline's hyperparameters rather than re-tuning. With only three years in the window there are just two cross validation folds available, which is far too thin for a meaningful search, and tuning on it would mostly fit noise.

### The three policies

- **Never retrain.** Keep the original model forever. The control group.
- **Scheduled.** Attempt a retrain every year, regardless of what the monitoring says.
- **Triggered by drift.** Attempt a retrain only when the alert is red.

## 11. Results

### SBA 7(a)

| Policy | Attempts | Promotions | Mean AUC, monitoring | AUC on the held back years |
|---|---|---|---|---|
| Never retrain | 0 | 0 | 0.7513 | 0.7441 |
| Scheduled | 19 | 13 | 0.8246 | **0.9251** |
| Triggered by drift | 19 | 13 | 0.8246 | **0.9251** |

The untouched model falls from 0.9440 on FY1996 to a low of 0.5987 on FY2010 and limps back to 0.7597 by FY2018. The retrained models recover above 0.94 from FY2013 onward. On the three held back years the gap is **0.1810 AUC**, which is about a fifth of the retrained model's sorting power.

The two retraining policies are **identical**, because when every period is red the drift trigger fires every year and simply becomes the yearly schedule. This is a real property of SBA, not a coding mistake, and it is worth saying plainly rather than hiding: on this dataset the sophisticated policy bought nothing over the simple one.

Two honest wrinkles:

- **Even retraining does not save you during the crisis.** Through FY2006 to FY2010 the retrained models still sit between 0.63 and 0.80, because the rolling window is training on pre crisis data while the crisis is happening. No retraining policy can learn a regime that has not yet produced labels. The recovery only arrives from FY2011, when crisis era loans finally enter the training window.
- **The six rejections** happened at FY2000, FY2001, FY2007, FY2008, FY2015 and FY2017.

### Freddie Mac

| Policy | Attempts | Promotions | Mean AUC, monitoring | AUC on the held back years |
|---|---|---|---|---|
| Never retrain | 0 | 0 | 0.7866 | 0.7532 |
| Scheduled | 16 | 9 | 0.7970 | **0.7836** |
| Triggered by drift | 13 | 9 | 0.7970 | **0.7836** |

Here the policies finally separate, and the result is the most practically useful one in the project: **triggering on alerts reached exactly the same accuracy as retraining every year, using 13 retraining runs instead of 16.** Three fewer full retrain and validation cycles for free.

The effect of retraining is smaller than on SBA, about 3 points of AUC rather than 18, because mortgage drift here is gentler. The worst year is 2019 at 0.6795.

### UCI benchmark

Baseline AUC 0.819 on held out data, 0.755 cross validated. All 20 synthetic periods were flagged red, including the mildest injected level, where the score PSI was only 0.036 and the feature level tests did the work.

The honest reading: the detector is **sensitive rather than precise**. It will not miss drift, and it will also fire on small amounts of it. For a monitoring system that feeds a human reviewer this is the right trade, since a missed alert costs more than an extra one. For a system that triggers expensive retraining automatically, the thresholds would need raising.

## 12. Reading the dashboard

Seven tabs.

| Tab | What to look at |
|---|---|
| **Overview** | Headline cards and the alert timeline. The wall of red is the honest result. The default rate chart shows the crisis spike to 27.5% on FY2007 loans. |
| **Drift Analysis** | Score PSI against the alert bands, the feature PSI heatmap, and baseline AUC decay. The heatmap is where you see interest rate fields turn deep red from FY2009. |
| **Model Performance** | The key chart. Grey never recovers, green and blue climb back above 0.94. Blue is hidden underneath green because the two policies are identical here. Below it, every retrain attempt with its bootstrap interval and the promote or reject decision. |
| **Strategy Comparison** | The summary table, the held back year results, and the AUC lift bars. |
| **Freddie Mac** | The whole pipeline again on mortgages, where the alert mix is 1 green, 3 amber, 13 red. |
| **UCI Benchmark** | The synthetic drift test, with the injected intensity shown against what the detector reported. |
| **About** | Method, thresholds and dataset summary. |

## 13. Questions an interviewer will ask

### On the modelling

**Why AUC rather than accuracy?**
The default rate is about 9%, so a model that predicts "no default" for everyone scores 91% accuracy and is worthless. AUC asks a question that survives class imbalance: given one loan that defaulted and one that did not, how often does the model rank the defaulter as riskier? 0.50 is a coin toss, 1.00 is perfect.

**Why gradient boosting over logistic regression?**
On this data it is worth about 7 points of AUC, 0.9440 against 0.8784. Logistic regression is still trained and reported, because in regulated credit decisioning explainability is sometimes worth more than accuracy, and a reviewer will want to know you understand that rather than reaching for the strongest model by reflex.

**Is the model calibrated, or just good at ranking?**
Both, as it happens. Gradient boosting predicted 7.54% against an actual 8.13%, logistic regression 7.91% against 8.13%. That is close enough that no recalibration layer was added. Ranking and calibration are different properties and a model can be excellent at one and poor at the other, which matters when the score feeds a pricing decision rather than just an approve or decline.

**How did you choose the decision threshold?**
From an explicit cost assumption, that a missed default costs five times a wrongly declined good loan, which gives 0.15. Sensitivities at three to one and ten to one are also computed. The ratio is an assumption and is labelled as one, because in a real engagement that number comes from the business, not from the data.

### On preventing leakage

**How do you know there is no look ahead?**
Four separate guards. Features are restricted to fields known at origination. Cross validation is time ordered, so every validation fold is later than its training data. Retraining can only use years whose outcome window has closed, which is enforced in `drift/label_delay.py` and covered by 12 tests. And the final evaluation years are scored exactly once, at the end, by a model that was already fixed.

**What is label delay and why does it matter so much here?**
An SBA loan's outcome is not known for five years. So at year Y the freshest usable labels are from year Y minus 6. The first retrain is therefore impossible before FY2000, and every retrained model is six years stale on the day it goes live. Most write ups of this kind quietly retrain on data they could not have had, which makes retraining look better than it is.

**Why is the test set scored only once?**
Because a test set you look at twice is a validation set. If you check the held back years, see a disappointing number and change anything, that number stops estimating future performance.

### On the drift work

**What is PSI and where does 0.25 come from?**
PSI compares the share of data falling into each of 10 bins, reference against current, and sums the divergence. The 0.10 and 0.25 cut points are the long standing convention in credit risk rather than anything derived from this data. They are configurable in one place, `drift/config.py`.

**Why three drift tests instead of just PSI?**
Because on this dataset one test would have failed. For nine years the score PSI stayed green while four or more features were already red. A team watching only the model output would have seen nothing wrong until FY2005; a team watching the inputs saw it in FY1996. PSI is a binned summary and summaries hide shape changes, so KS covers numeric features and chi-square covers categorical ones.

**You said every SBA period is red. Isn't that a broken detector?**
It is a real property of the data, and the project says so rather than tuning until the chart looks nicer. Two structural things happen: the 2007 to 2009 crisis reshapes default behaviour, and SBA begins recording interest rates in FY2009 after leaving the field blank for every loan before then, which alone produces a PSI around 18. There is no quiet stretch for a trigger to skip. The Freddie Mac dataset exists in this project precisely because this result on its own would not be informative.

**How do you tell real drift from a data quality problem?**
You often cannot, automatically, and this project has a clean example. The largest SBA signal is three interest rate fields jumping in FY2009 and then sitting perfectly flat. Genuine population change drifts; a reporting change steps once and stops. The shape is the clue, but confirming it means going to the source, which is what the raw file check in this repository did. This is the main argument against fully automated retraining triggers.

### On the retraining design

**Why champion and challenger rather than just deploying the new model?**
Because most retrains are not improvements. Here 6 of 19 challengers failed to clear the bar and were rejected. In a real organisation every model swap costs validation, documentation and sign off, so swapping on noise is expensive as well as pointless.

**Why bootstrap the comparison?**
A single AUC difference on one year of data is a point estimate with real uncertainty. Resampling 200 times gives an interval, so a 0.004 difference can be recognised as noise rather than treated as a win.

**Why 0.005 as the promotion threshold?**
It is deliberately conservative and is a judgement call, not an optimum. Set it too low and you churn models; too high and you sit on a stale one. It is a single constant in config, which is where a business owner would want it.

**Why reuse hyperparameters instead of re-tuning each time?**
A 3 year rolling window gives only two time ordered cross validation folds. Searching a grid on that would mostly fit noise, and it would multiply runtime for a result you could not trust.

**Scheduled and drift triggered tie on SBA. Doesn't that make the whole exercise pointless?**
On SBA, yes, and that is the finding. On Freddie Mac they separate, and the trigger reaches identical accuracy with 13 retrains instead of 16. The general conclusion is the useful one: alert driven retraining is never worse than a fixed schedule, and it is cheaper whenever drift is episodic rather than constant.

**Why does AUC stay poor through the crisis even with retraining?**
Because of label delay. During 2007 the only labelled data available is pre crisis, so the retrained model is learning the old regime while the new one is happening. No policy can fix this; it is a hard limit of supervised learning under delayed outcomes. Performance recovers from FY2011, once crisis era loans have outcomes.

### On engineering and production

**How would you run this in production?**
Drift metrics need no labels, so PSI, KS and chi-square run on every new batch in real time and feed the alert. AUC cannot be computed until the outcome window closes, so it arrives years later and acts as confirmation rather than as a trigger. The retraining decision stays behind a human review gate, for the data quality reason above.

**What would you monitor beyond what is here?**
Prediction volume and input null rates, which catch pipeline breakage faster than any statistical test. Score distribution by segment rather than only in aggregate, since a large shift in a small segment averages away. And approval and decline rates, because the business feels those before any metric moves.

**Why are results stored as JSON and read by the dashboard?**
So the dashboard never trains anything. It renders exactly what the pipeline produced, it starts instantly, and anyone can clone the repository and see the real results without the raw data, which cannot be redistributed anyway.

**What is tested?**
59 tests. 30 on the drift mathematics, including the binary indicator case that naive quantile binning gets wrong, empty inputs, unseen categories and the alert logic. 12 on label delay, which is the no look ahead guarantee. 10 on label derivation, including boundary cases exactly at the five year mark. 7 on the split rule.

### On judgement

**What was the hardest decision?**
Treating 211,323 recent exempt loans as censored and dropping them, rather than labelling them non defaults. Keeping them would have added 14% more training data and improved every headline number, and it would have been wrong, because their outcome is genuinely unknown.

**What would you do differently?**
Re-tune hyperparameters on a longer rolling window, test window lengths other than three years, and add a drift signal that distinguishes a one time step change from gradual movement, which would have flagged the FY2009 interest rate shift as a data issue rather than a population shift.

## 14. Limitations and what I would do next

1. **This is a retrospective study.** AUC during the monitoring window is computed using outcomes that are known today but would not have been known at the time. The drift metrics are honest in real time; the AUC curve is a hindsight view used to check whether drift actually corresponded to damage.
2. **SBA drift is constant, which weakens the comparison.** Every period being red means the two retraining policies collapse into one. Freddie Mac rescues the comparison, but a dataset with long quiet stretches would test the trigger harder.
3. **The largest SBA signal is a reporting change, not borrower behaviour.** The pipeline scores it the same as a genuine population shift. Separating the two still needs a person reading the feature level alerts.
4. **Freddie Mac uses the public sample files**, roughly a 1 in 20 draw, so year to year movements carry more noise than they would on the full book.
5. **One model family and one window length.** Only gradient boosting is carried through the replay, always on a 3 year rolling window. Longer windows, sample weighting and online updating are all worth comparing, and the framework is built to take them.

## 15. Running it yourself

The dashboard reads the committed JSON results, so it runs with no raw data at all.

```bash
git clone https://github.com/vamshidharsingh/credit-risk-model-drift-monitoring.git
cd credit-risk-model-drift-monitoring
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
streamlit run streamlit_app.py
```

```bash
pytest tests/ -v
```

Regenerating the results needs the raw data in `data/`:

```bash
python -c "from drift.freddie.benchmark import run_freddie_benchmark; run_freddie_benchmark()"
```

```bash
python -c "from drift.uci.benchmark import run_uci_benchmark; run_uci_benchmark()"
```

## 16. Where things live in the code

| File | What it holds |
|---|---|
| `drift/config.py` | Every threshold, path, seed and tuning grid. Change behaviour here, not in the pipeline. |
| `drift/features.py` | Feature engineering per dataset, plus rare category grouping. |
| `drift/pipeline.py` | Model pipelines and time ordered cross validation. |
| `drift/drift.py` | PSI, KS, chi-square and the alert classification. |
| `drift/label_delay.py` | The no look ahead guarantee. Small file, load bearing. |
| `drift/splits.py` | The fixed reference, monitoring and evaluation split. |
| `drift/replay.py` | Walks the model forward through the monitoring years. |
| `drift/retrain.py` | Champion and challenger logic and the three policies. |
| `drift/metrics.py` | AUC, calibration and cost sensitive thresholds. |
| `drift/charts.py` | Shared Plotly styling and the event annotations. |
| `drift/sba/`, `drift/freddie/`, `drift/uci/` | Per dataset loaders and label derivation. |
| `streamlit_app.py` | The dashboard. Reads JSON, trains nothing. |
| `reports/*.json` | Pipeline output. The dashboard's only data source. |
| `docs/decisions.md` | Every design decision with its reasoning. |
