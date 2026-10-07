# Account Risk and Rep Prioritisation Model

## Executive summary

This prototype helps sales representatives identify eligible accounts that may need attention next month, understand the signals behind a risk score, and order the eligible accounts using a separate account-importance score.

The predictive layer is one account-risk scoring system implemented with three separately fitted XGBoost binary classifiers. This means it is **not one multi-output classifier and not an ensemble voting across three models**. The three classifiers share the same feature definitions and train/test design, but each learns a different binary target:

- `high_risk_probability`: probability that the next eligible monthly snapshot is labelled high risk.
- `churn_probability`: probability that the next eligible monthly snapshot meets the operational churn rule.
- `loss_probability`: probability that the next eligible monthly snapshot meets the operational order-value-loss rule.

An account is only scored if it has at least three orders in the trailing 18 months at the scoring snapshot. Accounts below that threshold are returned with `insufficient_data` and blank model probabilities. This is a deliberate abstention, not a low-risk prediction.

For action ordering, the prototype calculates account importance as the account's percentile rank in 18-month revenue, then combines that with high-risk probability:

$$
\text{priority score} = \text{high-risk probability} \times (0.5 + 0.5 \times \text{account-importance percentile})
$$

Risk prediction and account importance remain separate fields in the output so a rep can see why an account ranks highly.

## Business problem and unit of analysis

The intended user is a sales representative managing a portfolio of business accounts. The model is meant to identify behaviour that merits a proactive review, not to make an autonomous customer-contact or commercial decision.

The unit of analysis is `account_id`, treated as the customer/account grouping specified in the assignment. The data is an anonymised Brazilian marketplace dataset adapted for the exercise, not HEINEKEN sales data. Product categories and customer reviews should therefore be interpreted as proxy signals in this demonstration.

## Snapshot and forecast design

- Snapshot frequency: monthly, at month end.
- Feature history: up to the 18 months ending at the snapshot date.
- Forecast target: the immediately following monthly snapshot.
- Data range in the supplied exercise: January 2017 through August 2018; model snapshots with a valid following month run through July 2018 and target labels run through August 2018.
- Point-in-time rule: order-derived features use orders dated no later than the feature snapshot. Review-derived features use reviews with `review_creation_date` no later than the feature snapshot.
- Monthly windows are implemented with date-relative cutoffs (for example, `snapshot_date - 3 months`), not necessarily whole calendar-month buckets.

Training and evaluation rows must be eligible at both the feature snapshot and the next-month target snapshot. The target label is undefined when the next snapshot does not meet the account-history eligibility rule, so these rows are excluded rather than treated as negative examples.

## Eligibility and no-score handling

The business rule is:

- `eligible = 1` when the account has at least 3 distinct orders in the trailing 18-month history at the feature snapshot.
- `eligible = 0` otherwise.

At the latest supplied snapshot, 31 August 2018, the output roster has 14,989 accounts: 9,577 eligible for a model score and 5,412 marked `insufficient_data`.

For an ineligible account:

- `score_status` is `insufficient_data`.
- `high_risk_probability`, `churn_probability`, `loss_probability`, `predicted_high_risk`, `priority_score`, and `priority_rank` are blank.
- No model estimate should be shown or interpreted as low risk.

This policy is especially important for intermittent and low-history accounts: the model does not manufacture certainty where the chosen minimum history is absent.

## Target definitions

The labels are rule-defined operational outcomes derived at each eligible monthly snapshot. The classifiers forecast whether each outcome will hold at the next eligible monthly snapshot.

### Churn

For an eligible snapshot, calculate:

- `days_since_last_order`: snapshot date minus the latest order date in the 18-month history.
- `normal_cycle_days`: median gap in days between distinct order dates in that history.

Then:

$$
\text{churn} = 1 \quad \text{if} \quad
\text{days since last order} \geq 2.5 \times \text{normal cycle days}
$$

If a valid positive normal cycle cannot be calculated, the current label implementation sets churn to 0. The minimum three-order eligibility rule generally provides repeated order observations, but distinct order dates still control whether a cycle can be calculated.

### Loss

For an eligible snapshot, compare the latest order value with the mean value of earlier orders in the 18-month history:

$$
\text{loss} = 1 \quad \text{if} \quad
\text{latest order value} \leq 0.3 \times \text{mean earlier order value}
$$

The implementation also requires a positive overall mean order value. This detects a very small latest basket relative to that account's own history; it does not represent lost revenue measured against a causal counterfactual.

### High risk

The binary label is the union of the two rule labels:

$$
\text{high risk} = \text{churn} \;\text{OR}\; \text{loss}
$$

The high-risk, churn, and loss classifiers are fit separately. Their probabilities are not additive: churn and loss can overlap, and separate models do not guarantee that the high-risk probability equals the sum or union of the two component probabilities. The high-risk classifier is the primary rep-prioritisation risk output; churn and loss are companion risk indicators for understanding the type of issue.

## Model inputs

The final high-risk predictor list contains 22 numeric, point-in-time features. Latitude, longitude, city/state text and city/state context variables are not model inputs.

| Feature | Business meaning |
|---|---|
| `orders_last_18m` | Number of orders in the trailing 18 months; also supports the eligibility gate. |
| `revenue_last_18m` | Sum of order item prices in the trailing 18 months; freight is excluded. |
| `avg_order_value_18m` | Mean order value across the trailing 18-month order history. |
| `max_order_value_18m` | Largest order value in the trailing 18-month order history. |
| `days_since_last_order` | Recency of the account's latest order at the snapshot. |
| `normal_cycle_days` | Median interval between distinct order dates in the trailing 18 months. |
| `recent_3m_orders` | Orders from the date-relative recent three-month window. |
| `recent_6m_orders` | Orders from the date-relative recent six-month window. |
| `revenue_drop_ratio_3m` | Recent three-month revenue divided by revenue in the preceding three-month window. A value below 1 indicates less recent spend; undefined if the comparison window is zero. |
| `revenue_drop_ratio_6m` | Recent six-month revenue divided by revenue in the preceding six-month window. A value below 1 indicates less recent spend; undefined if the comparison window is zero. |
| `recent_3m_revenue` | Sum of order values in the recent three-month window. |
| `active_months_3m` | Number of distinct order months with activity in the recent three-month cutoff window. |
| `active_months_6m` | Number of distinct order months with activity in the recent six-month window. |
| `active_months_12m` | Number of distinct order months with activity in the recent 12-month window. |
| `active_months_18m` | Number of distinct order months with activity in the trailing 18 months. |
| `latest_order_value` | Total item-price value of the latest order. |
| `avg_historic_order_value` | Average order value excluding the latest order; an account-specific historical baseline. |
| `value_drop_ratio` | Latest order value divided by the mean value of earlier orders; below 1 indicates a smaller latest order. |
| `recent_review_count_6m` | Number of reviews created in the trailing six months. |
| `latest_review_vs_prior_avg` | Latest review score minus the mean of earlier scores in the trailing 18-month review history. |
| `review_score_trend_3m` | Mean review score in the recent three months minus the mean in the preceding three months; undefined if either window has no reviews. |
| `negative_review_share_6m` | Share of trailing six-month reviews with score 1 or 2. The six-month window has been corrected and checked against source reviews. |

For ratios whose comparison-window denominator is zero, review aggregates with no observations, and cycle metrics without enough distinct order dates, the feature remains missing (`NaN`). XGBoost's native missing-value routing is used; these values are not imputed. Counts and sums are zero only when their feature definition means no qualifying events.

The full calculations, units, edge cases, point-in-time rules, and current global gain rankings are maintained in [model_feature_dictionary.csv](model_feature_dictionary.csv).

## Model and scoring implementation

The notebook fits three XGBoost binary classifiers with the same predictor set and broad hyperparameter configuration:

- 500 trees (`n_estimators=500`)
- maximum depth 4
- learning rate 0.04
- row and column subsampling 0.8
- L2 regularisation 2.0
- minimum child weight 3
- binary logistic objective
- positive-class weighting based on the training class balance
- fixed random seed 42

The high-risk model uses a 0.50 classification threshold for its binary `predicted_high_risk` flag. The underlying probability is also retained; the thresholded flag is not a substitute for the probability or the priority ranking.

Evaluation classifiers are trained only on target snapshots before June 2018 and tested on target snapshots from June through August 2018. After evaluation, separate deployment/scoring classifiers are refit on all available eligible and labeled history, then applied to the latest snapshot. This preserves the temporal holdout for reported performance while making use of all labels for the latest account scores.

## Holdout performance

The latest saved holdout has 26,816 account-month observations: 16,287 not-high-risk and 10,529 high-risk. The positive/high-risk rate is 39.26%. The confusion matrix below uses rows for actual class and columns for predicted class, at the 0.50 high-risk threshold.

| Actual \\ Predicted | Not high risk | High risk |
|---|---:|---:|
| Not high risk | 13,334 | 2,953 |
| High risk | 1,439 | 9,090 |

| Metric | Result | Interpretation |
|---|---:|---|
| Accuracy | 83.62% | Overall share of correct thresholded predictions. |
| Balanced accuracy | 84.10% | Mean of positive-class recall and negative-class recall. |
| High-risk precision | 75.48% | Of accounts flagged high risk, the share labelled high risk next month. |
| High-risk recall | 86.33% | Of next-month high-risk cases, the share flagged by the model. |
| High-risk F1 | 80.54% | Harmonic mean of precision and recall. |
| ROC-AUC | 0.8986 | Ranking discrimination across possible thresholds. |
| PR-AUC (average precision) | 0.8277 | Precision-recall ranking quality; compare with 39.26% test prevalence. |
| Brier score | 0.1253 | Mean squared probability error; lower is better. |

The component-model ranking metrics on the same eligible temporal holdout are:

| Outcome | ROC-AUC | PR-AUC (average precision) |
|---|---:|---:|
| Churn | 0.9512 | 0.8176 |
| Loss | 0.9030 | 0.7523 |

These are historical, rule-label prediction metrics for this dataset and split. They are not evidence of causal retention impact, guaranteed future-market performance, or performance on current HEINEKEN accounts.

## Global feature importance

The high-risk classifier's XGBoost gain ranking is a global summary of which predictors contributed to tree splits. It is not a causal ranking and can distribute importance across correlated features. The strongest current signals are:

1. `value_drop_ratio`
2. `active_months_3m`
3. `normal_cycle_days`
4. `days_since_last_order`
5. `recent_3m_orders`
6. `avg_historic_order_value`
7. `recent_6m_orders`
8. `recent_3m_revenue`
9. `active_months_6m`
10. `latest_order_value`

Review signals also enter the fitted model. Global gain and split counts for all 22 features are in [high_risk_feature_importance.csv](../outputs/high_risk_feature_importance.csv).

## Per-account explanations

XGBoost prediction contributions (`pred_contribs`) are exported for every eligible account and every model feature. Each contribution is in raw-margin/log-odds units:

- positive contribution: pushes the model's high-risk score upward;
- negative contribution: pushes it downward;
- contribution sum plus the model bias: reconstructs the model raw margin, which is verified in the notebook.

The account score CSV contains the top three drivers in each direction for scanning. The detailed long-format file contains all 22 feature values and contributions for each eligible account. These are explanations of the model calculation, not verified causes of customer behaviour.

## Account importance and priority

Account importance is intentionally separate from predicted risk:

- `account_revenue_18m`: item-price revenue over the account's trailing 18-month snapshot window, with zero assigned for no activity in that window.
- `account_importance_score`: percentile rank of that revenue among the full account roster at the snapshot, between 0 and 1.

The operational ranking is:

$$
\text{priority score} = P(\text{high risk}) \times (0.5 + 0.5 \times \text{account importance score})
$$

The risk score remains the principal signal. Revenue importance can uplift its ordering by up to a factor of two, while never reducing a nonzero risk probability below half its original value. The 0.5 floor and equal revenue percentile treatment are business-policy choices for the prototype, not weights learned or validated from outcomes. `priority_rank` ranks eligible accounts in the current snapshot; ineligible accounts have no priority score or rank.

This is a queue-ordering index, not expected revenue lost, profit at risk, or an estimate of intervention return. Revenue is used as an available value proxy because margins and account-level profit are not provided.

## Data lineage and scope

| Dataset/file | Use in this version | Why |
|---|---|---|
| `order_lines.csv` | Active, primary purchase source. Used to form one order per account/order, sum item prices, derive order dates, frequency, recency, order value, and labels. | It is the prepared joined table with account identifiers, order dates, item prices and related fields needed for the final predictors and labels. |
| `order_reviews.csv` | Active review source. Joined to accounts through `order_id`; only review score and review creation date are used for selected rolling review features. | Enables recent review level and trend features without using future reviews. Review text is not used. |
| `geolocation.csv` | Loaded and used by the builder to calculate geographic context fields, but those fields are excluded from the final 22 predictors and do not affect final model scores. | Coordinates and city/state aggregates were deliberately removed from the final model input list. |
| `orders.csv`, `order_items.csv`, `customers.csv`, `order_payments.csv`, `products.csv` | Not directly read by the final builder/model workflow. | Their relevant prepared order/account/item fields are already represented in `order_lines.csv`; separate fields such as payment method, product dimensions, customer identifiers, and product text are not final predictors. |

The model does not use raw review comments, geographic location, seller data, customer names/IDs other than the account grouping key, payment type, delivery-lateness fields, or product text/category indicators as final predictors.

## Output data contract

The generated current account-level audit output is `outputs/current_account_risk_scores.csv`, with one row per account for the latest available snapshot. Important fields include:

- `account_id`, `snapshot_month`, `eligibility_order_count_18m`, `eligible`, `score_status`
- `high_risk_probability`, `churn_probability`, `loss_probability`, `predicted_high_risk`
- `account_revenue_18m`, `account_importance_score`, `priority_score`, `priority_rank`
- top risk-increasing and risk-reducing feature names, values, and log-odds contributions

The detailed explanation file, `outputs/current_account_feature_contributions.csv`, is in long format: one row per eligible account-feature pair. The holdout classification file, `outputs/high_risk_test_predictions.csv`, includes actual next-month labels and predictions for evaluation. These account-level generated CSVs are excluded from the public repository by default; regenerate them by running the model notebook. All probability values refer to the next eligible monthly snapshot; output must display the `snapshot_month` and the data freshness date.

For a compact rep-facing table, `outputs/current_account_priority_queue.csv` contains the account identifier, snapshot/status, risk/churn/loss probabilities, importance score, priority score/rank, and leading risk drivers without the full feature vector. It is generated locally and is also excluded from public Git by default.

## Guidance for presentation and an LLM/agentic front end

A presentation can tell the story in this order: business problem and account unit; eligibility and no-score rule; rule-based outcome definitions; one-month-ahead forecasting design; final feature groups; temporal holdout performance and confusion matrix; example account explanation; importance-adjusted rep ranking; limitations and pilot next steps.

For an interactive front end or LLM agent:

- Treat the CSVs and model artifacts as the source of truth. The LLM should explain supplied fields, not invent scores, features, history, or causal reasons.
- Show risk, churn, loss, importance, and priority as separate values. Label component probabilities as separate estimates; do not sum them.
- Show `insufficient_data` as “not scored: fewer than 3 orders in the last 18 months,” not as low risk. Do not generate an outreach recommendation based on a missing score.
- For a risk explanation, use the supplied signed feature contributions and the feature dictionary. Describe them as factors that pushed this model score up/down, not confirmed reasons the account changed.
- Display the snapshot date, prediction horizon (next eligible month), the eligibility condition, and that the sample ends 31 August 2018.
- Keep any generated talking point, offer, call script, or recommended action grounded in the account data and subject to representative review. Do not allow an agent to autonomously contact an account, change prices, or issue offers based only on a model rank.
- Keep an audit trail of the score snapshot, displayed explanation, generated suggestion, user edits/approval, and eventual outcome if this progresses beyond a prototype.

## Limitations and appropriate interpretation

- This is an MVP prototype trained on an adapted historical marketplace dataset, not production HEINEKEN data.
- The target is a transparent rule-defined proxy. High apparent performance partly reflects the continuity of behaviour used by the rules; it does not prove that intervention prevents churn or protects revenue.
- The data end date is 31 August 2018. The generated current-roster file is “current” only relative to this supplied dataset, not the calendar date of deployment.
- The importance percentile is relative to the supplied account roster and changes when that roster changes.
- Separate churn/loss/high-risk models need not produce mathematically consistent probabilities; component probabilities are presented as independent model estimates.
- A business pilot should confirm the operational definitions, test the workflow with representatives, and assess actual outcomes before the system is treated as a production decision tool.
