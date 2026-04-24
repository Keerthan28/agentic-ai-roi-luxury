# Methodology

## 1. Objective

Quantify the ROI of deploying agentic AI in luxury retail at the per-customer level, so that Deloitte client engagements can prioritize segments and use cases with confidence.

## 2. Data

| Dataset                    | n raw | n cleaned | Use                     |
| -------------------------- | ----- | --------- | ----------------------- |
| Synthetic (AI-positive)    | 300   | 253       | Feature pipeline + training |
| Real Qualtrics survey      | 43    | ~40       | Transfer + retrain test |

Cleaning uses a per-column VALID_VALUES whitelist; any cell whose value is not in the allowed set (for example Qualtrics-exported question text appearing in a response cell) is coerced to `NaN`. Rows with a valid `Q1` (age band) are retained.

## 3. OSINT Retention Benchmarks

All of the following are drawn from publicly available industry reports (Bain & Company Luxury Market Study; McKinsey Luxury Report; Deloitte Global Powers of Luxury Goods; BCG x Altagamma).

| Assumption                          | Value                | Source                                  |
| ----------------------------------- | -------------------- | --------------------------------------- |
| Baseline luxury retention rate      | 82%                  | Bain / McKinsey luxury reports          |
| AI-enhanced retention uplift (max)  | 15 pts               | McKinsey personalization study          |
| Avg luxury CLV multiplier           | 8.5x annual spend    | BCG x Altagamma                         |
| Churn cost multiplier               | 5x                   | Industry standard acquisition vs. retention |
| Personalization revenue lift        | 20%                  | McKinsey / Bain                         |
| AI implementation cost ratio        | 3% of revenue        | Allocated cost assumption               |

## 4. Composite ROI Target

The target variable `net_roi` is built in six steps:

1. **CLV** = `annual_spend × 8.5 × purchase_freq_weight`
2. **Retention improvement** = `0.15 × ai_readiness × digital_propensity`
3. **Retention-adjusted CLV** = `CLV × (0.82 + retention_improvement)`
4. **AI revenue uplift** = `retention_adjusted_CLV × 0.20 × ai_readiness`
5. **Retention savings** = `annual_spend × 5 × retention_improvement`
6. **AI cost** = `annual_spend × 0.03 × 8.5`

Final target:

```
net_roi = (ai_revenue_uplift + retention_savings - ai_cost) / ai_cost
```

## 5. AI Readiness Composite

`ai_readiness` is a 0–1 weighted blend of nine signals:

| Signal                     | Weight | Source        |
| -------------------------- | ------ | ------------- |
| ai_usage_freq / 3          | 0.15   | Q13           |
| ai_helpfulness / 4         | 0.15   | Q19           |
| ai_desire / 4              | 0.20   | Q21           |
| ai_balance / 2             | 0.10   | Q22           |
| ai_use_case_count / 7      | 0.10   | Q15           |
| ai_for_decision            | 0.10   | Q16           |
| desired_ai_roles / 5       | 0.10   | Q23           |
| digital_propensity         | 0.05   | Q1 + OSINT    |
| 1 − ai_concern_count / 4   | 0.05   | Q20 (inverse) |

## 6. Models

Three regressors, identical hyperparameters across the two scripts:

| Model             | n_estimators | max_depth | learning_rate | subsample | colsample_bytree |
| ----------------- | ------------ | --------- | ------------- | --------- | ---------------- |
| XGBoost           | 200          | 5         | 0.10          | 0.8       | 0.8              |
| Random Forest     | 200          | 8         | n/a           | n/a       | n/a              |
| Gradient Boosting | 200          | 5         | 0.10          | 0.8       | n/a              |

Evaluation protocol:
- **Synthetic script:** 75/25 train/test split on synthetic, plus 5-fold CV.
- **Real-data script:** train on full synthetic; evaluate transfer on real; then retrain on a 75/25 real split; report CV R² on real.

Seeds fixed at 42 throughout.

## 7. Sensitivity Considerations

Three assumptions should be stress-tested before any client-facing ROI sign-off:
1. Baseline retention rate (82%) — replace with client-specific figure.
2. Personalization revenue lift (20%) — run at 10% and 15%.
3. AI implementation cost ratio (3%) — replace with a loaded TCO from client finance.

## 8. Limitations

- Synthetic training data is deliberately AI-positive; it scaffolds the model but cannot be used alone for inference.
- Real-data n is small (43 raw, ~40 cleaned); CV variance is wide.
- OSINT benchmarks are industry-average; a given client will deviate.
- The model assumes that the allocated 3% AI cost scales linearly across segments; in practice fixed platform costs may dominate early-stage deployments.
