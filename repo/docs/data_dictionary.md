# Data Dictionary

Every field produced by the feature-engineering pipeline. Raw Qualtrics columns Q1–Q23 are carried through untouched (cleaned against a per-column whitelist) and all engineered features are derived from them.

## Raw Survey Columns

| Column | Question Topic                                         | Scale / Values |
| ------ | ------------------------------------------------------ | -------------- |
| Q1     | Age band                                               | Under 18, 18–24, 25–34, 35–44, 45–54, 55–64, 65+ |
| Q2     | Gender                                                 | Male, Female, Non-binary, Prefer not to say |
| Q3     | Household income tier                                  | Under $25k, $25–50k, $50–100k, $100–200k, $200–500k, $500k+ |
| Q4     | Luxury purchase frequency                              | Multiple/yr, Once/yr, Few yrs, Rarely, Never |
| Q5     | Luxury brands purchased (multi-select)                 | Louis Vuitton, Chanel, Gucci, Hermès, Dior, Prada, Rolex, Cartier |
| Q7     | Overall luxury shopping satisfaction                   | 5-pt Likert |
| Q11    | Trust in authentication methods                        | Not at all / A little / A moderate amount / A lot |
| Q13    | Frequency of AI tool use for luxury                    | Never / Rarely / Occasionally / Frequently |
| Q15    | AI use cases adopted (multi-select, 7 items)           | Recommendations, Price compare, Authentication, Styling, Virtual try-on, Chat, Brand research |
| Q16    | AI influenced recent luxury decision?                  | Yes / No |
| Q18    | General AI personal assistant frequency                | Never / Rarely / Occasionally / Frequently |
| Q19    | Perceived helpfulness of AI for luxury                 | N/A, Not helpful, Indifferent, Somewhat helpful, Helpful |
| Q20    | AI concerns (multi-select)                             | Personalization, Privacy, Inaccuracy, Human touch, None |
| Q21    | Desire for more AI in luxury                           | No, not at all → Yes, definitely |
| Q22    | Preferred AI / human mix                               | No AI / Mostly human / Balanced |
| Q23    | Desired AI roles (multi-select)                        | Personalization, Efficiency, Authenticity, Assist staff, Immersive |

## Engineered Features

| Field                    | Type          | Source          | Description                                                       |
| ------------------------ | ------------- | --------------- | ----------------------------------------------------------------- |
| annual_spend             | int $         | Q3 + OSINT      | Income-tier-derived estimated annual luxury spend                 |
| purchase_freq_weight     | float [0–1]   | Q4              | Frequency weight used to scale CLV                                |
| digital_propensity       | float [0–1]   | Q1 + OSINT      | Age-band-derived digital engagement propensity                    |
| satisfaction_score       | int [1–5]     | Q7              | Luxury shopping satisfaction score                                |
| ai_usage_freq            | int [0–3]     | Q13             | Frequency of AI tool use for luxury shopping                      |
| ai_helpfulness           | int [0–4]     | Q19             | Perceived helpfulness of AI tools                                 |
| ai_desire                | int [0–4]     | Q21             | Desire for more AI in luxury shopping                             |
| ai_balance               | int [0–2]     | Q22             | Preferred AI / human service mix                                  |
| auth_trust               | int [0–3]     | Q11             | Trust in authentication methods                                   |
| ai_for_decision          | binary        | Q16             | Whether AI influenced a recent luxury decision                    |
| ai_use_case_count        | int [0–7]     | Q15             | Number of luxury AI use cases adopted                             |
| ai_concern_count         | int [0–4]     | Q20             | Number of AI concerns raised                                      |
| desired_ai_roles         | int [0–5]     | Q23             | Number of roles respondent wants AI to play                       |
| brand_count              | int [0–8]     | Q5              | Number of tracked luxury brands purchased                         |
| is_female                | binary        | Q2              | Gender indicator                                                  |
| ai_assistant_freq        | int [0–3]     | Q18             | Frequency of general AI personal assistant use                    |
| ai_readiness             | float [0–1]   | composite       | Weighted composite of AI engagement signals (see §3.2 of report)  |
| clv                      | float $       | derived         | annual_spend × 8.5 × purchase_freq_weight                         |
| retention_improvement    | float [0–1]   | derived         | Retention uplift attributable to AI                               |
| retention_adjusted_clv   | float $       | derived         | CLV adjusted for post-AI retention                                |
| ai_revenue_uplift        | float $       | derived         | Expected revenue uplift per customer                              |
| retention_savings        | float $       | derived         | Avoided churn cost per customer                                   |
| ai_cost                  | float $       | derived         | Allocated per-customer AI implementation cost                     |
| net_roi                  | float         | target          | Net ROI ratio — modeling target                                   |
