# Survey Instrument Summary

The underlying Qualtrics survey ("Luxury Goods Shopping Experience") contained 23 questions in four blocks. This page documents the question stems and response scales used in the feature pipeline.

## Block A — Demographics

- **Q1.** What is your age? *(single-select: Under 18, 18–24, 25–34, 35–44, 45–54, 55–64, 65+)*
- **Q2.** What is your gender? *(Male, Female, Non-binary, Prefer not to say)*
- **Q3.** What is your annual household income? *(Under $25k → $500k+ bands)*

## Block B — Luxury Behavior

- **Q4.** How often do you purchase luxury goods? *(Multiple times/yr → Never)*
- **Q5.** Which luxury brands have you purchased in the last two years? *(multi-select, 8+ brands)*
- **Q7.** How satisfied are you with your overall luxury shopping experience? *(5-pt Likert)*

## Block C — Authentication

- **Q11.** How much do you trust authentication methods used by luxury retailers? *(Not at all → A lot)*

## Block D — AI Engagement

- **Q13.** How often do you use AI tools when shopping for luxury goods? *(Never → Yes, frequently)*
- **Q15.** In which ways have you used AI for luxury shopping? *(multi-select — recommendations, price comparison, authenticating second-hand items, styling, virtual try-on, customer service chat, brand research)*
- **Q16.** Have you used AI as a personal assistant for a luxury purchase decision? *(Yes / No)*
- **Q18.** How often do you use AI as a personal assistant generally? *(Never → Frequently)*
- **Q19.** How helpful was AI for your luxury shopping? *(N/A, 1 Not Helpful → 4 Helpful)*
- **Q20.** What concerns, if any, do you have about AI in luxury? *(multi-select)*
- **Q21.** Would you like more AI in luxury shopping? *(No, not at all → Yes, definitely)*
- **Q22.** What is your ideal mix of AI and human assistance in luxury? *(No AI, Mostly human, Balanced)*
- **Q23.** In your opinion, what role should AI play in luxury retail? *(multi-select — personalization, efficiency, authenticity, assist staff, immersive)*

Not every question is used as a feature; some support downstream narrative analysis. The mapping from question to engineered feature is fully documented in `data_dictionary.md`.
