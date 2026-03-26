# 🎯 Smart Text Input Feature - Documentation

## Overview

The Home Input page now includes two powerful methods for selecting customer profiles and pain points:

1. **📋 Structured Input** - Traditional dropdown selections (original method)
2. **✍️ Smart Text Input** - Natural language description with AI-powered category detection (NEW!)

## How It Works

### Smart Text Input Method

Users can now describe their customers and business challenges in their own words, and the system will automatically identify which predefined categories they fall into.

#### Example Usage:

**User Types:**
```
"We focus on ultra-high-net-worth VIP clients who make expensive purchases 
and are very engaged with our brand community. They are frequent luxury 
buyers looking for exclusive limited edition items."
```

**System Detects:**
- ✓ Ultra-high-net-worth VIC/VIP clients
- ✓ High-spending (High AOV) customers
- ✓ Luxury limited/rare item buyers
- ✓ High-frequency repurchase customers
- ✓ High private domain/community engagement customers

---

## Features

### 1. Intelligent Keyword Matching

The `match_features_from_text()` function uses:
- **Keyword Matching**: Recognizes domain-specific terminology
- **Multi-word Phrases**: Better matching for complex terms
- **Score-Based Ranking**: Returns matches sorted by relevance
- **Comprehensive Synonyms**: Understands variations and similar terms

### 2. Two Input Modes

#### Tab 1: Structured Input
- Use when you know exactly which category you need
- Click dropdowns and select from predefined options
- Most straightforward and fastest method
- Perfect for technical users

#### Tab 2: Smart Text Input
- Describe your situation in natural language
- Get automatic suggestions
- Great for stakeholders who aren't familiar with taxonomy
- More flexible and conversational approach

### 3. Interactive Detection

As you type in the text areas:
- Live detection happens as you write
- Matched categories appear below the text area
- Show top 3 matches with option to see more
- Helpful messages if no matches found yet

### 4. Easy Confirmation

- Click "✅ Use Detected Categories" button to confirm auto-detected matches
- Session remembers which method you used
- Can switch between methods freely
- All selections transfer to analysis

---

## Keyword Categories & Mappings

### Customer Features

#### 1. High-spending (High AOV) customers
**Keywords:** high, spending, aov, expensive, premium, luxury, wealth, rich, affluent, high-value, big spenders, vip, high-ticket

**Example Input:** 
> "Our customers are wealthy and make expensive purchases with high average order value"

#### 2. High-frequency repurchase customers
**Keywords:** frequent, repeat, loyalty, regular, purchase, recurrent, habitual, steady, consistent, recurring, loyal customer, high volume

**Example Input:**
> "We have a very loyal customer base with consistent repeat purchases"

#### 3. Luxury limited/rare item buyers
**Keywords:** limited, edition, rare, exclusive, collector, collectible, premium, scarce, sought-after, limited edition

**Example Input:**
> "Our customers are collectors who seek exclusive and hard-to-find items"

#### 4. Omnichannel (online + offline) shoppers
**Keywords:** omnichannel, online, offline, channel, digital, store, web, mobile, seamless, cross-channel, multi-channel

**Example Input:**
> "Our customers shop both online and in our physical stores"

#### 5. High private domain/community engagement customers
**Keywords:** community, engagement, private, domain, social, network, group, member, social network, brand community

**Example Input:**
> "We have an active brand community with high member engagement"

#### 6. Ultra-high-net-worth VIC/VIP clients
**Keywords:** ultra, high-net-worth, hnw, vip, vic, premium member, elite, exclusive, top tier, c-level, executive, wealthy

**Example Input:**
> "We serve ultra-high-net-worth individuals and executive clients"

#### 7. At-risk churn customers
**Keywords:** churn, risk, at-risk, lapsed, inactive, lost, dormant, decline, no purchase, long time

**Example Input:**
> "We need to reduce customer churn among our inactive customers"

#### 8. Leather goods/jewelry/watch preference buyers
**Keywords:** leather, jewelry, watch, accessories, luxury goods, premium, apparel, designer, fashion

**Example Input:**
> "Our store specializes in luxury leather, jewelry, and watches"

---

### Pain Points

#### 1. Insufficient personalized service for high-value clients
**Keywords:** personalization, personalized, service, customization, tailor, individual, premium service, one-on-one, bespoke

**Example Input:**
> "We struggle to provide personalized services to our VIP clients"

#### 2. Chaotic limited-edition product allocation & appointment
**Keywords:** allocation, appointment, limited, chaos, chaotic, confusion, scheduling, booking, distribution

**Example Input:**
> "Managing limited edition drops and appointments is chaos"

#### 3. Low customer repurchase & retention rate
**Keywords:** retention, repurchase, repeat, coming back, loyalty, reduce churn, keep, maintain

**Example Input:**
> "Our retention rates are declining and customers aren't coming back"

#### 4. High customer service operational costs
**Keywords:** cost, operational, expensive, labor, efficiency, overhead, manual, workload, labor intensive

**Example Input:**
> "Customer service is expensive and labor-intensive"

#### 5. Inconsistent omnichannel customer experience
**Keywords:** inconsistent, omnichannel, experience, channel, touchpoint, unified, seamless, integration

**Example Input:**
> "Our online and offline customer experiences are inconsistent"

#### 6. Unfocused and inefficient luxury marketing
**Keywords:** marketing, unfocused, inefficient, targeting, campaign, promotion, strategy, effectiveness

**Example Input:**
> "Our marketing efforts are inefficient and unfocused"

---

## Technical Implementation

### Core Function: `match_features_from_text()`

```python
match_features_from_text(user_text, feature_list) → list[str]
```

**Algorithm:**
1. Convert user text to lowercase for comparison
2. Initialize score dictionary for each feature
3. For each feature:
   - Get associated keywords from mapping
   - Count keyword occurrences (boost for multi-word phrases)
   - Boost score if full feature name appears
4. Sort by score descending
5. Return features with score > 0

**Scoring:**
- Single keyword: +1 point
- Multi-word phrase: +1.5 points
- Feature name match: +2 points

---

## Session State Management

The feature uses Streamlit session state to:

```python
st.session_state.auto_selected_features  # Stores auto-detected features
st.session_state.auto_selected_pains     # Stores auto-detected pain points
st.session_state.input_method             # Tracks which tab was used ('structured' or 'smart')
```

This allows:
- Persistence between reruns
- Switching between tabs without losing state
- Remembering user's preferred method
- Smooth data flow to analysis

---

## Usage Examples

### Scenario 1: Premium Customer Experience Issue

**User Input:**
```
We need to improve service for our ultra-wealthy clients. We have 
many high-net-worth individuals making premium purchases, but our 
service isn't personalized enough. Our VIP clients need more 
attention and customization.
```

**Auto-Detected:**
- Customer Feature: Ultra-high-net-worth VIC/VIP clients
- Pain Point: Insufficient personalized service for high-value clients

---

### Scenario 2: Multi-Channel Operations

**User Input:**
```
We're having trouble managing our online and offline shopping 
experiences. Customers are frustrated with the inconsistent 
experience between our web platform and physical stores. We need 
seamless omnichannel support.
```

**Auto-Detected:**
- Customer Feature: Omnichannel (online + offline) shoppers
- Pain Point: Inconsistent omnichannel customer experience

---

### Scenario 3: Community Engagement

**User Input:**
```
Our loyal community members are very engaged with our brand. 
We have a strong private community with regular repeat purchases 
and high engagement levels, but we struggle to keep them coming 
back and personalize their experience.
```

**Auto-Detected:**
- Customer Feature: High private domain/community engagement customers
- Customer Feature: High-frequency repurchase customers
- Pain Point: Low customer repurchase & retention rate

---

## Best Practices

### For Maximum Accuracy:

1. **Be Specific**: Use terminology from your industry
   - ✅ "Ultra-high-net-worth clients making luxury purchases"
   - ❌ "Rich customers"

2. **Combine Multiple Aspects**: Describe multiple characteristics
   - ✅ "Collectors looking for exclusive limited edition items"
   - ❌ "Collectors"

3. **Use Domain Terms**: Include keywords the system recognizes
   - ✅ "Omnichannel shoppers shopping online and offline"
   - ❌ "Multi-location shoppers"

4. **Be Conversational**: Natural language descriptions work best
   - ✅ "We have trouble managing appointments for limited drops"
   - ❌ "Appointments limited"

5. **Multi-line Entries**: More detail = Better matching
   - ✅ 2-3 sentences describing situation
   - ❌ Single word or very brief phrase

---

## Limitations & Notes

### Current Limitations:

1. **Fixed Categories**: Matching works with predefined categories only
2. **English Only**: Keywords are in English
3. **Keyword-Based**: No machine learning/LLM integration currently
4. **Score Threshold**: Only returns features with score > 0

### Future Enhancements:

- [ ] Multi-language support
- [ ] Semantic similarity using embeddings
- [ ] Custom category definition
- [ ] Machine learning model for better matching
- [ ] Confidence scores for each match
- [ ] Feedback mechanism to improve matching

---

## Troubleshooting

### No matches detected?

**Problem:** "No customer features matched yet"

**Solutions:**
1. Use more specific industry terminology
2. Spell check your keywords
3. Try alternative terms (e.g., "VIP" instead of "Important customers")
4. Add more context and description
5. Switch to Structured Input for exact selection

### Wrong categories detected?

**Problem:** Got irrelevant matches

**Solutions:**
1. Review and remove irrelevant text
2. Be more specific about which aspect you're describing
3. Avoid mixing multiple unrelated topics
4. Try Structured Input if you need precise control

### Want to manually adjust?

**Solution:**
1. Copy the auto-detected categories
2. Take note of them
3. Switch to Structured Input tab
4. Manually select exactly what you want
5. Any selection method works for analysis

---

## Architecture

```
User Input (Text Area)
        ↓
match_features_from_text()
        ↓
Keyword Matching & Scoring
        ↓
Filtered & Ranked Results
        ↓
Display Top Matches
        ↓
User Confirmation (Button Click)
        ↓
Session State Update
        ↓
Run Analysis
```

---

## Code Reference

### Matching Function Location
File: `app.py`
Function: `match_features_from_text(user_text, feature_list)`
Lines: ~273-375

### UI Implementation Location
File: `app.py`
Section: "Home Input page" → "TAB 2: SMART TEXT INPUT"
Lines: ~943-1043

---

## Support & Feedback

If you encounter issues or have suggestions:
1. Check the troubleshooting section above
2. Review usage examples for your specific case
3. Provide detailed input text for debugging
4. Test with both Structured and Smart methods

---

**Feature Version**: 1.0
**Last Updated**: March 25, 2026
**Status**: ✅ Production Ready
