# 🎨 UI Styling Guide & Best Practices

## Dashboard Design System

### Color Palette

#### Primary Colors
- **Luxury Red**: `#C41E3A` - Main brand color for CTAs and highlights
- **Dark Charcoal**: `#1f1f1f` - Secondary text and accents
- **Luxury Gold**: `#FFB81C` - Accent touches

#### Semantic Colors
- **Success**: `#10B981` - Positive metrics, completed items
- **Info**: `#3B82F6` - Informational elements, insights
- **Warning**: `#F59E0B` - Warnings, attention needed
- **Error**: `#EF4444` - Problems, required actions
- **Light**: `#F9FAFB` - Light backgrounds, cards
- **Dark**: `#111827` - Dark text, emphasis

### Typography

#### Font Sizes
- **Headers (h1)**: 2.2rem (35px), Bold, Color: `#111827`
- **Subheaders (h2)**: 1.8rem (29px), Bold
- **Section Headers (h3)**: 1.3rem (21px), Bold
- **Body Text**: 1rem (16px), Regular, Color: `#374151`
- **Small Text**: 0.875rem (14px), Regular, Color: `#6B7280`
- **Captions**: 0.75rem (12px), Regular, Color: `#9CA3AF`

#### Font Weights
- **Headers**: 700 (Bold)
- **Section titles**: 600 (Semi-bold)
- **Default text**: 400 (Regular)

### Spacing

#### Margins & Padding
- **Container Padding**: 1.5rem (24px)
- **Section Spacing**: 2rem (32px)
- **Card Padding**: 1.5rem (24px)
- **Element Gap**: 0.5rem - 1rem (8-16px)
- **Button Padding**: 0.75rem 1.5rem (12px 24px)

#### Border Radius
- **Cards & Containers**: 12px
- **Buttons**: 8px
- **Inputs**: 8px
- **Small Elements**: 4px

### Shadow & Depth

#### Card Shadows
```css
box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
```

#### Button Shadows
```css
box-shadow: 0 2px 8px rgba(196, 30, 58, 0.3);
```

#### Hover Effects
```css
box-shadow: 0 4px 12px rgba(196, 30, 58, 0.4);
transform: translateY(-2px);
```

## Component Styles

### Buttons

#### Primary Button
- Background: Linear gradient from `#C41E3A` to `#9B1733`
- Color: White
- Padding: 0.75rem 1.5rem
- Border Radius: 8px
- Font Weight: 600
- Hover: Lift effect + shadow increase

```python
st.button("🚀 Run Analysis", type="primary", use_container_width=True)
```

#### Secondary Button
- Background: `#F3F4F6`
- Color: `#111827`
- Border: 1px solid `#E5E7EB`
- Hover: Background lightens

### Metric Cards

#### Standard Metric Card
```html
<div class="metric-card">
    <div style="font-size: 0.85rem; color: #666;">Label</div>
    <div style="font-size: 1.8rem; font-weight: bold; color: #111827;">
        📊 Value
    </div>
    <div style="font-size: 0.8rem; color: #999;">Subtext</div>
</div>
```

Features:
- Left border: 4px solid `#C41E3A`
- Gradient background: `#F3F4F6` to `#E5E7EB`
- Smooth box shadow
- Proper padding and spacing

### Status Cards

#### Success Card (Green)
- Background Gradient: `#F0FDF4` to `#DCFCE7`
- Border: 1px solid `#86EFAC`

#### Warning Card (Yellow)
- Background Gradient: `#FEF2F2` to `#FEE2E2`
- Border: 1px solid `#FECACA`

#### Info Card (Blue)
- Background Gradient: `#F0F9FF` to `#E0F2FE`
- Border: 1px solid `#7DD3FC`

### Expanders

#### Default Expander
- Border: 1px solid `#E5E7EB`
- Border Radius: 8px
- Header Background: `#F9FAFB`
- Hover: Slight background change

#### Active Expander
- Header Border Bottom: 2px solid `#C41E3A`
- Highlighted visual state

### Input Fields

#### Text Input / Number Input
- Border: 2px solid `#E5E7EB`
- Border Radius: 8px
- Padding: 0.75rem
- Focus: Border color changes to `#C41E3A`
- Background: White

#### Select Dropdown
- Border: 2px solid `#E5E7EB`
- Border Radius: 8px
- Padding: 0.75rem
- Background: White with dropdown arrow

### Data Tables

#### Default Table
- Font Size: 0.9rem
- Border Radius: 8px
- Row Alternating: Light backgrounds
- Header: Bold with gray background

#### Heatmap
- Color Scale: RdYlGn (Red → Yellow → Green)
- Cell Values: Rounded to 2 decimals
- Hover: Shows exact value

## Progress Indicator

### Progress Step Structure
```html
<div class="progress-step [active|completed]">
    <span>✓ or 📊</span>
    <div>
        <strong>Step Title</strong>
        <div>Description</div>
    </div>
</div>
```

### Progress States
- **Completed**: Green border, background `#F0FDF4`
- **Active**: Red border, background `#FEF2F2`
- **Future**: Gray border, background `#F9FAFB`

## Usage Patterns

### Creating a Styled Section
```python
st.markdown("### 📌 Section Title")
st.divider()

# Content here

st.divider()
```

### Creating Metric Cards
```python
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Label", "$1,234,567", delta="trend")
with col2:
    st.metric("Label", "8.5x", delta_color="off")
with col3:
    st.metric("Label", "95%", delta="good")
```

### Creating Expandable Sections
```python
with st.expander("📚 Learn More", expanded=False):
    st.markdown("Details here...")
    st.code("code example")
```

### Creating Column Layouts
```python
col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    st.write("Large column")
with col2:
    st.write("Medium column")
with col3:
    st.write("Thin column")
```

## Responsive Design

### Screen Sizes
- **Desktop**: Full width layouts with 2-3 columns
- **Tablet**: 2-column layouts
- **Mobile**: Single column layouts (current Streamlit limitation)

### Column Ratios
- **Balanced**: 1, 1, 1
- **Main + Sidebar**: 2, 1
- **Main + Features**: 1.5, 1, 0.5
- **Custom**: Use numerical ratios

## Accessibility

### Color Contrast
- Body text vs background: Minimum 7:1 (AAA)
- Headings: 4.5:1 (AA standard)
- Links: Underline for clarity

### Icons
- Always pair with text labels
- Use emoji for quick recognition
- Maintain consistent emoji set

### Text
- Clear hierarchy
- Descriptive link text
- Spell out abbreviations on first use

## Best Practices

### DO ✅
- Use consistent color scheme throughout
- Keep borders and shadows subtle
- Maintain visual hierarchy
- Use icons to enhance understanding
- Group related controls
- Provide helpful tooltips
- Show loading states for long operations
- Use dividers to separate sections

### DON'T ❌
- Mix too many colors
- Use heavy shadows
- Clutter with too many elements
- Use undefined colors
- Forget about mobile view
- Ignore accessibility
- Use all caps for body text
- Overcomplicate layouts

## Common Patterns

### Status Display Pattern
```python
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Status", "✓ Complete")
with col2:
    st.caption("Last updated 2 minutes ago")
```

### Input Group Pattern
```python
st.markdown("#### Configuration")
col1, col2 = st.columns(2)
with col1:
    param1 = st.slider("Parameter 1", 0, 100)
with col2:
    param2 = st.slider("Parameter 2", 0, 100)
```

### Results Display Pattern
```python
st.markdown("#### Results")
col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]
for i, (label, value) in enumerate(results.items()):
    with cols[i]:
        st.metric(label, value)
```

## Theme Customization

To change the theme colors globally, update the CSS variables in the `<style>` section:

```css
:root {
    --primary: #C41E3A;
    --secondary: #1f1f1f;
    --accent: #FFB81C;
    --success: #10B981;
    --warning: #F59E0B;
    --error: #EF4444;
    --light: #F9FAFB;
    --dark: #111827;
}
```

## Testing Checklist

Before deploying changes:
- [ ] Verify color contrast ratios
- [ ] Test on multiple browsers
- [ ] Check mobile responsiveness
- [ ] Validate accessibility
- [ ] Test all interactive elements
- [ ] Review typography scaling
- [ ] Verify animations/transitions are smooth
- [ ] Check loading states
- [ ] Test error messaging
- [ ] Review spacing and alignment

---

**Design System Version**: 2.0
**Last Updated**: March 25, 2026
**Maintained by**: Design Team
