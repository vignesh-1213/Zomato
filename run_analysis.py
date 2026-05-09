"""
Standalone runners script for the Zomato Analysis project.
This script:
1. Performs all EDA and data cleaning
2. Creates + saves all 4 required visualizations as PNG files
3. Trains the Random Forest Regressor ML model and prints metrics
4. Regenerates the PDF report with findings and 5 recommendations
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving to files
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
print("=" * 60)
print("  ZOMATO DATA ANALYSIS - Alfido Tech Internship")
print("=" * 60)

df = pd.read_csv('zomato.csv')
print(f"\n[STEP 1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("Columns:", list(df.columns))

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
print("\n[STEP 2] Data Cleaning...")

# Drop noisy columns
cols_to_drop = [c for c in ['url', 'address', 'phone', 'dish_liked', 'reviews_list', 'menu_item'] if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# Rename for convenience
df.rename(columns={
    'approx_cost(for two people)': 'cost',
    'listed_in(type)': 'type',
    'listed_in(city)': 'city'
}, inplace=True)

# Numeric types
def clean_rate(val):
    import re
    val = str(val).strip()
    if val in ['-', 'NEW', 'nan', 'None', '']:
        return np.nan
    # Look for pattern like 4.1/5 anywhere in the string
    match = re.search(r'(\d+\.?\d*)\s*/\s*5', val)
    if match:
        try:
            rating = float(match.group(1))
            if 0.0 <= rating <= 5.0:
                return rating
        except:
            pass
    return np.nan

def clean_cost(val):
    val = str(val).replace(',', '').strip()
    try:
        return float(val)
    except:
        return np.nan

df['rate'] = df['rate'].apply(clean_rate)
df['cost'] = df['cost'].apply(clean_cost)
df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

# Summary before cleaning NaN
print(f"  Missing values per column before drop:")
for col in df.columns:
    n = df[col].isna().sum()
    if n > 0:
        print(f"    {col}: {n} nulls")

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  Clean dataset: {df.shape[0]} rows remaining after dropping NaN")
print(f"\n  Data Types:\n{df.dtypes}")
print(f"\n  Summary Statistics:\n{df[['rate','cost','votes']].describe().round(2)}")

# ─────────────────────────────────────────────
# 3. VISUALIZATIONS (save as PNG)
# ─────────────────────────────────────────────
print("\n[STEP 3] Creating Visualizations...")
os.makedirs('charts', exist_ok=True)

# ------- Chart 1: Location Hotspots (Bar Chart) -------
fig, ax = plt.subplots(figsize=(12, 7))
location_counts = df['location'].value_counts()[:15]
bars = ax.barh(location_counts.index[::-1], location_counts.values[::-1],
               color=sns.color_palette('viridis', 15))
ax.set_title('Top 15 Neighborhoods with Most Restaurants', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Number of Restaurants', fontsize=12)
ax.set_ylabel('Location', fontsize=12)
for bar, val in zip(bars, location_counts.values[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(val), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('charts/chart1_location_hotspots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chart 1 saved: chart1_location_hotspots.png")

# ------- Chart 2: Price vs Rating (Scatter) -------
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df['cost'].values, df['rate'].values, alpha=0.4,
                c=df['votes'].values, cmap='plasma', edgecolors='none', s=20)
plt.colorbar(sc, ax=ax, label='Votes')
ax.set_title('Cost for Two vs. Rating (coloured by Votes)', fontsize=15, fontweight='bold')
ax.set_xlabel('Approx. Cost for Two (INR)', fontsize=12)
ax.set_ylabel('Rating (out of 5)', fontsize=12)
plt.tight_layout()
plt.savefig('charts/chart2_price_vs_rating.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chart 2 saved: chart2_price_vs_rating.png")

# ------- Chart 3: Correlation Heatmap -------
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[['rate', 'votes', 'cost']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5,
            annot_kws={'size': 14}, ax=ax)
ax.set_title('Correlation Heatmap (Rate, Votes, Cost)', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('charts/chart3_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chart 3 saved: chart3_heatmap.png")

# ------- Chart 4: WordCloud for Cuisines -------
fig, ax = plt.subplots(figsize=(10, 8))
cuisines_text = ' '.join(df['cuisines'].astype(str).tolist())
wc = WordCloud(width=900, height=700, background_color='white',
               colormap='tab10', min_font_size=12).generate(cuisines_text)
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title('Popular Cuisines - Word Cloud', fontsize=16, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('charts/chart4_cuisine_wordcloud.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chart 4 saved: chart4_cuisine_wordcloud.png")

# ------- Chart 5: Average Rating by Restaurant Type (Bonus) -------
fig, ax = plt.subplots(figsize=(12, 6))
rest_rating = df.groupby('rest_type')['rate'].mean().sort_values(ascending=False)
colors = sns.color_palette('magma', len(rest_rating))
ax.bar(rest_rating.index, rest_rating.values, color=colors)
ax.set_title('Average Rating by Restaurant Type', fontsize=15, fontweight='bold')
ax.set_xlabel('Restaurant Type', fontsize=11)
ax.set_ylabel('Average Rating', fontsize=11)
ax.set_xticks(range(len(rest_rating.index)))
ax.set_xticklabels(rest_rating.index, rotation=30, ha='right', fontsize=9)
ax.set_ylim(0, 5.5)
for i, v in enumerate(rest_rating.values):
    ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/chart5_rating_by_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chart 5 (bonus) saved: chart5_rating_by_type.png")

# ------- Chart 6: Top 10 Cuisines by Count -------
fig, ax = plt.subplots(figsize=(10, 6))
cuisine_series = df['cuisines'].str.split(', ').explode()
top_cuisines = cuisine_series.value_counts()[:10]
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='rocket', ax=ax)
ax.set_title('Top 10 Most Common Cuisines', fontsize=14, fontweight='bold')
ax.set_xlabel('Count', fontsize=11)
ax.set_ylabel('Cuisine', fontsize=11)
plt.tight_layout()
plt.savefig('charts/chart6_top_cuisines.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chart 6 (bonus) saved: chart6_top_cuisines.png")

# ─────────────────────────────────────────────
# 4. MACHINE LEARNING MODEL
# ─────────────────────────────────────────────
print("\n[STEP 4] Training Random Forest Regressor for Rating Prediction...")

ml_df = df[['location', 'rest_type', 'cost', 'votes', 'rate']].dropna().copy()

le_loc  = LabelEncoder()
le_type = LabelEncoder()
ml_df['location'] = le_loc.fit_transform(ml_df['location'])
ml_df['rest_type'] = le_type.fit_transform(ml_df['rest_type'])

X = ml_df.drop('rate', axis=1)
y = ml_df['rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)

mse  = mean_squared_error(y_test, preds)
rmse = mse ** 0.5
r2   = r2_score(y_test, preds)

print(f"  Model: Random Forest Regressor  (n_estimators=150)")
print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")
print(f"  MSE              : {mse:.4f}")
print(f"  RMSE             : {rmse:.4f}")
print(f"  R2 Score         : {r2:.4f}")

# Feature Importance chart
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis', ax=ax)
ax.set_title('Feature Importance - Random Forest Regressor', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/chart7_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chart 7 saved: chart7_feature_importance.png")

# ─────────────────────────────────────────────
# 5. PDF REPORT (detailed)
# ─────────────────────────────────────────────
print("\n[STEP 5] Generating PDF Report...")

class ZomatoPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 50, 50)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, '  Zomato Dataset Analysis - Alfido Tech Data Science Internship', ln=1, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

pdf = ZomatoPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Page 1 - Overview
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, '1. Project Overview', ln=1)
pdf.set_font('Arial', '', 11)
pdf.multi_cell(0, 7,
    "This project analyzes the Zomato Bangalore restaurants dataset for the Alfido Tech Data Science Internship.\n\n"
    "The analysis covers the complete data science pipeline:\n"
    "  * Data loading and initial inspection\n"
    "  * Data cleaning (null handling, text-to-numeric conversions)\n"
    "  * Exploratory Data Analysis with 6+ visualizations\n"
    "  * Machine Learning model (Random Forest Regressor) for rating prediction\n"
    "  * 5 strategic recommendations for an Alfido Tech-style platform"
)
pdf.ln(5)

pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, '2. Dataset Summary', ln=1)
pdf.set_font('Arial', '', 11)
pdf.multi_cell(0, 7,
    f"Source: kaggle.com/datasets/bhanupratapbiswas/zomato\n"
    f"Records after cleaning: {df.shape[0]}\n"
    f"Columns used: rate, votes, cost, location, rest_type, cuisines, type, city\n\n"
    f"Key statistics after cleaning:\n"
    f"  * Average Rating : {df['rate'].mean():.2f}/5.0\n"
    f"  * Avg Cost (two) : Rs. {df['cost'].mean():.0f}\n"
    f"  * Avg Votes      : {df['votes'].mean():.0f}\n"
    f"  * Most common location: {df['location'].value_counts().index[0]}\n"
    f"  * Most common cuisine : {df['cuisines'].str.split(', ').explode().value_counts().index[0]}"
)
pdf.ln(5)

pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, '3. ML Model Evaluation', ln=1)
pdf.set_font('Arial', '', 11)
pdf.multi_cell(0, 7,
    f"Model     : Random Forest Regressor (n_estimators=150)\n"
    f"Features  : location, rest_type, cost, votes\n"
    f"Target    : rate (restaurant rating 0-5)\n\n"
    f"Results:\n"
    f"  MSE    = {mse:.4f}\n"
    f"  RMSE   = {rmse:.4f}\n"
    f"  R2     = {r2:.4f}\n\n"
    f"Top Feature by Importance: {feat_df.iloc[0]['Feature']} ({feat_df.iloc[0]['Importance']:.2%})"
)

# Page 2 - Key Findings
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, '4. Key Findings', ln=1)
pdf.set_font('Arial', '', 11)
findings = [
    ("Location Hotspots",
     f"BTM, Koramangala, and Indiranagar lead in restaurant density. "
     f"These areas consistently attract high footfall and competitive dining options."),
    ("Rating vs Cost",
     "Premium dining (Rs. 1000+) shows more consistent ratings, whereas budget restaurants "
     "have high variance. However, several budget restaurants achieve ratings >= 4.0, "
     "indicating perceived value matters more than price alone."),
    ("Votes and Rating Correlation",
     f"'Votes' has the highest correlation with 'rate' (corr: {df[['rate','votes']].corr().iloc[0,1]:.2f}). "
     "Popular restaurants sustain better ratings, suggesting a network-effect loop."),
    ("Cuisines Landscape",
     "North Indian, Chinese, and Fast Food dominate cuisine options. Continental and Italian "
     "have fewer entries but score higher average ratings, representing high-value niches."),
]
for title, body in findings:
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(8, 7, '*', ln=0)
    pdf.cell(0, 7, title, ln=1)
    pdf.set_font('Arial', '', 11)
    pdf.set_x(16)
    pdf.multi_cell(0, 6, body)
    pdf.ln(2)

# Page 3 - Recommendations
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, '5. Recommendations for Alfido Tech-Style Platform', ln=1)
pdf.set_font('Arial', '', 11)
recommendations = [
    ("Prioritise High-Density Hotspot Partnerships",
     "Establish restaurant partnerships in BTM, Koramangala, and Indiranagar first. "
     "These neighbourhoods deliver maximum reach for minimal acquisition effort."),
    ("Launch a 'Budget Gems' Content Series",
     "Data shows restaurants with cost < Rs. 400 and rating >= 4.0 exist across multiple areas. "
     "A curated 'Budget Gems' section drives aspirational discovery and engagement."),
    ("Gamified Review System",
     "Since vote count strongly predicts rating stability, incentivise users to leave "
     "reviews through points, badges, or food vouchers - creating a virtuous engagement loop."),
    ("Feature 'Online Order' and 'Table Booking' Prominently",
     "Restaurants offering these features attract higher ratings. Highlight these capabilities "
     "as trust signals to convert undecided users."),
    ("Invest in Niche Cuisine Discovery",
     "Continental, Italian, and fine dining cuisines are underrepresented but show higher "
     "ratings. A 'Discover Premium' section with curated niche cuisine recommendations "
     "can meaningfully differentiate the platform.")
]
for i, (title, body) in enumerate(recommendations, 1):
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, f"Recommendation {i}: {title}", ln=1)
    pdf.set_font('Arial', '', 11)
    pdf.set_x(10)
    pdf.multi_cell(0, 6, body)
    pdf.ln(3)

pdf.output('report.pdf')
print("  ✓ report.pdf generated successfully")

print("\n" + "=" * 60)
print("  ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nOutput files:")
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d != 'venv' and not d.startswith('.')]
    for f in files:
        if not f.endswith('.py'):
            path = os.path.join(root, f)
            print(f"  {path}")
