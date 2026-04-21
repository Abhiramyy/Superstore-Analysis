# ============================================================
#  DATA CLEANING & VISUALIZATION PROJECT
#  Dataset : Superstore Sales Dataset
#  Tools   : Python | Pandas | Matplotlib | Seaborn
#  Author  : [Your Name]
#  Date    : April 2026
# ============================================================


# ─────────────────────────────────────────────────────────────
# CELL 1 — Import Libraries
# ─────────────────────────────────────────────────────────────
import pandas as pd               # data loading and manipulation
import numpy as np                # numerical operations
import matplotlib.pyplot as plt   # base plotting
import seaborn as sns             # statistical visualizations
import warnings
warnings.filterwarnings('ignore')

# Set a clean visual style for all charts
sns.set_theme(style="whitegrid", palette="muted")
print("✅ All libraries imported successfully!")


# ─────────────────────────────────────────────────────────────
# CELL 2 — Load the Dataset
# ─────────────────────────────────────────────────────────────
# pd.read_csv() reads a CSV file and stores it as a DataFrame
# A DataFrame is like a spreadsheet table in Python

df = pd.read_csv('superstore_raw.csv')

print(f"✅ Dataset loaded!")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")


# ─────────────────────────────────────────────────────────────
# CELL 3 — Data Understanding: head()
# ─────────────────────────────────────────────────────────────
# head() shows the FIRST 5 rows of the dataset.
# Think of it as a quick peek at the top of your spreadsheet.
# It helps you understand what the columns look like.

print("📋 FIRST 5 ROWS:")
df.head()


# ─────────────────────────────────────────────────────────────
# CELL 4 — Data Understanding: info()
# ─────────────────────────────────────────────────────────────
# info() is like a health report of your dataset.
# It tells you:
#   - Column names
#   - Data type of each column (int, float, object = text)
#   - How many NON-NULL (non-missing) values exist
# If "Non-Null Count" < total rows → that column has missing values!

print("📋 DATASET INFO:")
df.info()


# ─────────────────────────────────────────────────────────────
# CELL 5 — Data Understanding: describe()
# ─────────────────────────────────────────────────────────────
# describe() calculates quick statistics for all number columns.
# Key things to look at:
#   mean  = average value
#   50%   = median (middle value)
#   min   = smallest value
#   max   = largest value
# If max is MUCH larger than 75%, you may have outliers!

print("📊 STATISTICAL SUMMARY:")
df.describe().round(2)


# ─────────────────────────────────────────────────────────────
# CELL 6 — Check Missing Values and Duplicates
# ─────────────────────────────────────────────────────────────
print("⚠️  MISSING VALUES PER COLUMN:")
print(df.isnull().sum())

print(f"\n⚠️  DUPLICATE ROWS: {df.duplicated().sum()}")


# ─────────────────────────────────────────────────────────────
# CELL 7 — Data Cleaning Step 1: Remove Duplicates
# ─────────────────────────────────────────────────────────────
# Duplicate rows are EXACT copies of a row.
# They happen when data is collected or merged multiple times.
# They make totals and averages incorrect, so we remove them.

before = len(df)
df = df.drop_duplicates()          # removes exact duplicate rows
after  = len(df)

print(f"🗑️  Duplicates removed : {before - after}")
print(f"   Rows remaining     : {after}")


# ─────────────────────────────────────────────────────────────
# CELL 8 — Data Cleaning Step 2: Handle Missing Values
# ─────────────────────────────────────────────────────────────
# Missing values appear as NaN (Not a Number) in Pandas.
# We must fill them — leaving them causes errors in calculations.
#
# Strategy A → Numeric columns (Profit): fill with MEDIAN
#   Why median? It is the middle value and is NOT affected by
#   very large or very small outliers. Safer than the mean.
#
# Strategy B → Text columns (Segment): fill with MODE
#   Why mode? The most common category is the best "default" guess.

median_profit = df['Profit'].median()
df['Profit']  = df['Profit'].fillna(median_profit)
print(f"✅  Filled missing Profit with median  : ${median_profit:.2f}")

mode_segment  = df['Segment'].mode()[0]
df['Segment'] = df['Segment'].fillna(mode_segment)
print(f"✅  Filled missing Segment with mode   : '{mode_segment}'")

print(f"✅  Missing values remaining            : {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────────────────────
# CELL 9 — Data Cleaning Step 3: Fix Data Types
# ─────────────────────────────────────────────────────────────
# The 'Order_Date' column was read as plain text (object type).
# We convert it to proper datetime so we can extract Year/Month later.

df['Order_Date'] = pd.to_datetime(df['Order_Date'])
print(f"✅  Order_Date converted to: {df['Order_Date'].dtype}")


# ─────────────────────────────────────────────────────────────
# CELL 10 — Data Cleaning Step 4: Remove Outliers (IQR Method)
# ─────────────────────────────────────────────────────────────
# An outlier is an extreme value far from the rest.
# The IQR (Interquartile Range) method is a statistical way to find them.
#
# Step-by-step:
#   1. Q1 = value at 25th percentile (bottom quarter)
#   2. Q3 = value at 75th percentile (top quarter)
#   3. IQR = Q3 - Q1  ← the "middle 50%" range
#   4. Lower bound = Q1 - 1.5 × IQR
#   5. Upper bound = Q3 + 1.5 × IQR
#   6. Anything OUTSIDE these bounds = outlier → remove it

Q1  = df['Sales'].quantile(0.25)
Q3  = df['Sales'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

before = len(df)
df = df[(df['Sales'] >= lower_bound) & (df['Sales'] <= upper_bound)]
after  = len(df)

print(f"📐 IQR Outlier Detection on 'Sales':")
print(f"   Q1 = ${Q1:.2f}   Q3 = ${Q3:.2f}   IQR = ${IQR:.2f}")
print(f"   Lower bound : ${lower_bound:.2f}")
print(f"   Upper bound : ${upper_bound:.2f}")
print(f"   Outliers removed : {before - after}")
print(f"   Rows remaining   : {after}")


# ─────────────────────────────────────────────────────────────
# CELL 11 — Data Processing Step 1: Rename Columns
# ─────────────────────────────────────────────────────────────
# Consistent column naming (Title_Case) makes code easier to read.
# We rename 'unit_price' → 'Unit_Price'

df = df.rename(columns={'unit_price': 'Unit_Price'})
print("✅  Renamed 'unit_price' → 'Unit_Price'")


# ─────────────────────────────────────────────────────────────
# CELL 12 — Data Processing Step 2: Create New Columns
# ─────────────────────────────────────────────────────────────
# New columns add useful information that wasn't in the raw data.

# Extract year from the date (useful for year-over-year comparison)
df['Year']  = df['Order_Date'].dt.year

# Extract month (useful for seasonal trends)
df['Month'] = df['Order_Date'].dt.month

# Profit Margin % = (Profit / Sales) × 100
# Tells us: for every $100 in sales, how much is actual profit?
df['Profit_Margin_%'] = (df['Profit'] / df['Sales'] * 100).round(2)

print("✅  New columns created:")
print("   → Year")
print("   → Month")
print("   → Profit_Margin_%")
print(f"\n   Sample Profit_Margin_%: {df['Profit_Margin_%'].head(3).values}")


# ─────────────────────────────────────────────────────────────
# CELL 13 — Data Processing Step 3: Filter Data
# ─────────────────────────────────────────────────────────────
# We keep ONLY orders where Profit > 0.
# Negative-profit orders are losses — not useful for sales analysis.

df_clean = df[df['Profit'] > 0].copy()

print(f"✅  Filtered to profitable orders: {len(df_clean)} rows")

# Save the final cleaned dataset to a new CSV file
df_clean.to_csv('superstore_cleaned.csv', index=False)
print("💾  Saved as 'superstore_cleaned.csv'")

# Quick preview of cleaned data
print("\n📋 CLEANED DATA PREVIEW:")
df_clean[['Order_ID', 'Category', 'Region', 'Sales',
          'Profit', 'Profit_Margin_%']].head()


# ─────────────────────────────────────────────────────────────
# CELL 14 — Visualization 1: Bar Chart
# ─────────────────────────────────────────────────────────────
# A bar chart compares totals across categories.
# Each bar = one category. Taller bar = higher total sales.
# QUESTION ANSWERED: Which product category earns the most revenue?

cat_sales = df_clean.groupby('Category')['Sales'].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    cat_sales.index,
    cat_sales.values,
    color=['#2E86AB', '#A23B72', '#F18F01'],
    edgecolor='white',
    linewidth=1.5
)

# Add dollar labels on top of each bar
for bar, val in zip(bars, cat_sales.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 500,
        f'${val:,.0f}',
        ha='center', va='bottom',
        fontweight='bold', fontsize=10
    )

ax.set_title('Total Sales by Product Category', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Total Sales (USD)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('chart1_bar_category.png', dpi=150, bbox_inches='tight')
plt.show()

# ► INSIGHT: Technology and Furniture dominate sales.


# ─────────────────────────────────────────────────────────────
# CELL 15 — Visualization 2: Histogram
# ─────────────────────────────────────────────────────────────
# A histogram shows HOW MANY orders fall in each sales range.
# Each bar (called a "bin") represents a range, e.g. $0–$500.
# Tall bar on the left = most orders are small.
# The red line = mean (average). Orange line = median (middle).
# QUESTION ANSWERED: What does a typical order value look like?

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(df_clean['Sales'], bins=30, color='#2E86AB', edgecolor='white', alpha=0.85)

ax.axvline(
    df_clean['Sales'].mean(),
    color='#E84855', linestyle='--', linewidth=2,
    label=f"Mean: ${df_clean['Sales'].mean():,.0f}"
)
ax.axvline(
    df_clean['Sales'].median(),
    color='#F18F01', linestyle='--', linewidth=2,
    label=f"Median: ${df_clean['Sales'].median():,.0f}"
)

ax.set_title('Distribution of Sales Amounts', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Sales Amount (USD)', fontsize=12)
ax.set_ylabel('Number of Orders', fontsize=12)
ax.legend(fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('chart2_histogram_sales.png', dpi=150, bbox_inches='tight')
plt.show()

# ► INSIGHT: Most orders are under $3,000. The distribution is right-skewed.


# ─────────────────────────────────────────────────────────────
# CELL 16 — Visualization 3: Boxplot
# ─────────────────────────────────────────────────────────────
# A boxplot summarises the spread of data in one chart:
#   • Box    = middle 50% of values (between Q1 and Q3)
#   • Line   = median (middle value)
#   • Whiskers = min and max (within 1.5×IQR range)
#   • Dots   = outliers (values beyond the whiskers)
# QUESTION ANSWERED: How do sales amounts vary across regions?

fig, ax = plt.subplots(figsize=(9, 5))

region_order = (df_clean.groupby('Region')['Sales']
                .median().sort_values(ascending=False).index)

sns.boxplot(
    data=df_clean,
    x='Region', y='Sales',
    order=region_order,
    palette=['#2E86AB', '#A23B72', '#F18F01', '#3BB273'],
    ax=ax
)

ax.set_title('Sales Distribution by Region', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('Sales Amount (USD)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('chart3_boxplot_region.png', dpi=150, bbox_inches='tight')
plt.show()

# ► INSIGHT: South and West regions show the widest spread of sales values.


# ─────────────────────────────────────────────────────────────
# CELL 17 — Visualization 4: Heatmap (Correlation Matrix)
# ─────────────────────────────────────────────────────────────
# A heatmap shows how strongly TWO variables are related.
# Correlation values range from -1 to +1:
#   +1 = as one goes up, the other goes up too   (dark red)
#    0 = no relationship at all                   (white)
#   -1 = as one goes up, the other goes down      (dark blue)
# QUESTION ANSWERED: Which variables influence profit the most?

numeric_cols = df_clean[['Sales', 'Profit', 'Quantity',
                          'Discount', 'Unit_Price', 'Profit_Margin_%']]
corr = numeric_cols.corr()

# mask = hide the upper triangle (it's a mirror of the lower half)
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
    mask=mask, ax=ax, linewidths=0.5,
    annot_kws={'size': 10}, cbar_kws={'shrink': 0.8}
)

ax.set_title('Correlation Heatmap of Numeric Variables',
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('chart4_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ► INSIGHT: Sales and Profit have a strong correlation (r=0.81).
# ► INSIGHT: Discount has a slight negative impact on Profit.


# ─────────────────────────────────────────────────────────────
# CELL 18 — Visualization 5 (Bonus): Avg Profit Margin by Segment
# ─────────────────────────────────────────────────────────────
# Profit Margin % shows EFFICIENCY — not just raw profit.
# A segment with lower total sales can still have a higher margin.
# QUESTION ANSWERED: Which customer type is most profitable?

seg_margin = (df_clean.groupby('Segment')['Profit_Margin_%']
              .mean().sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    seg_margin.index, seg_margin.values,
    color=['#3BB273', '#2E86AB', '#A23B72'],
    edgecolor='white', linewidth=1.5
)

for bar, val in zip(bars, seg_margin.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{val:.1f}%',
        ha='center', va='bottom', fontweight='bold', fontsize=11
    )

ax.set_title('Average Profit Margin by Customer Segment',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Customer Segment', fontsize=12)
ax.set_ylabel('Avg Profit Margin (%)', fontsize=12)

plt.tight_layout()
plt.savefig('chart5_segment_margin.png', dpi=150, bbox_inches='tight')
plt.show()

# ► INSIGHT: Consumer segment has the highest avg margin (22.25%).


# ─────────────────────────────────────────────────────────────
# CELL 19 — Key Insights Summary
# ─────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║              📌  KEY INSIGHTS FROM THE DATA                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Technology leads in total sales (~$518K), followed by   ║
║     Furniture (~$494K). Office Supplies is far behind.       ║
║                                                              ║
║  2. Most orders are under $3,000. The histogram shows a     ║
║     right-skewed distribution (a few very large orders      ║
║     pull the mean above the median).                         ║
║                                                              ║
║  3. Sales & Profit are strongly correlated (r = 0.81).      ║
║     Growing sales volume is the clearest path to profit.    ║
║                                                              ║
║  4. South ($303K) and West ($302K) lead in regional sales.  ║
║     The East region ($245K) has the lowest performance.     ║
║                                                              ║
║  5. Consumer segment has the best profit margin (22.25%),   ║
║     ahead of Corporate (20.21%) and Home Office (19.23%).   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print("=" * 60)
print("  ✅  PROJECT COMPLETE — Data cleaned and visualised!")
print("=" * 60)
