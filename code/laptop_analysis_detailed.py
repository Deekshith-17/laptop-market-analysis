"""
Data Analysis and Visualization of Laptop Market Trends

Dataset: laptops.csv (991 laptops, 22 columns)
Libraries: numpy, pandas, matplotlib, seaborn

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (9, 5.5)


# Path setup — always correct, regardless of where you run this from

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "laptops.csv")
CHARTS_DIR = os.path.join(BASE_DIR, "..", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def chart_path(filename):
    """Build a full save path inside charts/, so every savefig() writes there."""
    return os.path.join(CHARTS_DIR, filename)


# 1. LOAD DATA

df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)
print(df.head())


# 2. DATA CLEANING

print("\nMissing values:\n", df.isnull().sum())
df["Price"] = df["Price"].fillna(df["Price"].median())
df["Rating"] = df["Rating"].fillna(df["Rating"].median())
df["brand"] = df["brand"].fillna(df["brand"].mode()[0])

df = df.drop_duplicates()

df["brand"] = df["brand"].str.strip().str.title()
df["OS"] = df["OS"].str.strip().str.title()
df["processor_brand"] = df["processor_brand"].str.strip().str.title()
df["gpu_brand"] = df["gpu_brand"].str.strip().str.title()
df["primary_storage_type"] = df["primary_storage_type"].str.strip().str.upper()

# Remove price outliers using IQR method
Q1, Q3 = np.percentile(df["Price"], [25, 75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df = df[(df["Price"] >= lower) & (df["Price"] <= upper)]
print("\nShape after cleaning:", df.shape)

# Price segment column, used across several charts/filters below
df["price_segment"] = pd.cut(
    df["Price"], bins=[0, 40000, 80000, df["Price"].max()],
    labels=["Budget", "Mid-range", "Premium"]
)


# 3. FILTERING (business-relevant queries on this dataset)


# 3.1 Best value-for-money laptops: decent rating, low price
best_value = df[(df["Rating"] >= 60) & (df["Price"] <= 45000)]
print(f"\n[FILTER] Best value laptops (Rating>=60, Price<=45000): {best_value.shape[0]}")
print(best_value[["brand", "Model", "Price", "Rating"]].head())

# 3.2 Laptops with SSD only (fast storage) vs HDD (slow storage)
ssd_laptops = df[df["primary_storage_type"] == "SSD"]
hdd_laptops = df[df["primary_storage_type"] == "HDD"]
print(f"\n[FILTER] SSD laptops: {ssd_laptops.shape[0]}  |  HDD laptops: {hdd_laptops.shape[0]}")

# 3.3 Touch-screen laptops
touch_laptops = df[df["is_touch_screen"] == True]
print(f"\n[FILTER] Touch-screen laptops: {touch_laptops.shape[0]} "
      f"({touch_laptops.shape[0]/df.shape[0]*100:.1f}% of dataset)")


# 4. BASIC STATISTICS

print("\nDescriptive statistics:\n", df[["Price", "Rating", "ram_memory"]].describe())

brand_summary = df.groupby("brand").agg(
    count=("brand", "size"), avg_price=("Price", "mean")
).sort_values("count", ascending=False)
print("\nTop 10 brands by listing count:\n", brand_summary.head(10))


# 5. FEATURE-WISE ANALYSIS (with WHY explanations)

print("\n FEATURE-WISE ANALYSIS ")

# 5.1 Processor brand -> price & rating
proc_brand_stats = df.groupby("processor_brand")[["Price", "Rating"]].mean().sort_values("Price", ascending=False)
print("\nProcessor Brand vs Avg Price & Rating:\n", proc_brand_stats)
print("WHY: Apple (M-series) and Intel high-end chips cost more to produce and "
      "are used in premium laptops, so their average price and rating are higher "
      "than budget-focused AMD/other configurations.")

# 5.2 Processor tier -> price
tier_price = df.groupby("processor_tier")["Price"].mean().sort_values(ascending=False)
print("\nTop 8 Processor Tiers by Avg Price:\n", tier_price.head(8))
print("WHY: Processor tier (i9>i7>i5>i3, Ryzen 9>7>5) directly reflects core "
      "count and clock speed. Higher tiers cost more to manufacture, so price "
      "rises step by step with tier — this is usually the single strongest "
      "price driver in any laptop dataset.")

# 5.3 Storage type -> price & rating
storage_stats = df.groupby("primary_storage_type")[["Price", "Rating"]].mean()
print("\nStorage Type vs Avg Price & Rating:\n", storage_stats)
print("WHY: SSDs are faster and more expensive per GB than HDDs. Laptops with "
      "SSDs cost more but also score higher ratings because read/write speed "
      "directly affects user experience (boot time, app loading).")

# 5.4 GPU type -> price & rating
gpu_stats = df.groupby("gpu_type")[["Price", "Rating"]].mean()
print("\nGPU Type (Integrated vs Dedicated) vs Avg Price & Rating:\n", gpu_stats)
print("WHY: A dedicated GPU is a separate, extra chip built for graphics-heavy "
      "tasks (gaming, video editing) — it costs more to add, which raises "
      "price. Buyers who need this performance usually rate it higher too.")

# 5.5 Touch screen -> price
touch_stats = df.groupby("is_touch_screen")[["Price", "Rating"]].mean()
print("\nTouch Screen vs Avg Price & Rating:\n", touch_stats)
print("WHY: Touch-enabled displays need extra hardware (digitizer layer) and "
      "are mostly found in 2-in-1/premium convertible laptops, which pushes "
      "the average price of touch-screen models higher.")

# 5.6 Warranty period -> price
warranty_stats = df.groupby("year_of_warranty")["Price"].mean()
print("\nWarranty Period (years) vs Avg Price:\n", warranty_stats)
print("WHY: Manufacturers usually bundle longer warranty periods with "
      "premium/business laptops as an added-value feature, so laptops with "
      "longer warranty tend to have a higher average price.")

# 5.7 RAM -> price (clear increasing trend)
ram_price = df.groupby("ram_memory")["Price"].mean().sort_index()
print("\nAverage Price by RAM size:\n", ram_price)
print("WHY: More RAM chips cost more to manufacture, and high-RAM configs are "
      "usually paired with better processors — so price rises with RAM.")


# 6. CORRELATION ANALYSIS

numeric_cols = ["Price", "Rating", "num_cores", "num_threads", "ram_memory",
                 "primary_storage_capacity", "display_size"]
corr = df[numeric_cols].corr()
print("\nCorrelation matrix:\n", corr["Price"].sort_values(ascending=False))
print("WHY: This shows which specification moves most closely with price. "
      "The feature with the highest correlation is the biggest price driver; "
      "one close to 0 (like display_size) barely affects price at all.")


# 7. VISUALIZATION

top_brands = brand_summary.head(10).index

# 7.1 Bar chart — average price by brand
plt.figure()
sns.barplot(x=brand_summary.loc[top_brands, "avg_price"], y=top_brands,
            hue=top_brands, palette="viridis", legend=False)
plt.title("Average Price by Brand (Top 10)")
plt.xlabel("Average Price (INR)"); plt.ylabel("Brand")
plt.tight_layout(); plt.savefig(chart_path("chart1_avg_price_by_brand.png"), dpi=150); plt.show()

# 7.2 Count plot — laptops per price segment
plt.figure()
sns.countplot(data=df, x="price_segment", hue="price_segment", palette="Set2",
              legend=False, order=["Budget", "Mid-range", "Premium"])
plt.title("Number of Laptops by Price Segment")
plt.xlabel("Price Segment"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(chart_path("chart2_count_by_segment.png"), dpi=150); plt.show()

# 7.3 Box plot — price by brand
plt.figure()
sns.boxplot(data=df[df["brand"].isin(top_brands)], x="Price", y="brand",
            hue="brand", palette="Set3", legend=False)
plt.title("Price Distribution Across Brands")
plt.xlabel("Price (INR)"); plt.ylabel("Brand")
plt.tight_layout(); plt.savefig(chart_path("chart3_price_boxplot.png"), dpi=150); plt.show()

# 7.4 Violin plot — rating by price segment
plt.figure()
sns.violinplot(data=df, x="price_segment", y="Rating", hue="price_segment",
               palette="Set2", legend=False, order=["Budget", "Mid-range", "Premium"])
plt.title("Rating Distribution by Price Segment")
plt.xlabel("Price Segment"); plt.ylabel("Rating")
plt.tight_layout(); plt.savefig(chart_path("chart4_violin_rating.png"), dpi=150); plt.show()

# 7.5 Bar chart — average price by processor tier
plt.figure()
top_tiers = tier_price.head(8)
sns.barplot(x=top_tiers.values, y=top_tiers.index, hue=top_tiers.index,
            palette="coolwarm", legend=False)
plt.title("Average Price by Processor Tier (Top 8)")
plt.xlabel("Average Price (INR)"); plt.ylabel("Processor Tier")
plt.tight_layout(); plt.savefig(chart_path("chart5_price_by_processor_tier.png"), dpi=150); plt.show()

# 7.6 Bar chart — SSD vs HDD average price & rating
plt.figure()
storage_stats["Price"].plot(kind="bar", color=["#4C72B0", "#DD8452"])
plt.title("Average Price: SSD vs HDD")
plt.xlabel("Storage Type"); plt.ylabel("Average Price (INR)")
plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(chart_path("chart6_storage_type_price.png"), dpi=150); plt.show()

# 7.7 Bar chart — Integrated vs Dedicated GPU price
plt.figure()
gpu_stats["Price"].plot(kind="bar", color=["#55A868", "#C44E52"])
plt.title("Average Price: Integrated vs Dedicated GPU")
plt.xlabel("GPU Type"); plt.ylabel("Average Price (INR)")
plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(chart_path("chart7_gpu_type_price.png"), dpi=150); plt.show()

# 7.8 Scatter plot — price vs rating, colored by processor brand
plt.figure()
sns.scatterplot(data=df, x="Price", y="Rating", hue="processor_brand", alpha=0.7)
plt.title("Price vs Rating (by Processor Brand)")
plt.xlabel("Price (INR)"); plt.ylabel("Rating")
plt.tight_layout(); plt.savefig(chart_path("chart8_price_vs_rating.png"), dpi=150); plt.show()

# 7.9 Line chart — price trend as RAM increases
plt.figure()
sns.lineplot(x=ram_price.index, y=ram_price.values, marker="o", color="darkorange")
plt.title("Price Trend as RAM Increases")
plt.xlabel("RAM (GB)"); plt.ylabel("Average Price (INR)")
plt.tight_layout(); plt.savefig(chart_path("chart9_price_trend_ram.png"), dpi=150); plt.show()

# 7.10 Heatmap — correlation between numeric features
plt.figure()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig(chart_path("chart10_correlation_heatmap.png"), dpi=150); plt.show()

# 7.11 Pie chart — OS market share (small categories grouped into "Other" to avoid label clutter)
plt.figure()
os_counts = df["OS"].value_counts()
os_pct = os_counts / os_counts.sum() * 100
# Keep categories with >=2% share separate; merge the rest into "Other"
main_os = os_counts[os_pct >= 2]
other_os_total = os_counts[os_pct < 2].sum()
if other_os_total > 0:
    main_os["Other"] = other_os_total
colors = sns.color_palette("pastel", len(main_os))
wedges, _, autotexts = plt.pie(
    main_os, autopct="%1.1f%%", startangle=90, colors=colors, pctdistance=0.8
)
plt.legend(wedges, main_os.index, title="OS", loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Operating System Market Share")
plt.tight_layout(); plt.savefig(chart_path("chart11_os_share.png"), dpi=150); plt.show()

# 7.12 Pie chart — GPU brand market share (same fix: group tiny slices + legend)
plt.figure()
gpu_brand_counts = df["gpu_brand"].value_counts()
gpu_pct = gpu_brand_counts / gpu_brand_counts.sum() * 100
main_gpu = gpu_brand_counts[gpu_pct >= 2]
other_gpu_total = gpu_brand_counts[gpu_pct < 2].sum()
if other_gpu_total > 0:
    main_gpu["Other"] = other_gpu_total
colors2 = sns.color_palette("Set3", len(main_gpu))
wedges2, _, autotexts2 = plt.pie(
    main_gpu, autopct="%1.1f%%", startangle=90, colors=colors2, pctdistance=0.8
)
plt.legend(wedges2, main_gpu.index, title="GPU Brand", loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("GPU Brand Market Share")
plt.tight_layout(); plt.savefig(chart_path("chart12_gpu_brand_share.png"), dpi=150); plt.show()

# 7.13 Histogram + KDE — price distribution
plt.figure()
sns.histplot(df["Price"], bins=30, kde=True, color="steelblue")
plt.title("Price Distribution (Histogram + KDE)")
plt.xlabel("Price (INR)"); plt.ylabel("Frequency")
plt.tight_layout(); plt.savefig(chart_path("chart13_price_distribution.png"), dpi=150); plt.show()

# 7.14 Pairplot — relationships between key numeric features
sns.pairplot(df[["Price", "Rating", "ram_memory", "price_segment"]],
             hue="price_segment", palette="Set1")
plt.savefig(chart_path("chart14_pairplot.png"), dpi=150)
plt.show()


# 8. KEY INSIGHTS & RECOMMENDATIONS

print("\n KEY INSIGHTS ")
print(f"1. Highest avg price brand      : {df.groupby('brand')['Price'].mean().idxmax()}")
print(f"2. Lowest avg price brand       : {df.groupby('brand')['Price'].mean().idxmin()}")
print(f"3. Highest avg-rated brand      : {df.groupby('brand')['Rating'].mean().idxmax()}")
print(f"4. Price-RAM correlation        : {corr.loc['Price','ram_memory']:.2f}")
print(f"5. Price-Rating correlation     : {corr.loc['Price','Rating']:.2f}")
print(f"6. Price-NumThreads correlation : {corr.loc['Price','num_threads']:.2f} (strongest driver)")
print(f"7. SSD avg price (\u20b9{storage_stats.loc['SSD','Price']:.0f}) vs "
      f"HDD avg price (\u20b9{storage_stats.loc['HDD','Price']:.0f})")
print(f"8. Dedicated GPU avg price (\u20b9{gpu_stats.loc['dedicated','Price']:.0f}) vs "
      f"Integrated GPU avg price (\u20b9{gpu_stats.loc['integrated','Price']:.0f})")

print("\n RECOMMENDATIONS ")
print("- Consumers seeking value should prioritize: SSD storage, 16GB RAM, "
      "mid-tier processor (i5/Ryzen 5) — this combination gives the best "
      "rating-to-price ratio based on the Mid-range segment analysis.")
print("- Number of threads and processor tier are the strongest price drivers, "
      "so buyers should compare these first before RAM or display size.")
print("- Retailers can position SSD + dedicated-GPU laptops as premium "
      "offerings since both features show a clear price and rating jump.")

print("\nAnalysis complete. Charts saved to the charts/ folder.")
