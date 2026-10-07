import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


base_dir = Path(__file__).resolve().parents[1]
data_dir = base_dir / "data" / "raw" / "drive-download-20261006T150120Z-1-001"
out_dir = base_dir / "outputs"
out_dir.mkdir(exist_ok=True)

# Analysis date as stated in the brief
analysis_date = pd.Timestamp("2018-08-31")

# Load datasets
orders = pd.read_csv(data_dir / "orders.csv", parse_dates=[
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
])
order_lines = pd.read_csv(data_dir / "order_lines.csv", parse_dates=[
    "order_date",
    "delivered_date",
    "estimated_delivery_date",
])
reviews = pd.read_csv(data_dir / "order_reviews.csv", parse_dates=[
    "review_creation_date",
    "review_answer_timestamp",
])

# Build account-level summary
order_lines["order_month"] = pd.to_datetime(order_lines["order_month"].astype(str) + "-01")
account_summary = order_lines.groupby("account_id").agg(
    first_order=("order_date", "min"),
    last_order=("order_date", "max"),
    order_count=("order_id", "nunique"),
    total_revenue=("price", "sum"),
    categories=("product_category", "nunique"),
).reset_index()
account_summary["days_since_last_order"] = (analysis_date - account_summary["last_order"]).dt.days

# Monthly account behavior to detect decline using own history
account_month = (
    order_lines.groupby(["account_id", "order_month"])
    .agg(order_count=("order_id", "nunique"), revenue=("price", "sum"))
    .reset_index()
)

# recent 3 months versus previous 3 months
recent_period = account_month[account_month["order_month"] >= pd.Timestamp("2018-06-01")].copy()
previous_period = account_month[
    (account_month["order_month"] >= pd.Timestamp("2018-03-01"))
    & (account_month["order_month"] < pd.Timestamp("2018-06-01"))
].copy()

recent_summary = recent_period.groupby("account_id").agg(
    recent_order_count=("order_count", "sum"),
    recent_revenue=("revenue", "sum"),
)
previous_summary = previous_period.groupby("account_id").agg(
    prior_order_count=("order_count", "sum"),
    prior_revenue=("revenue", "sum"),
)

account_summary = account_summary.merge(recent_summary, on="account_id", how="left").merge(previous_summary, on="account_id", how="left")
account_summary["recent_order_count"] = account_summary["recent_order_count"].fillna(0)
account_summary["prior_order_count"] = account_summary["prior_order_count"].fillna(0)
account_summary["recent_revenue"] = account_summary["recent_revenue"].fillna(0)
account_summary["prior_revenue"] = account_summary["prior_revenue"].fillna(0)
account_summary["order_ratio_recent_to_prior"] = (
    account_summary["recent_order_count"] / account_summary["prior_order_count"].replace(0, np.nan)
)
account_summary["revenue_ratio_recent_to_prior"] = (
    account_summary["recent_revenue"] / account_summary["prior_revenue"].replace(0, np.nan)
)

# Risk score: recency + drop in order volume + drop in spend
account_summary["recency_component"] = np.where(account_summary["days_since_last_order"] > 90, 1, 0)
account_summary["order_drop"] = np.where(account_summary["order_ratio_recent_to_prior"].fillna(1) <= 0.5, 1, 0)
account_summary["revenue_drop"] = np.where(account_summary["revenue_ratio_recent_to_prior"].fillna(1) <= 0.5, 1, 0)
account_summary["risk_score"] = (
    0.45 * account_summary["recency_component"]
    + 0.35 * account_summary["order_drop"]
    + 0.20 * account_summary["revenue_drop"]
)

# Datasets for plots
monthly_orders = (
    order_lines.assign(order_month=pd.to_datetime(order_lines["order_month"].astype(str) + "-01"))
    .groupby("order_month")
    .agg(orders=("order_id", "nunique"), revenue=("price", "sum"))
    .reset_index()
)

category_order = (
    order_lines.groupby("product_category")
    .agg(orders=("order_id", "nunique"), revenue=("price", "sum"))
    .sort_values("orders", ascending=False)
    .head(10)
    .reset_index()
)

late_share = order_lines["is_late"].value_counts(normalize=True, dropna=False)
late_share = late_share.rename(index={True: "Late", False: "On time", np.nan: "Not delivered"})

review_scores = reviews["review_score"].value_counts().sort_index()
review_scores.index = review_scores.index.astype(int)

# Plot 1: Monthly order volume and revenue
plt.figure(figsize=(12, 5))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(monthly_orders["order_month"], monthly_orders["orders"], color="steelblue", marker="o", linewidth=2, label="Orders")
ax2.plot(monthly_orders["order_month"], monthly_orders["revenue"], color="darkorange", marker="s", linewidth=2, label="Revenue")
ax1.set_title("Order volume and revenue over time")
ax1.set_xlabel("Month")
ax1.set_ylabel("Orders", color="steelblue")
ax2.set_ylabel("Revenue", color="darkorange")
ax1.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(out_dir / "monthly_orders_and_revenue.png", dpi=220)
plt.close()

# Plot 2: Order count distribution by account
plt.figure(figsize=(10, 5))
ax = sns.histplot(account_summary["order_count"], bins=40, kde=True, color="steelblue")
ax.set_title("Distribution of order counts per account")
ax.set_xlabel("Orders per account")
ax.set_ylabel("Accounts")
plt.tight_layout()
plt.savefig(out_dir / "account_order_count_distribution.png", dpi=220)
plt.close()

# Plot 3: Recency of last order
plt.figure(figsize=(10, 5))
ax = sns.histplot(account_summary["days_since_last_order"], bins=40, color="darkred")
ax.set_title("Distribution of days since last order")
ax.set_xlabel("Days since last order")
ax.set_ylabel("Accounts")
plt.tight_layout()
plt.savefig(out_dir / "days_since_last_order_distribution.png", dpi=220)
plt.close()

# Plot 4: Risk scatter
plot_df = account_summary[account_summary["order_count"] >= 2].copy()
plot_df["decline_ratio"] = plot_df["recent_order_count"] / plot_df["prior_order_count"].replace(0, np.nan)
plot_df["decline_ratio"] = plot_df["decline_ratio"].fillna(0)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    plot_df["days_since_last_order"],
    plot_df["decline_ratio"],
    c=plot_df["risk_score"],
    cmap="RdYlGn_r",
    s=35,
    alpha=0.75,
)
plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
plt.axvline(90, color="black", linestyle="--", linewidth=1)
plt.title("Account risk: recency vs. recent order drop")
plt.xlabel("Days since last order")
plt.ylabel("Recent/previous order count ratio")
plt.colorbar(scatter, label="Risk score")
plt.tight_layout()
plt.savefig(out_dir / "risk_scatter.png", dpi=220)
plt.close()

# Plot 5: Category mix and service quality
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.barplot(data=category_order, x="orders", y="product_category", palette="viridis", ax=axes[0])
axes[0].set_title("Top 10 product categories by order count")
axes[0].set_xlabel("Orders")
axes[0].set_ylabel("Category")

late_share.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#55A868", "#C44E52"])
axes[1].set_title("Late delivery share")
axes[1].set_xlabel("Delivery status")
axes[1].set_ylabel("Share")
plt.tight_layout()
plt.savefig(out_dir / "category_and_service_quality.png", dpi=220)
plt.close()

# Plot 6: Review distribution
plt.figure(figsize=(10, 5))
ax = sns.barplot(x=review_scores.index, y=review_scores.values, palette="coolwarm")
ax.set_title("Review score distribution")
ax.set_xlabel("Review score")
ax.set_ylabel("Number of reviews")
plt.tight_layout()
plt.savefig(out_dir / "review_scores.png", dpi=220)
plt.close()

# Print concise summary for the console
print("Accounts:", account_summary["account_id"].nunique())
print("Median orders per account:", account_summary["order_count"].median())
print("Mean days since last order:", account_summary["days_since_last_order"].mean())
print("Accounts with >90 days since last order:", (account_summary["days_since_last_order"] > 90).sum())
print("Accounts with recent order ratio <= 0.5:", (account_summary["order_ratio_recent_to_prior"].fillna(1) <= 0.5).sum())
print("Accounts with risk_score >= 0.8:", (account_summary["risk_score"] >= 0.8).sum())
print("Top risk accounts:")
print(
    account_summary[
        ["account_id", "order_count", "days_since_last_order", "recent_order_count", "prior_order_count", "risk_score"]
    ]
    .sort_values("risk_score", ascending=False)
    .head(10)
    .to_string(index=False)
)

# Save a compact CSV of the top risk accounts for action-oriented analysis
risk_df = (
    account_summary[
        [
            "account_id",
            "order_count",
            "total_revenue",
            "days_since_last_order",
            "recent_order_count",
            "prior_order_count",
            "order_ratio_recent_to_prior",
            "recent_revenue",
            "prior_revenue",
            "risk_score",
        ]
    ]
    .sort_values("risk_score", ascending=False)
    .head(500)
)
risk_df.to_csv(out_dir / "top_risk_accounts.csv", index=False)

print(f"Saved plots and summary data to {out_dir}")
