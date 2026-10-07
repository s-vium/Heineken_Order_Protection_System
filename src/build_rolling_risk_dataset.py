from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


base_dir = Path(__file__).resolve().parents[1]
data_dir = base_dir / "data" / "raw" / "drive-download-20261006T150120Z-1-001"
out_dir = base_dir / "outputs"
out_dir.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Load core data
# ---------------------------------------------------------------------------
order_lines = pd.read_csv(
    data_dir / "order_lines.csv",
    parse_dates=["order_date", "delivered_date", "estimated_delivery_date"],
)
order_lines["order_month"] = pd.to_datetime(order_lines["order_month"].astype(str) + "-01")

reviews = pd.read_csv(
    data_dir / "order_reviews.csv",
    parse_dates=["review_creation_date", "review_answer_timestamp"],
)

geolocation = pd.read_csv(data_dir / "geolocation.csv")
geolocation["account_id"] = geolocation["account_id"].astype(str)

# ---------------------------------------------------------------------------
# Build order-level values and account/date mappings
# ---------------------------------------------------------------------------
order_value_table = (
    order_lines.groupby(["account_id", "order_id"], as_index=False)
    .agg(
        order_date=("order_date", "min"),
        order_month=("order_month", "min"),
        order_value=("price", "sum"),
        order_line_count=("order_item_id", "count"),
        distinct_categories=("product_category", "nunique"),
    )
    .sort_values(["account_id", "order_date"])
    .reset_index(drop=True)
)
order_value_table["order_value"] = order_value_table["order_value"].fillna(0)

# Review features need a mapping from order_id -> account_id
order_account_map = order_value_table[["order_id", "account_id"]].drop_duplicates()
reviews_with_account = reviews.merge(order_account_map, on="order_id", how="left")
reviews_with_account = reviews_with_account[reviews_with_account["account_id"].notna()].copy()

# Geolocation density features
city_density = geolocation.groupby("city").size().rename("city_account_count")
state_density = geolocation.groupby("state").size().rename("state_account_count")
geolocation_by_account = (
    geolocation.drop_duplicates("account_id").set_index("account_id").to_dict("index")
)
city_by_account = {
    account_id: geo.get("city") for account_id, geo in geolocation_by_account.items()
}
state_by_account = {
    account_id: geo.get("state") for account_id, geo in geolocation_by_account.items()
}
orders_by_account = {
    account_id: account_orders.sort_values("order_date").reset_index(drop=True)
    for account_id, account_orders in order_value_table.groupby("account_id", sort=False)
}
reviews_by_account = {
    account_id: account_reviews.sort_values("review_creation_date").reset_index(drop=True)
    for account_id, account_reviews in reviews_with_account.groupby("account_id", sort=False)
}

# ---------------------------------------------------------------------------
# Monthly account snapshots for rolling modeling
# ---------------------------------------------------------------------------
month_ends = pd.date_range(start="2017-01-31", end="2018-08-31", freq="MS")
month_ends = month_ends + pd.offsets.MonthEnd(0)

all_accounts = order_value_table["account_id"].drop_duplicates().tolist()

rows = []

for month_end in month_ends:
    history_start = month_end - pd.DateOffset(months=18)
    window_hist = order_value_table[
        (order_value_table["order_date"] >= history_start)
        & (order_value_table["order_date"] <= month_end)
    ].copy()
    window_hist["city"] = window_hist["account_id"].map(city_by_account)
    window_hist["state"] = window_hist["account_id"].map(state_by_account)
    city_revenue_by_city = window_hist.groupby("city")["order_value"].sum()
    state_revenue_by_state = window_hist.groupby("state")["order_value"].sum()
    city_revenue_by_city = {
        city: revenue / max(int(city_density.get(city, 0)), 1)
        for city, revenue in city_revenue_by_city.items()
    }
    state_revenue_by_state = {
        state: revenue / max(int(state_density.get(state, 0)), 1)
        for state, revenue in state_revenue_by_state.items()
    }

    # make sure we work with all accounts in the scope even when no activity in this window
    for account_id in all_accounts:
        account_orders = orders_by_account[account_id]
        acc_hist = account_orders[
            (account_orders["order_date"] >= history_start)
            & (account_orders["order_date"] <= month_end)
        ]
        if acc_hist.empty:
            continue

        # basic history stats
        orders_last_18m = len(acc_hist)
        revenue_last_18m = float(acc_hist["order_value"].sum())
        avg_order_value = float(acc_hist["order_value"].mean())
        median_order_value = float(acc_hist["order_value"].median())
        max_order_value = float(acc_hist["order_value"].max())

        # recency and cycle
        latest_order_date = acc_hist["order_date"].iloc[-1]
        days_since_last_order = (month_end - latest_order_date).days

        if acc_hist["order_date"].nunique() > 1:
            ordered_dates = np.sort(acc_hist["order_date"].dropna().unique())
            diffs_days = np.diff(ordered_dates.astype("datetime64[ns]").astype("int64")) / (24 * 60 * 60 * 1e9)
            normal_cycle = float(np.median(diffs_days)) if len(diffs_days) > 0 else np.nan
            mean_cycle = float(np.mean(diffs_days)) if len(diffs_days) > 0 else np.nan
            std_cycle = float(np.std(diffs_days)) if len(diffs_days) > 0 else np.nan
        else:
            normal_cycle = np.nan
            mean_cycle = np.nan
            std_cycle = np.nan

        # recent vs history windows
        recent_3m = acc_hist[acc_hist["order_date"] >= (month_end - pd.DateOffset(months=3))]
        prior_3m = acc_hist[
            (acc_hist["order_date"] >= (month_end - pd.DateOffset(months=6)))
            & (acc_hist["order_date"] < (month_end - pd.DateOffset(months=3)))
        ]
        recent_6m = acc_hist[acc_hist["order_date"] >= (month_end - pd.DateOffset(months=6))]
        prior_6m = acc_hist[
            (acc_hist["order_date"] >= (month_end - pd.DateOffset(months=12)))
            & (acc_hist["order_date"] < (month_end - pd.DateOffset(months=6)))
        ]

        recent_3_orders = len(recent_3m)
        prior_3_orders = len(prior_3m)
        recent_6_orders = len(recent_6m)
        prior_6_orders = len(prior_6m)

        recent_3_revenue = float(recent_3m["order_value"].sum())
        prior_3_revenue = float(prior_3m["order_value"].sum())
        recent_6_revenue = float(recent_6m["order_value"].sum())
        prior_6_revenue = float(prior_6m["order_value"].sum())

        def safe_ratio(num: float, den: float) -> float:
            if pd.isna(num) or pd.isna(den) or den == 0:
                return np.nan
            return float(num / den)

        order_drop_3m = safe_ratio(recent_3_orders, prior_3_orders)
        revenue_drop_3m = safe_ratio(recent_3_revenue, prior_3_revenue)
        order_drop_6m = safe_ratio(recent_6_orders, prior_6_orders)
        revenue_drop_6m = safe_ratio(recent_6_revenue, prior_6_revenue)

        # recency and activity completeness
        active_months_3m = acc_hist[acc_hist["order_date"] >= (month_end - pd.DateOffset(months=3))]["order_month"].nunique()
        active_months_6m = acc_hist[acc_hist["order_date"] >= (month_end - pd.DateOffset(months=6))]["order_month"].nunique()
        active_months_12m = acc_hist[acc_hist["order_date"] >= (month_end - pd.DateOffset(months=12))]["order_month"].nunique()
        active_months_18m = acc_hist["order_month"].nunique()

        # Value drop signal based on the latest order versus historical mean
        latest_order_value = float(acc_hist.iloc[-1]["order_value"])
        avg_historic_order_value = float(acc_hist.iloc[:-1]["order_value"].mean()) if len(acc_hist) > 1 else latest_order_value
        value_drop_ratio = safe_ratio(latest_order_value, avg_historic_order_value)

        # Review features (last 18 months)
        account_reviews = reviews_by_account.get(account_id)
        if account_reviews is None:
            review_window = reviews_with_account.iloc[0:0]
        else:
            review_window = account_reviews[
                (account_reviews["review_creation_date"] >= history_start)
                & (account_reviews["review_creation_date"] <= month_end)
            ]

        review_count_18m = int(len(review_window))
        avg_review_score_18m = float(review_window["review_score"].mean()) if review_count_18m else np.nan
        positive_review_share_18m = float((review_window["review_score"] >= 4).mean()) if review_count_18m else np.nan
        negative_review_share_18m = float((review_window["review_score"] <= 2).mean()) if review_count_18m else np.nan

        review_start_3m = month_end - pd.DateOffset(months=3)
        review_start_6m = month_end - pd.DateOffset(months=6)
        recent_review_window = review_window[
            review_window["review_creation_date"] >= review_start_3m
        ]
        recent_review_window_6m = review_window[
            review_window["review_creation_date"] >= review_start_6m
        ]
        prior_review_window = review_window[
            (review_window["review_creation_date"] >= review_start_6m)
            & (review_window["review_creation_date"] < review_start_3m)
        ]
        latest_review_score = (
            float(review_window.sort_values("review_creation_date").iloc[-1]["review_score"])
            if review_count_18m
            else np.nan
        )
        latest_review_prior_avg = (
            float(review_window.sort_values("review_creation_date").iloc[:-1]["review_score"].mean())
            if review_count_18m > 1
            else np.nan
        )
        latest_review_vs_prior_avg = (
            latest_review_score - latest_review_prior_avg
            if pd.notna(latest_review_score) and pd.notna(latest_review_prior_avg)
            else np.nan
        )
        avg_review_score_3m = (
            float(recent_review_window["review_score"].mean())
            if not recent_review_window.empty
            else np.nan
        )
        prior_avg_review_score_3m = (
            float(prior_review_window["review_score"].mean())
            if not prior_review_window.empty
            else np.nan
        )
        review_score_trend_3m = (
            avg_review_score_3m - prior_avg_review_score_3m
            if pd.notna(avg_review_score_3m) and pd.notna(prior_avg_review_score_3m)
            else np.nan
        )
        negative_review_share_6m = (
            float((recent_review_window_6m["review_score"] <= 2).mean())
            if not recent_review_window_6m.empty
            else np.nan
        )

        if account_reviews is None:
            recent_reviews = reviews_with_account.iloc[0:0]
        else:
            recent_reviews = account_reviews[
                (account_reviews["review_creation_date"] >= (month_end - pd.DateOffset(months=6)))
                & (account_reviews["review_creation_date"] <= month_end)
            ]
        recent_review_count = int(len(recent_reviews))
        recent_avg_review_score = float(recent_reviews["review_score"].mean()) if recent_review_count else np.nan

        # Geolocation features
        geo_row = geolocation_by_account.get(account_id)
        if geo_row is None:
            city_name = np.nan
            state_name = np.nan
            lat = np.nan
            lng = np.nan
            city_account_count = 0
            state_account_count = 0
            city_revenue = np.nan
            state_revenue = np.nan
        else:
            city_name = geo_row["city"]
            state_name = geo_row["state"]
            lat = float(geo_row["lat"])
            lng = float(geo_row["lng"])

            city_account_count = int(city_density.get(city_name, 0))
            state_account_count = int(state_density.get(state_name, 0))

            city_revenue_value = city_revenue_by_city.get(city_name, np.nan)
            state_revenue_value = state_revenue_by_state.get(state_name, np.nan)

        # Main target definition
        eligible = orders_last_18m >= 3
        if eligible and pd.notna(normal_cycle) and normal_cycle > 0:
            churn = int(days_since_last_order >= 2.5 * normal_cycle)
        else:
            churn = 0

        loss = int(eligible and (avg_order_value > 0) and (latest_order_value <= 0.3 * avg_historic_order_value))
        high_risk = int(churn or loss)

        row = {
            "account_id": account_id,
            "snapshot_month": month_end,
            "eligible": int(eligible),
            "orders_last_18m": orders_last_18m,
            "revenue_last_18m": revenue_last_18m,
            "avg_order_value_18m": avg_order_value,
            "median_order_value_18m": median_order_value,
            "max_order_value_18m": max_order_value,
            "latest_order_date": latest_order_date,
            "days_since_last_order": days_since_last_order,
            "normal_cycle_days": normal_cycle,
            "mean_cycle_days": mean_cycle,
            "std_cycle_days": std_cycle,
            "recent_3m_orders": recent_3_orders,
            "prior_3m_orders": prior_3_orders,
            "recent_6m_orders": recent_6_orders,
            "prior_6m_orders": prior_6_orders,
            "order_drop_ratio_3m": order_drop_3m,
            "order_drop_ratio_6m": order_drop_6m,
            "revenue_drop_ratio_3m": revenue_drop_3m,
            "revenue_drop_ratio_6m": revenue_drop_6m,
            "recent_3m_revenue": recent_3_revenue,
            "prior_3m_revenue": prior_3_revenue,
            "recent_6m_revenue": recent_6_revenue,
            "prior_6m_revenue": prior_6_revenue,
            "active_months_3m": active_months_3m,
            "active_months_6m": active_months_6m,
            "active_months_12m": active_months_12m,
            "active_months_18m": active_months_18m,
            "latest_order_value": latest_order_value,
            "avg_historic_order_value": avg_historic_order_value,
            "value_drop_ratio": value_drop_ratio,
            "review_count_18m": review_count_18m,
            "avg_review_score_18m": avg_review_score_18m,
            "positive_review_share_18m": positive_review_share_18m,
            "negative_review_share_18m": negative_review_share_18m,
            "latest_review_score": latest_review_score,
            "latest_review_vs_prior_avg": latest_review_vs_prior_avg,
            "avg_review_score_3m": avg_review_score_3m,
            "prior_avg_review_score_3m": prior_avg_review_score_3m,
            "review_score_trend_3m": review_score_trend_3m,
            "negative_review_share_6m": negative_review_share_6m,
            "recent_review_count_6m": recent_review_count,
            "recent_avg_review_score_6m": recent_avg_review_score,
            "lat": lat,
            "lng": lng,
            "city": city_name,
            "state": state_name,
            "city_account_count": city_account_count,
            "state_account_count": state_account_count,
            "city_avg_revenue_18m": city_revenue_value if geo_row is not None else np.nan,
            "state_avg_revenue_18m": state_revenue_value if geo_row is not None else np.nan,
            "churn": churn,
            "loss": loss,
            "high_risk": high_risk,
        }
        rows.append(row)

risk_df = pd.DataFrame(rows)

# Keep lower-history snapshots so scoring can report insufficient information.
risk_df = risk_df.reset_index(drop=True)

# Include accounts with no orders in the latest window in the scoring-status roster.
latest_snapshot = month_ends[-1]
latest_history_start = latest_snapshot - pd.DateOffset(months=18)
latest_order_counts = (
    order_value_table[
        (order_value_table["order_date"] >= latest_history_start)
        & (order_value_table["order_date"] <= latest_snapshot)
    ]
    .groupby("account_id")
    .size()
    .reindex(all_accounts, fill_value=0)
)
latest_status = pd.DataFrame({
    "account_id": all_accounts,
    "snapshot_month": latest_snapshot,
    "eligibility_order_count_18m": latest_order_counts.reindex(all_accounts).to_numpy(),
})
latest_status["eligible"] = (latest_status["eligibility_order_count_18m"] >= 3).astype(int)
latest_status["score_status"] = np.where(
    latest_status["eligible"] == 1,
    "eligible_for_scoring",
    "insufficient_data",
)
latest_status.to_csv(out_dir / "latest_account_scoring_status.csv", index=False)

# Save the full snapshots; the training and scoring consumers apply the eligibility gate.
out_file = out_dir / "rolling_account_risk_dataset.csv"
risk_df.to_csv(out_file, index=False)

print(f"Rows in rolling account dataset: {len(risk_df)}")
print(f"Eligible accounts in latest snapshot: {latest_status['eligible'].sum()}")
print(f"Insufficient-data accounts in latest snapshot: {(latest_status['eligible'] == 0).sum()}")
print(f"Share high-risk: {risk_df['high_risk'].mean():.4f}")
print(f"Share churn: {risk_df['churn'].mean():.4f}")
print(f"Share loss: {risk_df['loss'].mean():.4f}")
print(f"Saved dataset to: {out_file}")

# Basic feature list to help downstream modeling
feature_cols = [
    c for c in risk_df.columns
    if c not in {"account_id", "snapshot_month", "city", "state", "latest_order_date", "eligible", "churn", "loss", "high_risk"}
]
print("Feature columns:")
for col in feature_cols:
    print(" -", col)
