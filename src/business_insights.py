
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "data" / "processed" / "aml_cleaned.csv"
OUTPUT = BASE / "reports" / "business_insights"
OUTPUT.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 25_000

total_transactions = 0
laundering_transactions = 0

payment_stats = defaultdict(lambda: [0, 0])
route_stats = defaultdict(lambda: [0, 0])
daily_stats = defaultdict(lambda: [0, 0])
currency_stats = defaultdict(lambda: [0, 0.0, 0.0])

typology_counts = Counter()
sender_counts = Counter()
receiver_counts = Counter()

columns = [
    "date", "sender_account", "receiver_account",
    "amount", "payment_currency", "payment_type",
    "sender_bank_location", "receiver_bank_location",
    "is_laundering", "laundering_type"
]

for number, chunk in enumerate(
    pd.read_csv(
        SOURCE,
        usecols=columns,
        dtype={
            "sender_account": "string",
            "receiver_account": "string",
            "payment_currency": "string",
            "payment_type": "string",
            "sender_bank_location": "string",
            "receiver_bank_location": "string",
            "laundering_type": "string"
        },
        chunksize=CHUNK_SIZE
    ), 1
):
    chunk["is_laundering"] = pd.to_numeric(
        chunk["is_laundering"], errors="coerce"
    )

    # Only valid binary labels enter rate calculations.
    chunk = chunk[
        chunk["is_laundering"].isin([0, 1])
    ].copy()

    chunk["amount"] = pd.to_numeric(
        chunk["amount"], errors="coerce"
    )

    chunk["date"] = pd.to_datetime(
        chunk["date"], errors="coerce"
    )

    total_transactions += len(chunk)
    laundering_transactions += int(
        chunk["is_laundering"].sum()
    )

    def aggregate_rates(keys, target):
        grouped = chunk.groupby(
            keys, dropna=False
        )["is_laundering"].agg(["count", "sum"])

        for key, row in grouped.iterrows():
            target[key][0] += int(row["count"])
            target[key][1] += int(row["sum"])

    aggregate_rates("payment_type", payment_stats)

    chunk["route"] = (
        chunk["sender_bank_location"].fillna("Unknown")
        + " -> "
        + chunk["receiver_bank_location"].fillna("Unknown")
    )
    aggregate_rates("route", route_stats)

    valid_dates = chunk.dropna(subset=["date"]).copy()
    valid_dates["date"] = valid_dates["date"].dt.strftime(
        "%Y-%m-%d"
    )
    grouped_dates = valid_dates.groupby(
        "date"
    )["is_laundering"].agg(["count", "sum"])

    for date, row in grouped_dates.iterrows():
        daily_stats[date][0] += int(row["count"])
        daily_stats[date][1] += int(row["sum"])

    for currency, group in chunk.groupby(
        "payment_currency", dropna=False
    ):
        key = str(currency)
        currency_stats[key][0] += len(group)
        currency_stats[key][1] += group["amount"].sum()
        currency_stats[key][2] += group.loc[
            group["is_laundering"] == 1, "amount"
        ].sum()

    flagged = chunk[chunk["is_laundering"] == 1]

    typology_counts.update(
        flagged["laundering_type"]
        .dropna()
        .value_counts()
        .to_dict()
    )

    sender_counts.update(
        flagged["sender_account"]
        .dropna()
        .value_counts()
        .to_dict()
    )

    receiver_counts.update(
        flagged["receiver_account"]
        .dropna()
        .value_counts()
        .to_dict()
    )

    print(
        f"Chunk {number}: "
        f"{total_transactions:,} transactions processed"
    )


def save_rates(stats, category, filename):
    rows = []

    for key, (count, flagged) in stats.items():
        rows.append({
            category: str(key),
            "transactions": count,
            "laundering_transactions": flagged,
            "laundering_rate_pct": (
                100 * flagged / count if count else 0
            )
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / filename, index=False)
    return df


payment_df = save_rates(
    payment_stats, "payment_type", "payment_methods.csv"
)
route_df = save_rates(
    route_stats, "route", "banking_routes.csv"
)
daily_df = save_rates(
    daily_stats, "date", "daily_trends.csv"
)

currency_df = pd.DataFrame([
    {
        "payment_currency": currency,
        "transactions": values[0],
        "total_amount": values[1],
        "laundering_amount": values[2]
    }
    for currency, values in currency_stats.items()
])
currency_df.to_csv(
    OUTPUT / "currency_exposure.csv", index=False
)

pd.DataFrame(
    typology_counts.most_common(),
    columns=["laundering_type", "transactions"]
).to_csv(OUTPUT / "laundering_typologies.csv", index=False)

pd.DataFrame(
    sender_counts.most_common(100),
    columns=["sender_account", "flagged_transactions"]
).to_csv(OUTPUT / "top_senders.csv", index=False)

pd.DataFrame(
    receiver_counts.most_common(100),
    columns=["receiver_account", "flagged_transactions"]
).to_csv(OUTPUT / "top_receivers.csv", index=False)

summary = pd.DataFrame([{
    "total_transactions": total_transactions,
    "laundering_transactions": laundering_transactions,
    "laundering_rate_pct": (
        100 * laundering_transactions / total_transactions
        if total_transactions else 0
    )
}])
summary.to_csv(OUTPUT / "summary.csv", index=False)

print("\nAML BUSINESS ANALYSIS COMPLETE")
print("=" * 50)
print(f"Transactions: {total_transactions:,}")
print(f"Laundering: {laundering_transactions:,}")
print(
    f"Laundering rate: "
    f"{summary.iloc[0]['laundering_rate_pct']:.3f}%"
)
print(f"Reports saved to: {OUTPUT}")
