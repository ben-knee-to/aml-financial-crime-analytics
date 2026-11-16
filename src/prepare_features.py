
from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "processed" / "aml_cleaned.csv"
OUTPUT = ROOT / "data" / "processed" / "aml_features.parquet"
TEMP = ROOT / "data" / "temp"

MEMORY_LIMIT = "1GB"
THREADS = 2


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {SOURCE}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET threads={THREADS}")

    temp_path = str(TEMP).replace("'", "''")
    con.execute(f"SET temp_directory = '{temp_path}'")

    source_path = str(SOURCE).replace("'", "''")
    output_path = str(OUTPUT).replace("'", "''")

    print("=" * 60)
    print("AML FEATURE ENGINEERING")
    print("=" * 60)
    print(f"Source: {SOURCE}")
    print("Processing transactions using DuckDB...")

    # Stage 1: Parse the original transaction data.
    # Preserve account identifiers as strings.
    con.execute(f"""
        CREATE TEMP VIEW transactions AS
        SELECT
            ROW_NUMBER() OVER () AS transaction_key,

            TRY_CAST(
                CAST(date AS VARCHAR) || ' ' ||
                CAST(time AS VARCHAR)
                AS TIMESTAMP
            ) AS timestamp,

            CAST(sender_account AS VARCHAR)
                AS sender_account,

            CAST(receiver_account AS VARCHAR)
                AS receiver_account,

            TRY_CAST(amount AS DOUBLE) AS amount,

            CAST(payment_type AS VARCHAR)
                AS transaction_type,

            CAST(payment_currency AS VARCHAR)
                AS currency,

            CAST(sender_bank_location AS VARCHAR)
                AS sender_bank_location,

            CAST(receiver_bank_location AS VARCHAR)
                AS receiver_bank_location,

            TRY_CAST(is_laundering AS INTEGER)
                AS is_laundering

        FROM read_csv(
            '{source_path}',
            header=true,
            all_varchar=true
        )
    """)

    # Stage 2: Build time and amount features.
    #
    # Account history uses only earlier sender transactions.
    # Rows sharing the same timestamp are treated as simultaneous.
    # This prevents same-timestamp future-information leakage.

    print("Creating temporal and account-history features...")

    con.execute(f"""
        COPY (
            WITH valid AS (
                SELECT *
                FROM transactions
                WHERE timestamp IS NOT NULL
                  AND is_laundering IN (0, 1)
            ),

            history AS (
                SELECT
                    *,

                    COUNT(*) OVER (
                        PARTITION BY sender_account
                        ORDER BY timestamp
                        RANGE BETWEEN UNBOUNDED PRECEDING
                                  AND INTERVAL '1 microsecond' PRECEDING
                    ) AS previous_transaction_count,

                    MAX(timestamp) OVER (
                        PARTITION BY sender_account
                        ORDER BY timestamp
                        RANGE BETWEEN UNBOUNDED PRECEDING
                                  AND INTERVAL '1 microsecond' PRECEDING
                    ) AS previous_timestamp

                FROM valid
            )

            SELECT
                transaction_key,
                timestamp,
                is_laundering,

                EXTRACT(HOUR FROM timestamp)
                    AS transaction_hour,

                EXTRACT(DOW FROM timestamp)
                    AS transaction_day_of_week,

                CASE
                    WHEN EXTRACT(DOW FROM timestamp)
                         IN (0, 6)
                    THEN 1 ELSE 0
                END AS is_weekend,

                previous_transaction_count,

                CASE
                    WHEN previous_timestamp IS NULL
                    THEN NULL
                    ELSE DATE_DIFF(
                        'second',
                        previous_timestamp,
                        timestamp
                    ) / 3600.0
                END AS hours_since_previous,

                LN(1 + ABS(amount))
                    AS log_abs_amount,

                CASE
                    WHEN amount < 0 THEN 1
                    ELSE 0
                END AS negative_amount_flag,

                transaction_type,
                currency

            FROM history
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print("\nFeature engineering completed.")

    # Read only aggregate information, not the full dataset.
    result = con.execute(f"""
        SELECT
            COUNT(*) AS total_transactions,
            SUM(is_laundering) AS laundering_transactions,
            MIN(timestamp) AS first_transaction,
            MAX(timestamp) AS last_transaction
        FROM read_parquet('{output_path}')
    """).fetchdf()

    print(result.to_string(index=False))
    print(f"\nSaved to: {OUTPUT}")

    con.close()


if __name__ == "__main__":
    main()
