
from pathlib import Path
import gc

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CHUNK_SIZE = 25_000


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        RAW_DIR.rglob("*.csv"),
        key=lambda p: p.stat().st_size,
        reverse=True
    )

    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    source = candidates[0]
    output = PROCESSED_DIR / "aml_cleaned.csv"

    print(f"Reading: {source.name}")
    print(
        f"File size: "
        f"{source.stat().st_size / (1024 ** 3):.2f} GB"
    )

    # Prevent mixing old and new outputs.
    if output.exists():
        output.unlink()

    total_rows = 0
    first_chunk = True

    try:
        reader = pd.read_csv(
            source,
            chunksize=CHUNK_SIZE,
            low_memory=True
        )

        for chunk_number, chunk in enumerate(reader, 1):

            # Standardise column names.
            chunk.columns = (
                chunk.columns
                .str.strip()
                .str.lower()
                .str.replace(
                    r"\s+", "_", regex=True
                )
            )

            # Remove completely empty records.
            chunk = chunk.dropna(how="all")

            # Do not remove transaction duplicates
            # without checking transaction IDs.
            # Repeated values may be legitimate.

            chunk.to_csv(
                output,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            first_chunk = False
            total_rows += len(chunk)

            print(
                f"Chunk {chunk_number}: "
                f"{len(chunk):,} rows | "
                f"Total: {total_rows:,}"
            )

            del chunk
            gc.collect()

    except Exception:
        # Remove incomplete output after a failure.
        if output.exists():
            output.unlink()
        raise

    print("\nCleaning completed.")
    print(f"Total records: {total_rows:,}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()

