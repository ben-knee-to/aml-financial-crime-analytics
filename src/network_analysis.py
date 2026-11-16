from pathlib import Path
import json
import duckdb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'processed' / 'aml_cleaned.csv'
OUT = ROOT / 'reports' / 'network_analysis'
TEMP = ROOT / 'data' / 'temp'
MEMORY_LIMIT = '1GB'
THREADS = 2
TOP_N = 20


def save_query(con, filename, query):
    path = OUT / filename
    con.execute(f"COPY ({query}) TO '{str(path).replace(chr(39), chr(39)*2)}' (HEADER, DELIMITER ',')")
    print(f'Saved {path.name}')


def plot_top(csv_name, label_col, value_col, title, output_name):
    df = pd.read_csv(OUT / csv_name).head(15)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(df[label_col].astype(str).iloc[::-1], df[value_col].iloc[::-1])
    ax.set_title(title)
    ax.set_xlabel(value_col.replace('_', ' ').title())
    ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / output_name, dpi=160)
    plt.close(fig)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Missing input: {SOURCE}. Run clean_data.py first.')
    OUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f'SET threads={THREADS}')
    temp = str(TEMP).replace("'", "''")
    con.execute(f"SET temp_directory='{temp}'")
    src = str(SOURCE).replace("'", "''")

    print('AML NETWORK ANALYSIS')
    print('Reading transactions with DuckDB; no full Pandas load.')
    con.execute(f"""
        CREATE TEMP VIEW transfers AS
        SELECT
            NULLIF(TRIM(sender_account), '') AS sender,
            NULLIF(TRIM(receiver_account), '') AS receiver,
            TRY_CAST(amount AS DOUBLE) AS amount,
            NULLIF(TRIM(payment_currency), '') AS currency,
            TRY_CAST(is_laundering AS INTEGER) AS label,
            TRY_CAST(CAST(date AS VARCHAR) || ' ' || CAST(time AS VARCHAR) AS TIMESTAMP) AS ts
        FROM read_csv('{src}', header=true, all_varchar=true)
    """)
    con.execute("""
        CREATE TEMP VIEW edges AS
        SELECT * FROM transfers
        WHERE sender IS NOT NULL AND receiver IS NOT NULL
          AND sender <> receiver AND label IN (0, 1)
    """)

    save_query(con, 'network_overview.csv', """
        SELECT COUNT(*) AS transfers,
               COUNT(DISTINCT sender) AS unique_senders,
               COUNT(DISTINCT receiver) AS unique_receivers,
               COUNT(DISTINCT CASE WHEN label=1 THEN sender END) AS labelled_senders,
               SUM(CASE WHEN label=1 THEN 1 ELSE 0 END) AS labelled_transfers
        FROM edges
    """)

    # Aggregate directed edges first, substantially reducing later graph computations.
    con.execute("""
        CREATE TEMP TABLE pair_summary AS
        SELECT sender, receiver,
               COUNT(*) AS transaction_count,
               SUM(CASE WHEN label=1 THEN 1 ELSE 0 END) AS labelled_count,
               MIN(ts) AS first_seen, MAX(ts) AS last_seen
        FROM edges
        GROUP BY sender, receiver
    """)

    save_query(con, 'top_transaction_pairs.csv', f"""
        SELECT * FROM pair_summary
        ORDER BY transaction_count DESC, sender, receiver LIMIT {TOP_N}
    """)

    # Distinct neighbours and transaction counts are separate quantities.
    con.execute("""
        CREATE TEMP TABLE account_activity AS
        WITH outbound AS (
            SELECT sender AS account,
                   COUNT(*) AS outgoing_transactions,
                   COUNT(DISTINCT receiver) AS outgoing_counterparties,
                   SUM(CASE WHEN label=1 THEN 1 ELSE 0 END) AS outgoing_labelled
            FROM edges GROUP BY sender
        ), inbound AS (
            SELECT receiver AS account,
                   COUNT(*) AS incoming_transactions,
                   COUNT(DISTINCT sender) AS incoming_counterparties,
                   SUM(CASE WHEN label=1 THEN 1 ELSE 0 END) AS incoming_labelled
            FROM edges GROUP BY receiver
        )
        SELECT COALESCE(o.account, i.account) AS account,
               COALESCE(o.outgoing_transactions, 0) AS outgoing_transactions,
               COALESCE(i.incoming_transactions, 0) AS incoming_transactions,
               COALESCE(o.outgoing_counterparties, 0) AS outgoing_counterparties,
               COALESCE(i.incoming_counterparties, 0) AS incoming_counterparties,
               COALESCE(o.outgoing_labelled, 0) AS outgoing_labelled,
               COALESCE(i.incoming_labelled, 0) AS incoming_labelled
        FROM outbound o FULL OUTER JOIN inbound i ON o.account=i.account
    """)

    save_query(con, 'top_connected_accounts.csv', f"""
        SELECT *, outgoing_counterparties + incoming_counterparties AS directed_counterparty_connections
        FROM account_activity
        ORDER BY directed_counterparty_connections DESC, account LIMIT {TOP_N}
    """)
    save_query(con, 'top_labelled_accounts.csv', f"""
        SELECT *, outgoing_labelled + incoming_labelled AS labelled_involvements
        FROM account_activity
        WHERE outgoing_labelled + incoming_labelled > 0
        ORDER BY labelled_involvements DESC, account LIMIT {TOP_N}
    """)

    # A reciprocal pair is counted once using lexical ordering.
    save_query(con, 'reciprocal_pairs.csv', f"""
        SELECT a.sender AS account_a, a.receiver AS account_b,
               a.transaction_count AS a_to_b_count,
               b.transaction_count AS b_to_a_count,
               a.labelled_count + b.labelled_count AS labelled_transactions
        FROM pair_summary a
        JOIN pair_summary b
          ON a.sender=b.receiver AND a.receiver=b.sender
        WHERE a.sender < a.receiver
        ORDER BY labelled_transactions DESC,
                 (a.transaction_count+b.transaction_count) DESC,
                 account_a, account_b
        LIMIT 100
    """)

    # Three-account directed cycles A->B->C->A, with distinct accounts.
    # Restrict to the most frequent directed pairs for laptop-friendly exploration.
    # These are candidate motifs, not proof of sequential flow of funds.
    con.execute("""
        CREATE TEMP TABLE candidate_pairs AS
        SELECT sender, receiver, transaction_count, labelled_count
        FROM pair_summary
        ORDER BY transaction_count DESC, sender, receiver
        LIMIT 5000
    """)
    save_query(con, 'candidate_three_account_cycles.csv', """
        SELECT a.sender AS account_a, a.receiver AS account_b,
               b.receiver AS account_c,
               a.transaction_count AS a_to_b_count,
               b.transaction_count AS b_to_c_count,
               c.transaction_count AS c_to_a_count,
               a.labelled_count+b.labelled_count+c.labelled_count AS labelled_transactions
        FROM candidate_pairs a
        JOIN candidate_pairs b ON a.receiver=b.sender
        JOIN candidate_pairs c ON b.receiver=c.sender AND c.receiver=a.sender
        WHERE a.sender < a.receiver AND a.sender < b.receiver
          AND a.sender <> b.receiver
        ORDER BY labelled_transactions DESC,
                 a_to_b_count+b_to_c_count+c_to_a_count DESC
        LIMIT 100
    """)

    plot_top('top_connected_accounts.csv', 'account', 'directed_counterparty_connections',
             'Accounts with the Most Directed Counterparty Connections', 'top_connected_accounts.png')
    plot_top('top_labelled_accounts.csv', 'account', 'labelled_involvements',
             'Accounts Most Frequently Involved in Labelled Transfers', 'top_labelled_accounts.png')
    plot_top('top_transaction_pairs.csv', 'sender', 'transaction_count',
             'Most Active Sender Accounts in Top Directed Pairs', 'top_transaction_pairs.png')

    overview = pd.read_csv(OUT / 'network_overview.csv').iloc[0].to_dict()
    overview = {k: (int(v) if pd.notna(v) else None) for k, v in overview.items()}
    (OUT / 'network_summary.json').write_text(json.dumps({
        'overview': overview,
        'methodology': {
            'data': 'All valid directed transfers excluding self-transfers',
            'amounts': 'Not aggregated across currencies',
            'reciprocal_pairs': 'Two-way directed relationships, not necessarily laundering',
            'three_account_cycles': 'Structural motifs within top 5000 directed pairs; not time-ordered flows',
            'account_connections': 'Sum of distinct incoming and outgoing counterparties; may count the same counterparty twice',
            'labels': 'Synthetic dataset ground-truth labels; no production AML conclusions'
        }
    }, indent=2), encoding='utf-8')
    con.close()
    print('\nComplete. Reports saved in:', OUT)


if __name__ == '__main__':
    main()
