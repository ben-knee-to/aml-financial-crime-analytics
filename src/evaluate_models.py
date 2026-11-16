from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'modelling'
FEATURES_PATH = ROOT / 'data' / 'processed' / 'aml_features.parquet'
BATCH_SIZE = 25_000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((OUT / 'model_metadata.json').read_text(encoding='utf-8'))
    selected = meta['selected_model']
    features = meta['features']
    categorical = meta['categorical_features']
    names = ['random_forest', 'xgboost']
    models = {name: joblib.load(ROOT / 'models' / f'{name}.joblib') for name in names}
    thresholds = {name: float(meta['validation_results'][name]['threshold']) for name in names}

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(FEATURES_PATH)

    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'")
    con.execute('SET threads=2')
    test_start = pd.Timestamp(meta['test_start']).tz_localize(None) if pd.Timestamp(meta['test_start']).tzinfo is not None else pd.Timestamp(meta['test_start'])
    # Quote SQL identifiers only from the saved, locally generated metadata.
    columns = ['transaction_key', 'timestamp', 'is_laundering'] + features
    columns = list(dict.fromkeys(columns))
    escaped_cols = ', '.join('"' + c.replace('"', '""') + '"' for c in columns)
    # Parameter binding keeps the timestamp boundary safe and exact.
    sql = f'''SELECT {escaped_cols} FROM read_parquet(?)
              WHERE timestamp >= ? ORDER BY timestamp, transaction_key'''
    con.execute(
        sql,
        [str(FEATURES_PATH), test_start.to_pydatetime()]
    )

    # Keep just scores and labels in memory (~tens of MB for this holdout).
    labels = []
    scores = {name: [] for name in names}
    predictions_path = OUT / 'holdout_predictions.parquet'
    # Use a ParquetWriter so predictions are never accumulated in memory.
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    processed = 0
    try:
        while True:
            part = con.fetch_df_chunk(vectors_per_chunk=12)

            if part.empty:
                break
            y = part['is_laundering'].astype('int8').to_numpy()
            if not np.isin(y, [0, 1]).all():
                raise ValueError('Test set has non-binary labels')
            X = part[features].copy()
            for col in categorical:
                X[col] = X[col].astype('string').fillna('Unknown').astype(str)
            labels.append(y)
            for name in names:
                p = models[name].predict_proba(X)[:, 1].astype('float32')
                scores[name].append(p)
                if name == selected:
                    output = pd.DataFrame({
                        'transaction_key': part['transaction_key'].to_numpy(),
                        'timestamp': part['timestamp'].to_numpy(),
                        'is_laundering': y,
                        'model_score': p,
                        'alert': p >= thresholds[name],
                    })
                    table = pa.Table.from_pandas(output, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(str(predictions_path), table.schema, compression='zstd')
                    writer.write_table(table)
            processed += len(part)
            print(f'Processed {processed:,} test transactions', flush=True)
    finally:
        if writer is not None:
            writer.close()
        con.close()

    if not labels:
        raise ValueError('No test rows found; check model_metadata.json test_start')
    y = np.concatenate(labels)
    if np.unique(y).size != 2:
        raise ValueError('Test holdout must contain both classes')

    results = {}
    for name in names:
        p = np.concatenate(scores[name])
        threshold = thresholds[name]
        pred = (p >= threshold).astype('int8')
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        results[name] = {
            'pr_auc': float(average_precision_score(y, p)),
            'roc_auc': float(roc_auc_score(y, p)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'alert_rate_percent': float(pred.mean() * 100),
            'false_positives': int(fp),
            'true_positives': int(tp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'threshold': threshold,
            'test_rows': int(len(y)),
            'test_positives': int(y.sum()),
        }
        print(f'\n{name}:\n{json.dumps(results[name], indent=2)}')

    (OUT / 'test_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    comparison = pd.DataFrame.from_dict(results, orient='index')
    comparison.index.name = 'model'
    comparison.to_csv(OUT / 'test_model_comparison.csv')

    fig, ax = plt.subplots(figsize=(7, 4))
    comparison['pr_auc'].plot(kind='bar', ax=ax)
    ax.set_ylabel('Average precision (PR-AUC)')
    ax.set_xlabel('Model')
    ax.set_title('AML chronological holdout: PR-AUC')
    fig.tight_layout()
    fig.savefig(OUT / 'test_pr_auc_comparison.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    baseline = float(y.mean())
    ax.axhline(baseline, linestyle='--', label=f'Prevalence baseline: {baseline:.4%}')
    for name in names:
        precision, recall, _ = precision_recall_curve(y, np.concatenate(scores[name]))
        ax.plot(recall, precision, label=name)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('AML chronological holdout: precision-recall curves')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'test_precision_recall_curves.png', dpi=160)
    plt.close(fig)
    print(f'\nEvaluation complete. Outputs saved to {OUT}')
    print('Test thresholds were read from validation metadata and were not retuned.')


if __name__ == '__main__':
    main()
