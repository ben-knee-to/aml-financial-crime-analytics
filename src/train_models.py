from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import duckdb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'processed' / 'aml_features.parquet'
OUT = ROOT / 'reports' / 'modelling'
MODELS = ROOT / 'models'
TEMP = ROOT / 'data' / 'temp'
NUMERIC = ['transaction_hour', 'transaction_day_of_week', 'is_weekend', 'previous_transaction_count', 'hours_since_previous', 'log_abs_amount', 'negative_amount_flag']
CATEGORICAL = ['transaction_type', 'currency']
TRAIN_CAP = 250_000
VALIDATION_CAP = 200_000
SEED = 42


def select_split(con, start, end, cap, label):
    where = f"timestamp >= TIMESTAMP '{start}' AND timestamp < TIMESTAMP '{end}'"
    counts = con.execute(f"SELECT is_laundering, COUNT(*) FROM features WHERE {where} GROUP BY 1").fetchall()
    by_class = {int(k): int(v) for k, v in counts}
    if not by_class.get(0) or not by_class.get(1):
        raise ValueError(f'{label} does not contain both classes: {by_class}')
    total = sum(by_class.values())
    if total <= cap:
        query = f'SELECT * FROM features WHERE {where}'
    elif label == 'Validation':
        # Uniform deterministic sampling retains approximately natural prevalence.
        query = f'SELECT * FROM features WHERE {where} ORDER BY hash(transaction_key, {SEED}) LIMIT {cap}'
    else:
        positives = by_class[1]
        if positives >= cap:
            raise ValueError(f'{label}: positives ({positives}) exceed cap ({cap}); adjust strategy')
        negative_limit = cap - positives
        # Deterministic hash sampling of negatives; positives retained.
        query = f"""
            SELECT * FROM features WHERE {where} AND is_laundering=1
            UNION ALL
            SELECT * FROM (
                SELECT * FROM features WHERE {where} AND is_laundering=0
                ORDER BY hash(transaction_key, {SEED}) LIMIT {negative_limit}
            )
        """
    df = con.execute(query).df()
    if df.is_laundering.nunique() != 2:
        raise ValueError(f'{label} sample contains only one class; increase its cap')
    print(f'{label}: source={total:,}, selected={len(df):,}, positives={int(df.is_laundering.sum()):,}')
    return df


def threshold_f1(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def matrix(df):
    x = df[NUMERIC + CATEGORICAL].copy()
    for c in CATEGORICAL:
        x[c] = x[c].astype('string').fillna('Unknown').astype(str)
    return x


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'")
    con.execute('SET threads=2')
    escaped_temp = str(TEMP).replace("'", "''")
    con.execute(f"SET temp_directory='{escaped_temp}'")
    escaped_source = str(SOURCE).replace("'", "''")
    con.execute(f"CREATE VIEW features AS SELECT * FROM read_parquet('{escaped_source}')")
    columns = {row[0] for row in con.execute('DESCRIBE features').fetchall()}
    required = set(NUMERIC + CATEGORICAL + ['timestamp', 'transaction_key', 'is_laundering'])
    if missing := required - columns:
        raise ValueError(f'Missing feature columns: {sorted(missing)}')
    # Timestamp quantiles are approximate; all equal timestamps stay in one split.
    boundaries = con.execute('''
        SELECT approx_quantile(timestamp, 0.70), approx_quantile(timestamp, 0.85),
               MIN(timestamp), MAX(timestamp), COUNT(*) FROM features
    ''').fetchone()
    a, b, first, last, count = boundaries
    if not first < a < b <= last:
        raise ValueError('Invalid temporal split boundaries')
    print(f'Total rows: {count:,}; train < {a}; validation < {b}; test >= {b}')
    train = select_split(con, first, a, TRAIN_CAP, 'Training')
    val = select_split(con, a, b, VALIDATION_CAP, 'Validation')
    # Keep test completely untouched: evaluation script should query it separately.
    test_counts = con.execute('SELECT is_laundering, COUNT(*) FROM features WHERE timestamp >= ? GROUP BY 1', [b]).fetchall()
    test_by_class = {int(k): int(v) for k, v in test_counts}
    if not test_by_class.get(0) or not test_by_class.get(1):
        raise ValueError(f'Test split does not contain both classes: {test_by_class}')
    con.close()
    y = train.is_laundering.astype(int)
    y_val = val.is_laundering.astype(int)
    ratio = float((y == 0).sum()) / max(1, int((y == 1).sum()))
    candidates = {
        'random_forest': RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=2, random_state=SEED),
        'xgboost': XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=ratio, tree_method='hist', eval_metric='logloss', n_jobs=2, random_state=SEED),
    }
    scores = {}
    for name, estimator in candidates.items():
        prep = ColumnTransformer([
            ('num', SimpleImputer(strategy='median', add_indicator=True), NUMERIC),
            ('cat', Pipeline([
                ('fill', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=20))
            ]), CATEGORICAL)
        ])
        model = Pipeline([('preprocessor', prep), ('model', estimator)])
        print(f'Training {name}...')
        model.fit(matrix(train), y)
        p = model.predict_proba(matrix(val))[:, 1]
        scores[name] = {
            'pr_auc': float(average_precision_score(y_val, p)),
            'threshold': threshold_f1(y_val, p)
        }
        joblib.dump(model, MODELS / f'{name}.joblib')
        print(name, scores[name])
    best = max(scores, key=lambda k: scores[k]['pr_auc'])
    metadata = {
        'selected_model': best, 'validation_results': scores,
        'features': NUMERIC + CATEGORICAL,
        'numeric_features': NUMERIC, 'categorical_features': CATEGORICAL,
        'train_rows': len(train), 'validation_rows': len(val),
        'test_rows': sum(test_by_class.values()),
        'test_start': str(b), 'validation_end': str(b),
        'split_method': 'approximate timestamp quantiles; equal timestamps grouped',
        'validation_sampling': 'deterministic uniform subset when capped; preserves expected prevalence but adds sampling uncertainty',
        'warning': 'Synthetic data. Probabilities uncalibrated. Do not use validation-derived threshold as production threshold without prevalence-correct evaluation.'
    }
    (OUT / 'model_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Validation-selected model:', best)
    print('IMPORTANT: Test set remains untouched; adapt evaluate_models.py for DuckDB streaming evaluation.')

if __name__ == '__main__':
    main()
