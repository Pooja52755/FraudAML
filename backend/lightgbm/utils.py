# utils.py
import numpy as np
from sklearn.metrics import f1_score
from snapml import GraphFeaturePreprocessor


def increment_run_gfp(
    gfp: GraphFeaturePreprocessor, 
    data: np.ndarray, 
    chunk_size: int = 250_000,
    verbose: bool = False
) -> np.ndarray:
    """
    Run GFP transform incrementally over chunks of the transaction array.
    
    GFP maintains an internal graph state between calls to transform(), so
    chunking preserves continuity — edges from earlier chunks are visible
    when processing later ones. This is critical for cycle detection across
    chunk boundaries.
    
    Never use fit_transform() here — it inserts edges twice and corrupts
    the graph state.

    Args:
        gfp: A configured GraphFeaturePreprocessor instance.
        data: Transaction array of shape (n, 5) — [id, from, to, time, amount].
        chunk_size: Number of rows per chunk. Default 250k balances memory and speed.
        verbose: If True, prints the current row index at each chunk.

    Returns:
        Enriched array with GFP features appended as additional columns.
    """
    n = len(data)
    results = []
    for i in range(0, n, chunk_size):
        if verbose:
            print(f"Processing rows {i} to {min(i + chunk_size, n)}...")
        results.append(gfp.transform(data[i:i+chunk_size]))
    return np.concatenate(results, axis=0)



def temporal_split(X, y, n_original_cols, train_size_pct=0.6, val_size_pct=0.2):
    """
    Split features and target into train/val/test sets preserving chronological order.
    
    Also strips the first n_original_cols columns from X — these are the raw input
    columns passed to GFP (transactionID, from/to account, elapsed, amount) and
    should not be used as model features.

    Args:
        X: Full feature array of shape (n, n_original_cols + n_engineered_features).
        y: Target array of shape (n,).
        n_original_cols: Number of leading columns to drop (GFP input columns).
        train_size_pct: Fraction of data for training. Default 0.6.
        val_size_pct: Fraction of data for validation. Default 0.2.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    train_size = int(train_size_pct * len(X))
    val_size = int(val_size_pct * len(X))
    added_features = X.shape[1] - n_original_cols

    X_train = X[:train_size, -added_features:]
    X_val   = X[train_size:train_size + val_size, -added_features:]
    X_test  = X[train_size + val_size:, -added_features:]

    y_train = y[:train_size]
    y_val   = y[train_size:train_size + val_size]
    y_test  = y[train_size + val_size:]

    return X_train, X_val, X_test, y_train, y_val, y_test



def f1_eval(y_pred, dataset):
    """Custom LightGBM callback that returns minority class F1 for early stopping."""
    y_true = dataset.get_label()
    y_pred_binary = (y_pred >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_binary)
    return "f1", f1, True