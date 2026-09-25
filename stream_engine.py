import os
import sys
import time
import json
import requests
import argparse
import pandas as pd
from datetime import datetime

BACKEND_URL = "http://localhost:8000/api/transactions/stream"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_dataset_rows = None
_dataset_accounts = {}
_dataset_index = 0

def reset_dataset_cursor() -> None:
    global _dataset_rows, _dataset_accounts, _dataset_index
    _dataset_rows = None
    _dataset_accounts = {}
    _dataset_index = 0

def generate_raw_transaction() -> dict:
    """Return the next chronological transaction from the bundled test dataset."""
    global _dataset_rows, _dataset_accounts, _dataset_index
    if _dataset_rows is None:
        _dataset_rows, _dataset_accounts = load_and_prepare_data()
        _dataset_index = 0
    if _dataset_rows.empty:
        raise RuntimeError("Data/testing_accounts.csv contains no transactions")
    row = _dataset_rows.iloc[_dataset_index % len(_dataset_rows)]
    row_index = _dataset_index
    _dataset_index += 1
    from_acc = str(row.get("From Account", row.get("Account", ""))).strip()
    to_acc = str(row.get("To Account", row.get("Account.1", ""))).strip()
    sender_meta = _dataset_accounts.get(from_acc, {})
    return {
        "transaction_id": f"TX-DATA-{row_index + 1:05d}",
        "timestamp": str(row.get("Timestamp", "")),
        "from_bank": str(row.get("From Bank", sender_meta.get("bank_id", "0"))),
        "from_account": from_acc,
        "to_bank": str(row.get("To Bank", "0")),
        "to_account": to_acc,
        "amount_paid": float(row.get("Amount Paid", 0.0)),
        "amount_received": float(row.get("Amount Received", 0.0)),
        "payment_currency": str(row.get("Payment Currency", "USD")),
        "receiving_currency": str(row.get("Receiving Currency", "USD")),
        "payment_format": str(row.get("Payment Format", "ACH")),
        "bank_name": sender_meta.get("bank_name", ""),
        "bank_id": sender_meta.get("bank_id", ""),
        "account_number": from_acc,
        "entity_id": sender_meta.get("entity_id", ""),
        "entity_name": sender_meta.get("entity_name", ""),
    }

def load_and_prepare_data(data_path: str = "Data/testing_accounts.csv", accounts_path: str = "Data/testing_trans.csv"):
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_DIR, data_path)
    if not os.path.isabs(accounts_path):
        accounts_path = os.path.join(PROJECT_DIR, accounts_path)

    # 1. Fallback / Auto-resolution if default paths don't exist
    if not os.path.exists(data_path):
        for candidate in ["Data/testing_accounts.csv", "Data/HI-Small_FANOUT_testing_data.csv", "Data/HI-Small_FANOUT_testing_data.csv.csv"]:
            candidate = os.path.join(PROJECT_DIR, candidate)
            if os.path.exists(candidate):
                data_path = candidate
                break

    if not os.path.exists(accounts_path):
        for candidate in ["Data/testing_trans.csv", "Data/HI-Small_FANOUT_testing_accounts.csv"]:
            candidate = os.path.join(PROJECT_DIR, candidate)
            if os.path.exists(candidate):
                accounts_path = candidate
                break

    print(f"Loading transaction dataset from: {data_path}")
    print(f"Loading account metadata from: {accounts_path}")

    df_a = pd.read_csv(data_path)
    df_b = pd.read_csv(accounts_path) if os.path.exists(accounts_path) else pd.DataFrame()

    # Smart Swap Detection: Check which DF contains transaction columns vs account metadata
    if 'Timestamp' in df_b.columns and 'Timestamp' not in df_a.columns:
        print("Detected swapped filenames. Auto-correcting transaction vs metadata assignments.")
        df_trans, df_acc = df_b, df_a
    elif 'Timestamp' in df_a.columns:
        df_trans, df_acc = df_a, df_b
    else:
        df_trans, df_acc = df_a, df_b

    # Build account lookup dictionary for metadata (Bank Name, Entity Name, etc.)
    acc_map = {}
    if not df_acc.empty and 'Account Number' in df_acc.columns:
        for _, acc_row in df_acc.iterrows():
            anum = str(acc_row.get('Account Number', '')).strip()
            acc_map[anum] = {
                "bank_name": str(acc_row.get('Bank Name', '')),
                "bank_id": str(acc_row.get('Bank ID', '')),
                "entity_id": str(acc_row.get('Entity ID', '')),
                "entity_name": str(acc_row.get('Entity Name', ''))
            }

    # Standardize transaction columns
    if "Account.1" in df_trans.columns and "To Account" not in df_trans.columns:
        df_trans["To Account"] = df_trans["Account.1"]
    if "Account" in df_trans.columns and "From Account" not in df_trans.columns:
        df_trans["From Account"] = df_trans["Account"]

    # Sort chronologically by original dataset Timestamp
    if "Timestamp" in df_trans.columns:
        df_trans["dt_temp"] = pd.to_datetime(df_trans["Timestamp"], errors="coerce")
        df_trans = df_trans.sort_values("dt_temp").drop(columns=["dt_temp"]).reset_index(drop=True)

    print(f"Successfully loaded {len(df_trans):,} transactions and {len(acc_map):,} account profiles ready for real-time streaming.")
    return df_trans, acc_map

def stream_transactions(df: pd.DataFrame, acc_map: dict, delay_sec: float = 1.0, max_tx: int = None):
    print("==================================================")
    print(f"STARTING AML REAL-TIME STREAM ENGINE")
    print(f"Target Endpoint: {BACKEND_URL}")
    print(f"Wall-Clock Interval: {delay_sec} seconds / tx")
    print(f"Preserving Original Dataset Timestamps: TRUE")
    print("==================================================\n")

    total_streamed = 0
    start_time = time.time()

    for idx, row in df.iterrows():
        if max_tx and total_streamed >= max_tx:
            print(f"\nReached max transactions limit ({max_tx}). Stopping stream.")
            break

        from_acc = str(row.get("From Account", row.get("Account", ""))).strip()
        to_acc = str(row.get("To Account", row.get("Account.1", ""))).strip()
        sender_meta = acc_map.get(from_acc, {})
        recv_meta = acc_map.get(to_acc, {})

        tx_payload = {
            "transaction_id": f"TX-SIM-{idx+1:05d}",
            "timestamp": str(row.get("Timestamp", "")),
            "from_bank": str(row.get("From Bank", sender_meta.get("bank_id", "0"))),
            "account": from_acc,
            "to_bank": str(row.get("To Bank", recv_meta.get("bank_id", "0"))),
            "receiver_account": to_acc,
            "amount_received": float(row.get("Amount Received", 0.0)),
            "receiving_currency": str(row.get("Receiving Currency", "USD")),
            "amount_paid": float(row.get("Amount Paid", 0.0)),
            "payment_currency": str(row.get("Payment Currency", "USD")),
            "payment_format": str(row.get("Payment Format", "ACH")),
            "bank_name": sender_meta.get("bank_name", "Global Trust Bank"),
            "bank_id": sender_meta.get("bank_id", "BNK-001"),
            "account_number": from_acc,
            "entity_id": sender_meta.get("entity_id", "ENT-001"),
            "entity_name": sender_meta.get("entity_name", "Individual Account")
        }

        try:
            resp = requests.post(BACKEND_URL, json=tx_payload, timeout=5.0)
            if resp.status_code == 200:
                res = resp.json()
                total_streamed += 1
                level = res.get("risk_level", "LOW")
                score = res.get("risk_score", 0)
                pattern = res.get("pattern", "SINGLE TRANSFER")
                sender = res.get("sender", from_acc)
                receiver = res.get("receiver", to_acc)
                amt = res.get("amount", 0.0)
                ts = res.get("timestamp", "")
                gat_prob = res.get("risk_probability", 0.0)
                confidence = res.get("gat_confidence", f"{gat_prob*100:.4f}%")
                graph_info = res.get("graph_state", {})
                nodes_cnt = graph_info.get("total_nodes_in_graph", 0)
                edges_cnt = graph_info.get("total_edges_in_graph", 0)
                prior_out = graph_info.get("sender_prior_outgoing_count", 0)

                icon = "[HIGH RISK]" if level == "HIGH" else "[MED RISK]" if level == "MEDIUM" else "[LOW RISK]"

                print(f"[{datetime.now().strftime('%H:%M:%S')}] #{total_streamed:04d} | {icon} Score: {float(score):.2f}/100 | GAT Confidence: {confidence} | {pattern}")
                print(f"   TxID: {res.get('transaction_id')} | {sender} -> {receiver} | ${amt:,.2f} | TS: {ts}")
                print(f"   Graph Memory State: {nodes_cnt} Nodes, {edges_cnt} Edges | Historical Sender Out Degree Prior to TX: {prior_out}")
                if level == "HIGH":
                    print(f"   [!] GAT Elevated Risk Alert: Pure GAT Sigmoid Output = {gat_prob*100:.4f}% ({gat_prob:.6f})")
                print("-" * 80)
            else:
                print(f"Backend returned HTTP {resp.status_code}: {resp.text}")

        except Exception as e:
            print(f"Connection error posting transaction to backend: {e}")

        time.sleep(delay_sec)

    elapsed = time.time() - start_time
    print(f"\nStream completed. Streamed {total_streamed} transactions in {elapsed:.1f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Real-Time Transaction Stream Engine")
    parser.add_argument("--data", type=str, default="Data/testing_accounts.csv", help="Path to testing transaction CSV")
    parser.add_argument("--accounts", type=str, default="Data/testing_trans.csv", help="Path to accounts metadata CSV")
    parser.add_argument("--delay", type=float, default=1.0, help="Wall clock delay in seconds between transactions")
    parser.add_argument("--max-tx", type=int, default=None, help="Maximum number of transactions to stream")

    args = parser.parse_args()

    df_trans, acc_map = load_and_prepare_data(args.data, args.accounts)

    try:
        stream_transactions(df_trans, acc_map, delay_sec=args.delay, max_tx=args.max_tx)
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user. Exiting cleanly.")
