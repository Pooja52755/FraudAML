import requests
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import Dict, Any, List

BACKEND_URL = "http://localhost:8000"

TRANSACTIONS = []
CUSTOMER_PROFILES = {}
AUDITOR_DECISIONS = {}
REVIEW_STATES = {"PENDING_REVIEW", "FLAGGED", "APPROVED", "ESCALATED"}
HOP2_PROFILES = {}

def reset_system_state():
    global TRANSACTIONS, CUSTOMER_PROFILES, AUDITOR_DECISIONS
    TRANSACTIONS.clear()
    CUSTOMER_PROFILES.clear()
    AUDITOR_DECISIONS.clear()
    try:
        import stream_engine
        stream_engine.reset_dataset_cursor()
    except ImportError:
        pass
    try:
        requests.post(f"{BACKEND_URL}/api/system/reset", timeout=5.0)
    except requests.RequestException:
        pass

import json
import os

DB_FILE = os.path.join(os.path.dirname(__file__), "auditor_decisions.json")

def _load_decisions():
    global AUDITOR_DECISIONS
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                loaded = json.load(f)
                AUDITOR_DECISIONS.update(loaded)
        except Exception:
            pass

def _save_decisions():
    try:
        with open(DB_FILE, "w") as f:
            json.dump(AUDITOR_DECISIONS, f, indent=4)
    except Exception:
        pass

def _timestamp_value(value: Any):
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed if not pd.isna(parsed) else None

def get_customer_accounts() -> List[str]:
    accounts = {tx.get("account") for tx in TRANSACTIONS if tx.get("account")}
    accounts.update(tx.get("receiver") for tx in TRANSACTIONS if tx.get("receiver"))
    try:
        for tx in get_all_flagged_senders():
            accounts.update(value for value in (tx.get("account"), tx.get("receiver")) if value)
    except Exception:
        pass
    return sorted(accounts)

def _profile_transactions(acc_id: str, as_of_timestamp: Any = None) -> List[Dict[str, Any]]:
    transactions = list(TRANSACTIONS)
    try:
        known_ids = {tx.get("tx_id") for tx in transactions}
        transactions.extend(tx for tx in get_all_flagged_senders() if tx.get("tx_id") not in known_ids)
    except Exception:
        pass
    cutoff = _timestamp_value(as_of_timestamp) if as_of_timestamp else None
    filtered = [
        tx for tx in transactions
        if tx.get("account") == acc_id or tx.get("receiver") == acc_id
    ]
    if cutoff is None:
        return filtered
    return [
        tx for tx in filtered
        if (tx_timestamp := _timestamp_value(tx.get("timestamp"))) is not None
        and tx_timestamp <= cutoff
    ]

def add_realtime_simulation_transaction(raw_txs: List[Dict[str, Any]]) -> Dict[str, Any]:
    global TRANSACTIONS
    last_res = None
    for tx in raw_txs:
        payload = {
            "transaction_id": tx.get("transaction_id", f"TX-SIM-{len(TRANSACTIONS)+1:05d}"),
            "timestamp": str(tx.get("timestamp", tx.get("Timestamp", datetime.now().strftime("%Y/%m/%d %H:%M:%S")))),
            "from_bank": str(tx.get("from_bank", tx.get("From Bank", "0"))),
            "account": str(tx.get("from_account", tx.get("Account", tx.get("account", "ACC_UNK")))),
            "to_bank": str(tx.get("to_bank", tx.get("To Bank", "0"))),
            "receiver_account": str(tx.get("to_account", tx.get("Account.1", tx.get("receiver_account", "ACC_RECV_UNK")))),
            "amount_received": float(tx.get("amount_received", tx.get("Amount Received", tx.get("amount_paid", 1000.0)))),
            "receiving_currency": str(tx.get("receiving_currency", tx.get("Receiving Currency", "USD"))),
            "amount_paid": float(tx.get("amount_paid", tx.get("Amount Paid", 1000.0))),
            "payment_currency": str(tx.get("payment_currency", tx.get("Payment Currency", "USD"))),
            "payment_format": str(tx.get("payment_format", tx.get("Payment Format", "ACH")))
        }

        try:
            resp = requests.post(f"{BACKEND_URL}/api/transactions/stream", json=payload, timeout=5.0)
            if resp.status_code == 200:
                res = resp.json()
                tx_entry = {
                    "tx_id": res["transaction_id"],
                    "account": res["sender"],
                    "sender_bank": res["sender_bank"],
                    "receiver": res["receiver"],
                    "receiver_bank": res["receiver_bank"],
                    "amount": res["amount"],
                    "amount_formatted": f"${float(res['amount']):,.2f}",
                    "total_outgoing_amount": res.get("total_outgoing_amount", res["amount"]),
                    "total_transactions": res.get("total_transactions", 1),
                    "review_state": res.get("review_state", "APPROVED"),
                    "review_reason": res.get("review_reason", ""),
                    "timestamp": res["timestamp"],
                    "payment_format": res["payment_format"],
                    "payment_currency": res["payment_currency"],
                    "risk": res["risk_level"].capitalize(),
                    "risk_score": res["risk_score"],
                    "pattern": res["pattern"],
                    "explanations": [
                        f"GAT Graph Topology: Anomaly score {res['explanation'].get('gat_anomaly_score', '85%')}",
                        f"Velocity Burst: {res['explanation'].get('recent_outgoing_count', 1)} transfers executed",
                        f"Historical Behavior: {res['explanation'].get('historical_behavior', 'normal')}"
                    ],
                    "gat_confidence": res['explanation'].get('gat_anomaly_score', '85%'),
                    "lgb_confidence": f"{float(res.get('calibrated_probability', res['risk_score'] / 100)) * 100:.2f}%",
                    "rule_confidence": f"{min(100.0, (float(res.get('total_transactions', 1)) / 10.0) * 100):.2f}%",
                    "model_used": "GAT Graph Model",
                    "raw_res": res
                }
                TRANSACTIONS.insert(0, tx_entry)
                last_res = tx_entry
        except Exception as e:
            # Fallback if backend API offline
            tx_entry = {
                "tx_id": payload["transaction_id"],
                "account": payload["account"],
                "sender_bank": payload["from_bank"],
                "receiver": payload["receiver_account"],
                "receiver_bank": payload["to_bank"],
                "amount": payload["amount_paid"],
                "amount_formatted": f"${payload['amount_paid']:,.2f}",
                "total_outgoing_amount": payload["amount_paid"],
                "total_transactions": 1,
                "review_state": "APPROVED",
                "review_reason": "Backend unavailable",
                "timestamp": payload["timestamp"],
                "payment_format": payload["payment_format"],
                "payment_currency": payload["payment_currency"],
                "risk": "Low",
                "risk_score": 15,
                "pattern": "SINGLE TRANSFER",
                "explanations": ["Model evaluation offline"],
                "gat_confidence": "15%",
                "lgb_confidence": "0%",
                "rule_confidence": "0%",
                "model_used": "Offline Fallback",
                "raw_res": {}
            }
            TRANSACTIONS.insert(0, tx_entry)
            last_res = tx_entry

    return last_res or TRANSACTIONS[0]

def get_metrics_summary() -> Dict[str, Any]:
    # Ensure transactions are pre-loaded
    if not TRANSACTIONS:
        get_all_flagged_senders()
        
    return {
        "total_processed": len(TRANSACTIONS),
        "today_added": "Static Dataset Analysis",
        "high_risk": sum(1 for t in TRANSACTIONS if t["risk"] == "High"),
        "medium_risk": sum(1 for t in TRANSACTIONS if t["risk"] == "Medium"),
        "low_risk": sum(1 for t in TRANSACTIONS if t["risk"] == "Low")
    }

def get_all_flagged_senders() -> List[Dict[str, Any]]:
    global TRANSACTIONS
    _load_decisions()
    
    if not TRANSACTIONS:
        csv_path = os.path.join(os.path.dirname(__file__), "Data", "perfect_100_test.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            tx_list = []
            for idx, row in df.iterrows():
                is_fraud = int(row.get('Actual Label', 0))
                gat_prob = float(row.get('GAT Probability', 0.15)) * 100
                gat_signal = str(row.get('GAT Signal', 'LOW RISK'))
                risk_level = "High" if "HIGH" in gat_signal else "Medium" if "MEDIUM" in gat_signal else "Low"
                tx_id = str(row.get('Transaction ID', f"TX-SIM-{idx+1:05d}"))
                acc = str(row.get('From Account', ''))
                
                state = "FLAGGED" if is_fraud else "APPROVED"
                reason = str(row.get('Detection Reason', 'Legitimate'))
                if acc in AUDITOR_DECISIONS:
                    state = AUDITOR_DECISIONS[acc].get("review_state", state)
                    reason = AUDITOR_DECISIONS[acc].get("decision", reason)
                elif tx_id in AUDITOR_DECISIONS:
                    state = AUDITOR_DECISIONS[tx_id].get("review_state", state)
                    reason = AUDITOR_DECISIONS[tx_id].get("decision", reason)

                tx_list.append({
                    "tx_id": tx_id,
                    "account": acc,
                    "sender_bank": str(row.get('From Bank Name', row.get('From Bank', ''))),
                    "receiver": str(row.get('To Account', '')),
                    "receiver_bank": str(row.get('To Bank Name', row.get('To Bank', ''))),
                    "amount": float(row.get('Amount Paid', 0.0)),
                    "amount_formatted": f"${float(row.get('Amount Paid', 0.0)):,.2f}",
                    "total_outgoing_amount": float(row.get('Amount Paid', 0.0)),
                    "total_transactions": int(row.get('previous_outgoing', 1)),
                    "review_state": state,
                    "review_reason": reason,
                    "timestamp": str(row.get('Timestamp', '')),
                    "payment_format": str(row.get('Payment Format', 'ACH')),
                    "payment_currency": str(row.get('Payment Currency', 'USD')),
                    "risk": risk_level,
                    "risk_score": gat_prob,
                    "pattern": str(row.get('Detection Pattern', 'SINGLE TRANSFER')),
                    "explanations": [str(row.get('Detection Reason', 'Model prediction based on historical data'))],
                    "gat_confidence": f"{gat_prob:.2f}%",
                    "lgb_confidence": f"{gat_prob:.2f}%",
                    "rule_confidence": "90.00%",
                    "model_used": "GAT Graph Model",
                    "raw_res": row.to_dict()
                })
            
            try:
                tx_list.sort(key=lambda x: pd.to_datetime(x["timestamp"]), reverse=True)
            except:
                pass
            TRANSACTIONS.extend(tx_list)
                
    return TRANSACTIONS

def get_transaction_by_id(tx_id: str) -> Dict[str, Any]:
    all_txs = get_all_flagged_senders()
    for t in all_txs:
        if t["tx_id"] == tx_id:
            return t
    if all_txs:
        return all_txs[0]
    return {}

def get_fan_out_rows(tx_id: str) -> List[List[Any]]:
    curr = get_transaction_by_id(tx_id)
    sender = curr.get("account", "ACC_UNK")
    matching = [t for t in TRANSACTIONS if t.get("account") == sender]
    rows = []
    for idx, t in enumerate(matching):
        rows.append([
            t["tx_id"],
            t["receiver"],
            f"${t['amount']:,.2f}",
            t["timestamp"],
            f"{(idx+1)*2} days",
            t.get("review_state", "PENDING_REVIEW")
        ])
    if not rows:
        rows.append([
            tx_id,
            curr.get("receiver", ""),
            f"${curr.get('amount', 0.0):,.2f}",
            curr.get("timestamp", ""),
            "3 days",
            curr.get("review_state", "PENDING_REVIEW")
        ])
    return rows

def create_network_graph(tx_id: str, include_2hop: bool = True) -> nx.DiGraph:
    """Build the selected transaction's graph from currently ingested real records."""
    graph = nx.DiGraph()
    selected = get_transaction_by_id(tx_id)
    if not selected:
        return graph

    sender = selected.get("account")
    if not sender:
        return graph
    cutoff = selected.get("timestamp")
    records = _profile_transactions(sender, cutoff)
    direct_receivers = {tx.get("receiver") for tx in records if tx.get("account") == sender and tx.get("receiver")}
    graph.add_node(sender, hop=0, node_type="source", color="#ef4444")

    for receiver in direct_receivers:
        receiver_rows = [tx for tx in records if tx.get("account") == sender and tx.get("receiver") == receiver]
        amount = sum(float(tx.get("amount", 0.0)) for tx in receiver_rows)
        graph.add_node(receiver, hop=1, node_type="receiver", color="#f59e0b")
        graph.add_edge(sender, receiver, hop=1, amount=f"${amount:,.2f}")

        if include_2hop:
            downstream = _profile_transactions(receiver, cutoff)
            for tx in downstream:
                if tx.get("account") != receiver or not tx.get("receiver") or tx.get("receiver") == sender:
                    continue
                downstream_node = tx["receiver"]
                graph.add_node(downstream_node, hop=2, node_type="downstream", color="#f1f5f9")
                graph.add_edge(receiver, downstream_node, hop=2, amount=f"${float(tx.get('amount', 0.0)):,.2f}")
    return graph

ACCOUNT_METADATA_CACHE = {}

def get_account_metadata(acc_id: str) -> Dict[str, str]:
    global ACCOUNT_METADATA_CACHE
    if not ACCOUNT_METADATA_CACHE:
        import os
        import pandas as pd
        meta_paths = ["Data/testing_trans.csv", "Data/testing_accounts.csv"]
        for p in meta_paths:
            path = os.path.join(os.path.dirname(__file__), p)
            if os.path.exists(path):
                df = pd.read_csv(path)
                if 'Account Number' in df.columns:
                    for _, row in df.iterrows():
                        anum = str(row.get('Account Number', '')).strip()
                        ACCOUNT_METADATA_CACHE[anum] = {
                            "Bank Name": str(row.get('Bank Name', '')),
                            "Bank ID": str(row.get('Bank ID', '')),
                            "Entity ID": str(row.get('Entity ID', '')),
                            "Entity Name": str(row.get('Entity Name', ''))
                        }
                    break
    return ACCOUNT_METADATA_CACHE.get(acc_id, {})

def get_customer_profile(acc_id: str, as_of_timestamp: Any = None) -> Dict[str, Any]:
    transactions = _profile_transactions(acc_id, as_of_timestamp)
    outgoing = [tx for tx in transactions if tx.get("account") == acc_id]
    incoming = [tx for tx in transactions if tx.get("receiver") == acc_id]
    outgoing_amount = sum(float(tx.get("amount", 0.0)) for tx in outgoing)
    incoming_amount = sum(float(tx.get("amount", 0.0)) for tx in incoming)
    total_transactions = len({tx.get("tx_id") for tx in outgoing + incoming})
    amounts = [float(tx.get("amount", 0.0)) for tx in transactions]
    risk_scores = [float(tx.get("risk_score", 0.0)) for tx in transactions]
    average_model_probability = (sum(risk_scores) / len(risk_scores) / 100.0) if risk_scores else 0.0
    velocity_factor = min(1.0, total_transactions / 10.0)
    high_risk_count = sum(1 for tx in transactions if float(tx.get("risk_score", 0.0)) >= 75.0)
    high_risk_factor = min(1.0, high_risk_count / 5.0)
    volume_factor = min(1.0, outgoing_amount / 100000.0)
    calibrated_probability = min(1.0, max(0.0, (
        average_model_probability * 0.55
        + velocity_factor * 0.20
        + high_risk_factor * 0.15
        + volume_factor * 0.10
    )))
    prior_amounts = [float(tx.get("amount", 0.0)) for tx in outgoing[:-1]]
    latest_amount = float(outgoing[-1].get("amount", 0.0)) if outgoing else 0.0
    prior_average = sum(prior_amounts) / len(prior_amounts) if prior_amounts else 0.0
    sudden_spike = bool(prior_average and latest_amount >= prior_average * 2.0)
    decision = AUDITOR_DECISIONS.get(acc_id, {})
    review_state = decision.get("review_state")
    if review_state not in REVIEW_STATES:
        review_state = "PENDING_REVIEW" if calibrated_probability >= 0.75 or sudden_spike else "APPROVED"
    risk_score = round(calibrated_probability * 100.0, 2)
    risk_tier = "High" if risk_score >= 75 else "Medium" if risk_score >= 45 else "Low"
    behavior_summary = [
        {"Metric": "Total Outgoing Amount", "Value": f"${outgoing_amount:,.2f}"},
        {"Metric": "Total Incoming Amount", "Value": f"${incoming_amount:,.2f}"},
        {"Metric": "Total Transactions", "Value": total_transactions},
        {"Metric": "High-Risk Flags", "Value": high_risk_count},
        {"Metric": "Calibrated Risk Score", "Value": f"{risk_score}/100"},
        {"Metric": "Review State", "Value": review_state},
    ]
    
    meta = get_account_metadata(acc_id)
    name = meta.get("Entity Name", f"Account {acc_id}")
    bank_name = meta.get("Bank Name", "Unknown Bank")
    
    return {
        "name": name,
        "account_id": acc_id,
        "account_type": "Corporate / Commercial" if "Company" in name or "Partnership" in name or "Proprietorship" in name else "Personal Checking",
        "kyc_status": "Verified",
        "city": bank_name,
        "open_since": "2021-04-12",
        "last_login": "Recent activity",
        "device_count": "2 Devices",
        "risk_tier": risk_tier,
        "risk_score": risk_score,
        "risk_probability": round(calibrated_probability, 6),
        "total_outgoing_amount": round(outgoing_amount, 2),
        "total_outgoing": f"${outgoing_amount:,.2f}",
        "total_incoming_amount": round(incoming_amount, 2),
        "total_incoming": f"${incoming_amount:,.2f}",
        "total_transactions": total_transactions,
        "high_risk_flag_count": high_risk_count,
        "unique_senders": len({tx.get("account") for tx in incoming}),
        "unique_receivers": len({tx.get("receiver") for tx in outgoing}),
        "avg_tx_amount": f"${(sum(amounts) / len(amounts) if amounts else 0.0):,.2f}",
        "sudden_outgoing_spike": sudden_spike,
        "review_state": review_state,
        "review_reason": "Risk threshold exceeded" if risk_score >= 75 else "Sudden outgoing amount spike" if sudden_spike else "Within monitoring threshold",
        "behavior_summary": behavior_summary,
    }

def get_receiver_profile(acc_id: str, as_of_timestamp: Any = None) -> Dict[str, Any]:
    return get_customer_profile(acc_id, as_of_timestamp)

def record_auditor_decision(tx_key: str, decision: str, notes: str = ""):
    state = "ESCALATED" if "Escalate" in decision else "APPROVED" if "Approve" in decision or decision == "Legitimate" else "FLAGGED"
    matching_account = next(
        (tx.get("account") for tx in TRANSACTIONS if tx.get("tx_id") == tx_key),
        tx_key,
    )
    decision_record = {"decision": decision, "notes": notes, "review_state": state, "timestamp": datetime.now().isoformat()}
    AUDITOR_DECISIONS[tx_key] = decision_record
    AUDITOR_DECISIONS[matching_account] = decision_record
    for tx in TRANSACTIONS:
        if tx.get("tx_id") == tx_key or tx.get("account") == matching_account:
            tx["review_state"] = state
            
    _save_decisions()
    
    try:
        requests.post(f"{BACKEND_URL}/api/auditor/decision", json={"account": matching_account, "decision": decision, "review_state": state, "notes": notes}, timeout=3.0)
    except Exception:
        pass

def update_auditor_decision(tx_id: str, decision: str, notes: str = ""):
    curr = get_transaction_by_id(tx_id)
    account = curr.get("account", tx_id)
    record_auditor_decision(account, decision, notes)

def get_transactions_df() -> pd.DataFrame:
    if not TRANSACTIONS:
        return pd.DataFrame()
    return pd.DataFrame(TRANSACTIONS)
