import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

BACKEND_URL = "http://localhost:8000"

TRANSACTIONS = []
CUSTOMER_PROFILES = {}
AUDITOR_DECISIONS = {}

def reset_system_state():
    global TRANSACTIONS, CUSTOMER_PROFILES, AUDITOR_DECISIONS
    TRANSACTIONS.clear()
    CUSTOMER_PROFILES.clear()
    AUDITOR_DECISIONS.clear()

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
                    "lgb_confidence": f"{res['risk_score']}%",
                    "rule_confidence": "90%",
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
                "timestamp": payload["timestamp"],
                "payment_format": payload["payment_format"],
                "payment_currency": payload["payment_currency"],
                "risk": "Low",
                "risk_score": 15,
                "pattern": "SINGLE TRANSFER",
                "explanations": ["Model evaluation offline"],
                "gat_confidence": "15%",
                "lgb_confidence": "15%",
                "rule_confidence": "90%",
                "raw_res": {}
            }
            TRANSACTIONS.insert(0, tx_entry)
            last_res = tx_entry

    return last_res or TRANSACTIONS[0]

def get_metrics_summary() -> Dict[str, Any]:
    try:
        resp = requests.get(f"{BACKEND_URL}/api/dashboard/summary", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "total_processed": data.get("processed", len(TRANSACTIONS)),
                "today_added": "+ Real-time Live Stream",
                "high_risk": data.get("high_risk", sum(1 for t in TRANSACTIONS if t["risk"] == "High")),
                "medium_risk": data.get("medium_risk", sum(1 for t in TRANSACTIONS if t["risk"] == "Medium")),
                "low_risk": data.get("low_risk", sum(1 for t in TRANSACTIONS if t["risk"] == "Low"))
            }
    except Exception:
        pass
    
    return {
        "total_processed": len(TRANSACTIONS),
        "today_added": "+ Real-time Live Stream",
        "high_risk": sum(1 for t in TRANSACTIONS if t["risk"] == "High"),
        "medium_risk": sum(1 for t in TRANSACTIONS if t["risk"] == "Medium"),
        "low_risk": sum(1 for t in TRANSACTIONS if t["risk"] == "Low")
    }

def get_all_flagged_senders() -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{BACKEND_URL}/api/dashboard/summary", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            api_txs = data.get("recent_transactions", [])
            if api_txs:
                formatted = []
                for res in api_txs:
                    formatted.append({
                        "tx_id": res.get("transaction_id", "TX-001"),
                        "account": res.get("sender", "ACC_UNK"),
                        "sender_bank": res.get("sender_bank", "0"),
                        "receiver": res.get("receiver", "ACC_RECV"),
                        "receiver_bank": res.get("receiver_bank", "0"),
                        "amount": res.get("amount", 0.0),
                        "timestamp": res.get("timestamp", ""),
                        "payment_format": res.get("payment_format", "ACH"),
                        "payment_currency": res.get("payment_currency", "USD"),
                        "risk": str(res.get("risk_level", "Low")).capitalize(),
                        "risk_score": res.get("risk_score", 0.0),
                        "pattern": res.get("pattern", "SINGLE TRANSFER"),
                        "explanations": [
                            f"GAT Graph Topology: Anomaly score {res.get('explanation', {}).get('gat_anomaly_score', '0%')}",
                            f"Velocity Burst: {res.get('explanation', {}).get('recent_outgoing_count', 1)} transfers executed",
                            f"Historical Behavior: {res.get('explanation', {}).get('historical_behavior', 'normal')}"
                        ],
                        "gat_confidence": res.get("explanation", {}).get("gat_anomaly_score", "0%"),
                        "lgb_confidence": f"{res.get('risk_score', 0)}%",
                        "rule_confidence": "90%",
                        "raw_res": res
                    })
                return formatted
    except Exception:
        pass

    if not TRANSACTIONS:
        add_realtime_simulation_transaction([{
            "timestamp": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "from_bank": "70", "from_account": "ACC_INIT_01",
            "to_bank": "12", "to_account": "ACC_RECV_01",
            "amount_paid": 5000.0, "payment_currency": "USD", "payment_format": "ACH"
        }])
    return TRANSACTIONS

def get_transaction_by_id(tx_id: str) -> Dict[str, Any]:
    all_txs = get_all_flagged_senders()
    for t in all_txs:
        if t["tx_id"] == tx_id:
            return t
    if all_txs:
        return all_txs[0]
    return {
        "tx_id": tx_id, "account": "ACC_DEFAULT", "sender_bank": "0", "receiver": "ACC_RECV",
        "receiver_bank": "0", "amount": 1000.0, "timestamp": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "payment_format": "ACH", "risk": "Low", "risk_score": 10.0, "pattern": "SINGLE TRANSFER",
        "explanations": ["Initial system state"]
    }

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
            "Completed"
        ])
    if not rows:
        rows.append([tx_id, curr.get("receiver", "ACC_RECV"), f"${curr.get('amount', 1000.0):,.2f}", curr.get("timestamp", ""), "3 days", "Completed"])
    return rows

def get_customer_profile(acc_id: str) -> Dict[str, Any]:
    return {
        "name": f"Account {acc_id}",
        "type": "Corporate / Commercial" if "CORP" in str(acc_id) else "Personal Checking",
        "status": "Verified KYC" if "CORP" in str(acc_id) else "Standard Tier 1",
        "city": "New York, USA",
        "open_date": "2021-04-12",
        "account_id": acc_id
    }

def get_receiver_profile(acc_id: str) -> Dict[str, Any]:
    return get_customer_profile(acc_id)

def record_auditor_decision(tx_key: str, decision: str, notes: str = ""):
    AUDITOR_DECISIONS[tx_key] = {"decision": decision, "notes": notes}
    try:
        requests.post(f"{BACKEND_URL}/api/auditor/decision", json={"account": tx_key, "decision": decision}, timeout=3.0)
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
