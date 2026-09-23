import time
import requests

BACKEND_URL = "http://localhost:8000"

def test_backend():
    print("==================================================")
    print("VERIFYING FASTAPI GAT STREAMING BACKEND")
    print("==================================================")

    # 1. Check Root Endpoint
    resp = requests.get(f"{BACKEND_URL}/")
    assert resp.status_code == 200, f"Root endpoint failed: {resp.status_code}"
    print("[OK] Root Health Check passed:", resp.json())

    # 2. Test Stream 10 Sequential Transactions
    sample_txs = [
        {
            "transaction_id": f"TX-TEST-{i+1:03d}",
            "timestamp": "2022/09/01 10:00:00",
            "from_bank": "70",
            "account": "TEST_SENDER_01",
            "to_bank": "12",
            "receiver_account": f"TEST_RECEIVER_{i+1:02d}",
            "amount_received": 4500.0 + i * 100,
            "receiving_currency": "USD",
            "amount_paid": 4500.0 + i * 100,
            "payment_currency": "USD",
            "payment_format": "ACH"
        }
        for i in range(6)
    ]

    print("\nStreaming 6 sequential transactions for SENDER TEST_SENDER_01...")
    for idx, tx in enumerate(sample_txs):
        res = requests.post(f"{BACKEND_URL}/api/transactions/stream", json=tx)
        assert res.status_code == 200, f"Streaming failed at tx {idx}: {res.text}"
        data = res.json()
        print(f"  Tx #{idx+1} ({data['transaction_id']}): Score={data['risk_score']}/100 | Risk={data['risk_level']} | Pattern={data['pattern']} | OutCount={data['explanation']['recent_outgoing_count']}")

    # 3. Check Dashboard Summary Endpoint
    summary_resp = requests.get(f"{BACKEND_URL}/api/dashboard/summary")
    assert summary_resp.status_code == 200, f"Summary endpoint failed: {summary_resp.status_code}"
    sum_data = summary_resp.json()
    print("\n[OK] Dashboard Summary:", sum_data)
    assert sum_data["processed"] >= 6, "Expected at least 6 processed transactions"

    # 4. Test Auditor Approval Override
    auditor_resp = requests.post(f"{BACKEND_URL}/api/auditor/decision", json={"account": "TEST_SENDER_01", "decision": "Legitimate"})
    assert auditor_resp.status_code == 200, "Auditor endpoint failed"
    print("\n[OK] Auditor decision set to Legitimate for TEST_SENDER_01")

    # 5. Stream 7th transaction for TEST_SENDER_01 and verify score is overridden to LOW
    tx7 = {
        "transaction_id": "TX-TEST-007",
        "timestamp": "2022/09/01 10:15:00",
        "from_bank": "70",
        "account": "TEST_SENDER_01",
        "to_bank": "12",
        "receiver_account": "TEST_RECEIVER_07",
        "amount_received": 9900.0,
        "receiving_currency": "USD",
        "amount_paid": 9900.0,
        "payment_currency": "USD",
        "payment_format": "ACH"
    }
    res7 = requests.post(f"{BACKEND_URL}/api/transactions/stream", json=tx7).json()
    print(f"\n[OK] Tx #7 after Auditor Approval: Score={res7['risk_score']}/100 | Risk={res7['risk_level']} (Expected LOW / 12)")
    assert res7['risk_score'] == 12 and res7['risk_level'] == "LOW", f"Expected overridden score 12 LOW, got {res7['risk_score']} {res7['risk_level']}"

    print("\n==================================================")
    print("ALL BACKEND & STREAMING TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test_backend()
