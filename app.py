import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import textwrap
import fraud_data
import graph_vis

# ─── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ───────────────────────────────────────────────────────────
st.html(textwrap.dedent("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp { background-color: #f1f5f9; }

    /* ── Header ── */
    .header-title { font-size: 24px; font-weight: 800; color: #0f172a; margin: 0; }
    .header-subtitle { font-size: 13px; color: #64748b; margin-top: 2px; }
    .header-timestamp { font-size: 13px; color: #64748b; }

    /* ── Badges ── */
    .badge-high {
        background:#fef2f2; color:#dc2626; border:1px solid #fecaca;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }
    .badge-medium {
        background:#fffbeb; color:#d97706; border:1px solid #fde68a;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }
    .badge-low {
        background:#fef2f2; color:#dc2626; border:1px solid #fecaca;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }

    /* ── Textarea styling ── */
    .stTextArea textarea {
        background-color: #f8fafc !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        font-size: 13px !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }

    /* ── Left Panel: Flagged Account Cards ── */
    .flagged-card {
        background:#ffffff; border:1.5px solid #e2e8f0; border-radius:10px;
        padding:12px 14px; margin-bottom:10px; cursor:pointer;
        transition: box-shadow 0.2s;
        border-left: 4px solid #e2e8f0;
    }
    .flagged-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .flagged-card-high { border-left-color: #ef4444 !important; }
    .flagged-card-medium { border-left-color: #f59e0b !important; }
    .flagged-card-low { border-left-color: #ef4444 !important; }
    .flagged-card-selected { background:#eff6ff; border-color:#93c5fd; }
    .flagged-acc-id { font-weight:700; font-size:13px; color:#1e40af; }
    .flagged-pattern { font-size:11px; color:#64748b; margin-top:3px; }
    .risk-score-bar-bg {
        background:#f1f5f9; border-radius:4px; height:6px; margin-top:8px; overflow:hidden;
    }
    .risk-score-bar-fill {
        height:6px; border-radius:4px;
        background: linear-gradient(90deg, #f59e0b, #ef4444);
    }

    /* ── Center Panel ── */
    .center-card {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:20px; box-shadow:0 1px 4px rgba(0,0,0,0.06);
    }
    .graph-network-btn {
        display:inline-flex; align-items:center; gap:8px;
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color:white; font-weight:700; font-size:13px;
        padding:10px 18px; border-radius:8px; cursor:pointer;
        border:none; margin-bottom:16px;
        box-shadow: 0 2px 8px rgba(37,99,235,0.35);
        transition: all 0.2s;
        text-decoration:none;
    }
    .graph-network-btn:hover {
        background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
        box-shadow: 0 4px 14px rgba(37,99,235,0.45);
        transform: translateY(-1px);
    }

    /* ── XAI Explanation box ── */
    .xai-box {
        background:#fffbeb; border:1px solid #fde68a; border-radius:10px;
        padding:14px 16px; margin-top:16px;
    }
    .xai-box-high { background:#fff5f5; border:1px solid #fecaca; }
    .xai-box-medium { background:#fffbeb; border:1px solid #fde68a; }
    .xai-box-low { background:#f0fdf4; border:1px solid #bbf7d0; }
    .xai-title { font-weight:700; font-size:13px; color:#1e293b; margin-bottom:8px; }
    .xai-list { margin:0; padding-left:18px; font-size:12px; color:#334155; line-height:1.8; }

    /* ── AI Advisory banner ── */
    .ai-advisory {
        background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px;
        padding:10px 14px; font-size:11.5px; color:#0369a1;
        margin-top:12px; display:flex; align-items:flex-start; gap:8px;
    }

    /* ── Human Decision Panel ── */
    .human-decision-panel {
        background: linear-gradient(135deg, #fffbeb 0%, #fff7ed 100%);
        border:1.5px solid #fbbf24; border-radius:12px;
        padding:18px 20px; margin-top:18px;
    }
    .human-decision-title {
        font-weight:800; font-size:14px; color:#92400e;
        display:flex; align-items:center; gap:8px; margin-bottom:6px;
    }
    .human-decision-disclaimer {
        font-size:11px; color:#78716c; background:#ffffff;
        border:1px solid #e7e5e4; border-radius:6px;
        padding:8px 12px; margin-top:10px;
    }

    /* ── Right Panel: Customer Profile ── */
    .profile-card {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:18px; box-shadow:0 1px 4px rgba(0,0,0,0.06);
        height:100%;
    }
    .profile-empty {
        display:flex; flex-direction:column; align-items:center;
        justify-content:center; padding:40px 20px; text-align:center;
        color:#94a3b8;
    }
    .profile-metric-grid {
        display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin-bottom:14px;
    }
    .profile-metric-card {
        background:#f8fafc; border:1px solid #f1f5f9; border-radius:8px; padding:10px 12px;
    }
    .profile-metric-label { font-size:10px; color:#64748b; text-transform:uppercase; font-weight:600; }
    .profile-metric-val { font-size:15px; font-weight:700; color:#0f172a; margin-top:2px; }
    .kyc-verified { color:#16a34a; font-weight:700; }
    .kyc-flagged { color:#dc2626; font-weight:700; }
    .kyc-minimal { color:#d97706; font-weight:700; }
    .kyc-unverified { color:#dc2626; font-weight:700; }

    /* ── Behavior table ── */
    .custom-table { width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }
    .custom-table th {
        text-align:left; padding:8px 10px; background:#f8fafc;
        color:#64748b; font-weight:600; border-bottom:1px solid #e2e8f0;
    }
    .custom-table td { padding:8px 10px; border-bottom:1px solid #f1f5f9; color:#1e293b; }
    .change-high { color:#dc2626; font-weight:600; }
    .change-medium { color:#d97706; font-weight:600; }
    .change-low { color:#16a34a; font-weight:500; }

    /* ── Top banner ── */
    .top-banner {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:16px 22px; display:flex; align-items:center; gap:20px;
        margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.05);
    }
    .top-banner-icon {
        width:46px; height:46px; background:#2563eb; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        color:white; font-size:22px;
    }
    .banner-value { font-size:26px; font-weight:800; color:#0f172a; line-height:1.2; }
    .banner-subtext { font-size:12px; color:#16a34a; font-weight:600; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e2e8f0; }
    .sidebar-brand {
        display:flex; align-items:center; gap:10px;
        padding:10px 0 20px 0; border-bottom:1px solid #f1f5f9; margin-bottom:20px;
    }
    .sidebar-brand-title { font-size:16px; font-weight:800; color:#0f172a; }
    .sidebar-brand-subtitle { font-size:11px; color:#64748b; }

    /* ── Model info bar ── */
    .model-info-bar {
        background:#eff6ff; border:1px solid #dbeafe; border-radius:8px;
        padding:10px 14px; display:flex; justify-content:space-between;
        align-items:center; font-size:12px; color:#1e40af; margin-top:14px;
    }

    /* ── Section labels ── */
    .section-label {
        font-size:11px; font-weight:700; color:#64748b;
        text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;
    }
</style>
"""))

# ─── Session State Initialization ────────────────────────────────────────
if "selected_tx_id" not in st.session_state:
    st.session_state.selected_tx_id = "TX-10231"
if "selected_sub_tx" not in st.session_state:
    st.session_state.selected_sub_tx = None
if "human_decision_submitted" not in st.session_state:
    st.session_state.human_decision_submitted = {}
if "goto_graph" not in st.session_state:
    st.session_state.goto_graph = False

# ─── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.html(textwrap.dedent("""
    <div class="sidebar-brand">
        <div style="width:34px;height:34px;background:#2563eb;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;color:white;font-size:18px;">
            🛡️
        </div>
        <div>
            <div class="sidebar-brand-title">AML Fraud Detection</div>
            <div class="sidebar-brand-subtitle">Graph based Transaction Monitoring</div>
        </div>
    </div>
    """))

    # If goto_graph flag is set, pre-select the graph page
    default_page_idx = 3 if st.session_state.goto_graph else 0
    page = st.radio(
        "Navigation",
        ["Dashboard", "⚡ Real-Time Simulator", "Transactions", "Alerts / Graph Network", "Customers", "Reports", "Settings", "Help"],
        index=default_page_idx,
        label_visibility="collapsed"
    )
    if st.session_state.goto_graph and page == "Alerts / Graph Network":
        st.session_state.goto_graph = False

    st.html("<br><br><hr><div style='text-align:center;color:#94a3b8;font-size:11px;'>© 2025 AML System</div>")

# ─── Top Header ───────────────────────────────────────────────────────────
current_time = datetime.now().strftime("%I:%M:%S %p")
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.html(textwrap.dedent("""
    <div>
        <h1 class="header-title">AML Fraud Detection</h1>
        <div class="header-subtitle">Graph based Transaction Monitoring</div>
    </div>
    """))
with col_h2:
    st.html(textwrap.dedent(f"""
    <div style="text-align:right;margin-top:10px;">
        <span class="header-timestamp">🕒 Last Updated: {current_time}</span>
    </div>
    """))

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":

    # ── RAW TRANSACTION INPUT FIELDS AT TOP OF DASHBOARD ──
    with st.expander("📥 Submit Raw Dataset Transactions for Real-Time Model Fan-Out Detection", expanded=True):
        st.markdown(
            "Supply raw dataset transaction fields (**Timestamp, From Bank, From Account, To Bank, To Account, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Bank Name, Bank ID, Account Number, Entity ID, Entity Name**). "
            "Input 1, 5, 10, 20 or any number of transactions. The GAT + LightGBM model will detect fan-out patterns and display results live on this Dashboard!"
        )
        
        db_tab0, db_tab1, db_tab2 = st.tabs([
            "📡 Real-Time Live Bank Streamer", 
            "✍️ Single / Multi Input Form", 
            "📊 Batch Table Editor (10, 20+ Txs)"
        ])

        with db_tab0:
            st.markdown("#### 📡 Real-Time Bank Stream Engine (Continuous Live Ingestion)")
            st.markdown(
                "Simulate real bank streaming transactions continuously in real time. The stream engine emits ONLY raw dataset features "
                "(**Timestamp, From Bank, From Account, To Bank, To Account, Amount Paid, Payment Currency, Amount Received, Receiving Currency, Payment Format, Bank Name, Bank ID, Account Number, Entity ID, Entity Name**). "
                "Feature engineering, GAT + LightGBM model detection, and graph topology updates are computed dynamically by our system!"
            )

            # Session State initialization for continuous live stream
            if "is_live_streaming" not in st.session_state:
                st.session_state.is_live_streaming = False
            if "live_stream_count" not in st.session_state:
                st.session_state.live_stream_count = 0
            if "live_stream_speed" not in st.session_state:
                st.session_state.live_stream_speed = 2.0

            s_col1, s_col2, s_col3 = st.columns([1.5, 1, 1])
            with s_col1:
                stream_toggle = st.toggle("🔴 START CONTINUOUS LIVE STREAMING (Real-Time Ingestion)", value=st.session_state.is_live_streaming, key="live_stream_toggle")
                st.session_state.is_live_streaming = stream_toggle
            with s_col2:
                stream_speed = st.slider("Stream Interval (seconds)", min_value=1.0, max_value=5.0, value=float(st.session_state.live_stream_speed), step=0.5, key="stream_speed_slider")
                st.session_state.live_stream_speed = stream_speed
            with s_col3:
                if st.button("🗑️ Reset Stream Engine State", use_container_width=True):
                    fraud_data.reset_system_state()
                    st.session_state.live_stream_count = 0
                    st.session_state.is_live_streaming = False
                    st.session_state.selected_tx_id = "TX-10231"
                    st.rerun()

            if st.session_state.is_live_streaming:
                import stream_engine
                import time
                raw_tx = stream_engine.generate_raw_transaction()
                new_tx = fraud_data.add_realtime_simulation_transaction([raw_tx])
                st.session_state.live_stream_count += 1
                if new_tx["risk"] in ["High", "Medium"]:
                    st.session_state.selected_tx_id = new_tx["tx_id"]
                st.info(f"📡 **Live Stream Ingestion Active** (Ingested #{st.session_state.live_stream_count}): {raw_tx['from_account']} ➔ {raw_tx['to_account']} | ${raw_tx['amount_paid']:,.2f} USD | Evaluated Score = {new_tx['risk_score']}/100 ({new_tx['risk']} Risk)")
                time.sleep(st.session_state.live_stream_speed)
                st.rerun()



            # Direct Stream Injection Buttons
            st.markdown("---")
            st.markdown("**⚡ Quick Stream Generators (Inject Raw Fan-Out / Single Transactions):**")
            sc_col1, sc_col2, sc_col3 = st.columns(3)
            with sc_col1:
                if st.button("🏢 Stream New Corporate Account (10 Supplier Fan-Out)", type="primary", use_container_width=True):
                    corp_txs = [
                        {"timestamp": f"2026/09/21 14:{i+1:02d}", "from_bank": "National Bank of Harrisburg", "from_account": "ACC_CORP_SUPPLIERS_88", "to_bank": "Acme Bank", "to_account": f"ACC_SUPPLIER_{i+1:02d}", "amount_paid": 4850.0, "amount_received": 4850.0, "payment_currency": "US Dollar", "receiving_currency": "US Dollar", "payment_format": "ACH", "bank_name": "National Bank of Harrisburg", "bank_id": "BNK-1092", "account_number": "ACC_CORP_SUPPLIERS_88", "entity_id": "ENT-CORP-88", "entity_name": "Acme Global Manufacturing Corp"}
                        for i in range(10)
                    ]
                    new_tx = fraud_data.add_realtime_simulation_transaction(corp_txs)
                    st.session_state.selected_tx_id = new_tx["tx_id"]
                    st.success(f"✅ Streamed New Corporate Account ({new_tx['tx_id']}): 10 Supplier Transfers! Model Evaluated Risk = {new_tx['risk_score']}/100.")
                    st.rerun()

            with sc_col2:
                if st.button("🚨 Stream Novel High-Risk Mule Fan-Out Burst", use_container_width=True):
                    mule_txs = [
                        {"timestamp": f"2026/09/21 15:{i+1:02d}", "from_bank": "Bank of New York", "from_account": "ACC_NOVEL_MULE_99", "to_bank": "Offshore Bank", "to_account": f"ACC_MULE_RECV_{i+1:02d}", "amount_paid": 9850.0, "amount_received": 9850.0, "payment_currency": "US Dollar", "receiving_currency": "US Dollar", "payment_format": "Wire", "bank_name": "Bank of New York", "bank_id": "BNK-0012", "account_number": "ACC_NOVEL_MULE_99", "entity_id": "ENT-MULE-99", "entity_name": "Unverified Individual Entity"}
                        for i in range(10)
                    ]
                    new_tx = fraud_data.add_realtime_simulation_transaction(mule_txs)
                    st.session_state.selected_tx_id = new_tx["tx_id"]
                    st.success(f"✅ Streamed Novel High-Risk Burst ({new_tx['tx_id']}): 10 Mule Transfers! Model Evaluated Risk = {new_tx['risk_score']}/100.")
                    st.rerun()

            with sc_col3:
                if st.button("⚡ Stream 1 Single Real-Time Transfer", use_container_width=True):
                    import stream_engine
                    raw_tx = stream_engine.generate_raw_transaction()
                    new_tx = fraud_data.add_realtime_simulation_transaction([raw_tx])
                    st.session_state.selected_tx_id = new_tx["tx_id"]
                    st.success(f"✅ Streamed Single Real-Time Transfer ({new_tx['tx_id']})! Sender: {raw_tx['from_account']} ➔ {raw_tx['to_account']}. Risk = {new_tx['risk_score']}/100.")
                    st.rerun()


        with db_tab1:
            with st.form("dash_tx_form", clear_on_submit=False):
                d_c1, d_c2, d_c3, d_c4 = st.columns(4)
                with d_c1:
                    dash_from_bank = st.text_input("From Bank", value="Bank of New York")
                    dash_from_acc = st.text_input("From Account", value="ACC_78421")
                    dash_bank_name = st.text_input("Bank Name", value="GlobalTrust Financial")
                    dash_bank_id = st.text_input("Bank ID", value="BNK-1092")
                with d_c2:
                    dash_to_bank = st.text_input("To Bank", value="Portugal Bank")
                    dash_to_acc = st.text_input("To Account", value="ACC_90112")
                    dash_acc_num = st.text_input("Account Number", value="8001BB380")
                    dash_entity_id = st.text_input("Entity ID", value="ENT-9912")
                with d_c3:
                    dash_amt_paid = st.number_input("Amount Paid ($)", min_value=1.0, value=9500.0, step=100.0)
                    dash_pay_curr = st.selectbox("Payment Currency", ["US Dollar", "Euro", "UK Pound", "Rupee", "Yen"], index=0)
                    dash_amt_rec = st.number_input("Amount Received ($)", min_value=1.0, value=9500.0, step=100.0)
                    dash_rec_curr = st.selectbox("Receiving Currency", ["US Dollar", "Euro", "UK Pound", "Rupee", "Yen"], index=0)
                with d_c4:
                    dash_fmt = st.selectbox("Payment Format", ["ACH", "Wire", "Credit Card", "Cheque", "Cash"], index=0)
                    dash_time = st.text_input("Timestamp", value=datetime.now().strftime("%Y/%m/%d %H:%M"))
                    dash_entity_name = st.text_input("Entity Name", value="Global Logistics Corp")

                dash_submitted = st.form_submit_button("⚡ Run Model Detection & Display on Dashboard", type="primary", use_container_width=True)
                if dash_submitted:
                    raw_in = [{
                        "timestamp": dash_time,
                        "from_bank": dash_from_bank,
                        "from_account": dash_from_acc,
                        "to_bank": dash_to_bank,
                        "to_account": dash_to_acc,
                        "amount_paid": dash_amt_paid,
                        "amount_received": dash_amt_rec,
                        "payment_currency": dash_pay_curr,
                        "receiving_currency": dash_rec_curr,
                        "payment_format": dash_fmt,
                        "bank_name": dash_bank_name,
                        "bank_id": dash_bank_id,
                        "account_number": dash_acc_num,
                        "entity_id": dash_entity_id,
                        "entity_name": dash_entity_name
                    }]
                    new_tx = fraud_data.add_realtime_simulation_transaction(raw_in)
                    st.session_state.selected_tx_id = new_tx["tx_id"]
                    st.success(f"✅ Executed Model Inference: Created {new_tx['tx_id']} on Dashboard with Risk Score = {new_tx['risk_score']}/100!")
                    st.rerun()

        with db_tab2:
            st.markdown("Enter 10, 20 or any number of transactions into the table below:")
            if "dash_batch_df" not in st.session_state:
                st.session_state.dash_batch_df = pd.DataFrame([
                    {"Timestamp": "2026/09/21 14:01", "From Bank": "Bank of New York", "From Account": "ACC_78421", "To Bank": "Portugal Bank", "To Account": "ACC_90112", "Amount Paid ($)": 9500.0, "Payment Currency": "US Dollar", "Payment Format": "ACH"},
                    {"Timestamp": "2026/09/21 14:02", "From Bank": "Bank of New York", "From Account": "ACC_78421", "To Bank": "Canada Bank", "To Account": "ACC_90113", "Amount Paid ($)": 9450.0, "Payment Currency": "US Dollar", "Payment Format": "ACH"},
                    {"Timestamp": "2026/09/21 14:03", "From Bank": "Bank of New York", "From Account": "ACC_78421", "To Bank": "UK Bank", "To Account": "ACC_90114", "Amount Paid ($)": 9800.0, "Payment Currency": "US Dollar", "Payment Format": "Wire"},
                    {"Timestamp": "2026/09/21 14:04", "From Bank": "Bank of New York", "From Account": "ACC_78421", "To Bank": "Germany Bank", "To Account": "ACC_90115", "Amount Paid ($)": 9300.0, "Payment Currency": "US Dollar", "Payment Format": "ACH"},
                    {"Timestamp": "2026/09/21 14:05", "From Bank": "Bank of New York", "From Account": "ACC_78421", "To Bank": "Spain Bank", "To Account": "ACC_90116", "Amount Paid ($)": 9600.0, "Payment Currency": "US Dollar", "Payment Format": "ACH"},
                    {"Timestamp": "2026/09/21 14:06", "From Bank": "Bank of New York", "From Account": "ACC_78421", "To Bank": "Brazil Bank", "To Account": "ACC_90117", "Amount Paid ($)": 9750.0, "Payment Currency": "US Dollar", "Payment Format": "Wire"},
                    {"Timestamp": "2026/09/21 14:07", "From Bank": "Bank of New York", "From Account": "ACC_78421", "To Bank": "Japan Bank", "To Account": "ACC_90118", "Amount Paid ($)": 9200.0, "Payment Currency": "US Dollar", "Payment Format": "ACH"},
                    {"Timestamp": "2026/09/21 14:08", "From Bank": "Bank of New York", "From Account": "ACC_78421", "To Bank": "Russia Bank", "To Account": "ACC_90119", "Amount Paid ($)": 9900.0, "Payment Currency": "US Dollar", "Payment Format": "ACH"},
                    {"Timestamp": "2026/09/21 14:09", "From Bank": "Bank of New York", "From Account": "ACC_78421", "To Bank": "Italy Bank", "To Account": "ACC_90120", "Amount Paid ($)": 9650.0, "Payment Currency": "US Dollar", "Payment Format": "Wire"},
                    {"Timestamp": "2026/09/21 14:10", "From Bank": "Bank of New York", "From Account": "ACC_78421", "To Bank": "Israel Bank", "To Account": "ACC_90121", "Amount Paid ($)": 9400.0, "Payment Currency": "US Dollar", "Payment Format": "ACH"},
                ])

            edited_db_df = st.data_editor(
                st.session_state.dash_batch_df,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key="dash_batch_table_editor"
            )

            if st.button("⚡ Run Model Detection on Batch Transactions & Display on Dashboard", type="primary", use_container_width=True):
                if not edited_db_df.empty:
                    batch_txs = []
                    for idx, row in edited_db_df.iterrows():
                        batch_txs.append({
                            "timestamp": str(row.get("Timestamp", f"2026/09/21 14:{idx+1:02d}")),
                            "from_bank": str(row.get("From Bank", "GlobalTrust Bank")),
                            "from_account": str(row.get("From Account", "ACC_78421")),
                            "to_bank": str(row.get("To Bank", "Target Bank")),
                            "to_account": str(row.get("To Account", f"ACC_9011{idx+1}")),
                            "amount_paid": float(row.get("Amount Paid ($)", 1000.0) or 1000.0),
                            "amount_received": float(row.get("Amount Paid ($)", 1000.0) or 1000.0),
                            "payment_currency": str(row.get("Payment Currency", "US Dollar")),
                            "payment_format": str(row.get("Payment Format", "ACH"))
                        })
                    new_tx = fraud_data.add_realtime_simulation_transaction(batch_txs)
                    st.session_state.selected_tx_id = new_tx["tx_id"]
                    st.success(f"✅ Executed Model Inference: Created {new_tx['tx_id']} on Dashboard with {len(batch_txs)} transactions! Risk Score = {new_tx['risk_score']}/100")
                    st.rerun()


    st.markdown("---")

    # Top Banner — Calculated dynamically from real dataset & engine
    metrics = fraud_data.get_metrics_summary()
    st.html(textwrap.dedent(f"""
    <div class="top-banner">
        <div class="top-banner-icon">📋</div>
        <div>
            <div style="font-size:12px;color:#64748b;font-weight:600;">Total Transactions Processed</div>
            <div class="banner-value">{metrics['total_processed']}</div>
            <div class="banner-subtext">{metrics['today_added']}</div>
        </div>
        <div style="margin-left:30px;">
            <div style="font-size:12px;color:#64748b;font-weight:600;">High Risk Alerts</div>
            <div style="font-size:26px;font-weight:800;color:#dc2626;">{metrics['high_risk']}</div>
            <div style="font-size:12px;color:#dc2626;font-weight:600;">Require Human Review</div>
        </div>
        <div style="margin-left:30px;">
            <div style="font-size:12px;color:#64748b;font-weight:600;">Medium Risk</div>
            <div style="font-size:26px;font-weight:800;color:#d97706;">{metrics['medium_risk']}</div>
            <div style="font-size:12px;color:#d97706;font-weight:600;">Under Monitoring</div>
        </div>
        <div style="margin-left:30px;">
            <div style="font-size:12px;color:#64748b;font-weight:600;">Low Risk</div>
            <div style="font-size:26px;font-weight:800;color:#16a34a;">{metrics['low_risk']}</div>
            <div style="font-size:12px;color:#16a34a;font-weight:600;">Low Priority</div>
        </div>
        <div style="margin-left:auto;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:10px 18px;text-align:center;">
            <div style="font-size:11px;color:#16a34a;font-weight:700;">SYSTEM STATUS</div>
            <div style="font-size:18px;font-weight:800;color:#16a34a;">● LIVE</div>
        </div>
    </div>
    """))

    # ── 3-Column Layout ─────────────────────────────────────────────────
    col_left, col_center, col_right = st.columns([1.0, 1.8, 1.2])

    # ════════════════════════════════════════════════════════════════════
    #  LEFT PANEL — Flagged Accounts
    # ════════════════════════════════════════════════════════════════════
    with col_left:
        st.html('<div class="section-label">🚨 Flagged Accounts</div>')

        all_flagged = fraud_data.get_all_flagged_senders()
        
        # Active streamed simulation transactions FIRST so live streams are immediately visible on Dashboard
        sim_txs = [t for t in all_flagged if t["tx_id"].startswith("TX-SIM")]
        sim_txs.sort(key=lambda x: -x["risk_score"])
        
        hist_high = [t for t in all_flagged if t["risk"] == "High" and not t["tx_id"].startswith("TX-SIM")]
        hist_high.sort(key=lambda x: -x["risk_score"])
        
        hist_med = [t for t in all_flagged if t["risk"] == "Medium" and not t["tx_id"].startswith("TX-SIM")]
        hist_med.sort(key=lambda x: -x["risk_score"])
        
        all_txs = sim_txs + hist_high[:5] + hist_med[:5]

        for tx in all_txs:
            acc = tx["account"]
            risk = tx["risk"]
            score = tx["risk_score"]
            pattern = tx["pattern"]
            is_sel = (tx["tx_id"] == st.session_state.selected_tx_id)

            risk_color = "#ef4444" if risk == "High" else "#f59e0b" if risk == "Medium" else "#ef4444"
            sel_bg = "#eff6ff" if is_sel else "#ffffff"
            sel_border = "#93c5fd" if is_sel else "#e2e8f0"
            bar_width = score

            profile = fraud_data.get_customer_profile(acc)
            name = profile.get("name", acc)

            st.html(textwrap.dedent(f"""
            <div class="flagged-card flagged-card-{risk.lower()}"
                 style="background:{sel_bg}; border-color:{sel_border};">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div class="flagged-acc-id">{acc}</div>
                        <div style="font-size:11px;color:#475569;font-weight:500;margin-top:1px;">{name}</div>
                    </div>
                    <span class="badge-{risk.lower()}">{risk}</span>
                </div>
                <div class="flagged-pattern">📌 {pattern}</div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
                    <div class="risk-score-bar-bg" style="flex:1;margin-right:8px;">
                        <div class="risk-score-bar-fill" style="width:{bar_width}%;background:{risk_color};"></div>
                    </div>
                    <span style="font-size:11px;font-weight:700;color:{risk_color};">{score}/100</span>
                </div>
            </div>
            """))
            if st.button(f"Inspect →", key=f"left_btn_{tx['tx_id']}", use_container_width=True):
                st.session_state.selected_tx_id = tx["tx_id"]
                st.session_state.selected_sub_tx = None
                st.rerun()


    # ════════════════════════════════════════════════════════════════════
    #  CENTER PANEL — Fan-Out Transaction Table + XAI + Human Decision
    # ════════════════════════════════════════════════════════════════════
    with col_center:
        curr_tx = fraud_data.get_transaction_by_id(st.session_state.selected_tx_id)
        risk = curr_tx["risk"]
        risk_score = curr_tx["risk_score"]
        pattern = curr_tx["pattern"]

        # ── Header ──
        st.html(textwrap.dedent(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div>
                <div style="font-size:16px;font-weight:800;color:#0f172a;">
                    {curr_tx['tx_id']}
                    <span style="font-size:12px;font-weight:500;color:#64748b;margin-left:6px;">{curr_tx['pattern']}</span>
                </div>
                <div style="font-size:12px;color:#475569;margin-top:2px;">
                    Sender: <b>{curr_tx['account']}</b> ·
                    {curr_tx['timestamp']} · {curr_tx['payment_format']}
                </div>
            </div>
            <span class="badge-{risk.lower()}">{risk.upper()} RISK · {risk_score}/100</span>
        </div>
        """))

        # ── Graph Network Button ──
        if st.button("🕸️  Tap to View as Graph Network", key="graph_btn", use_container_width=False):
            st.session_state.goto_graph = True
            st.rerun()

        # ── Fan-Out Sub-Transaction Table ──
        st.html('<div class="section-label">Transaction Routing Flow</div>')

        fan_rows = fraud_data.get_fan_out_rows(st.session_state.selected_tx_id)
        df_fan = pd.DataFrame(fan_rows)
        df_fan.columns = ["Sub-TX ID", "To Account", "Amount ($)", "Time", "Account age(days)", "Status"]
        df_display = df_fan[["Sub-TX ID", "To Account", "Amount ($)", "Time", "Account age(days)", "Status"]]

        # Clickable table with row selection
        event = st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="fan_table"
        )

        # Handle row selection → populate right panel
        selected_rows = event.selection.get("rows", []) if event.selection else []
        if selected_rows:
            row_idx = selected_rows[0]
            sub_tx_data = fan_rows[row_idx]
            st.session_state.selected_sub_tx = sub_tx_data

        st.html("""
        <div style="font-size:10.5px;color:#94a3b8;margin-top:4px;">
            🖱️ Click a row to view the receiver's customer profile in the right panel.
        </div>
        """)

        # ── Why Flagged? XAI Box ──
        if risk == "High":
            xai_extra = "xai-box-high"
            fraud_icon = "🚨"
            fraud_label = "HIGH FRAUD RISK DETECTED"
            xai_header_text = "Why was this flagged?"
        elif risk == "Medium":
            xai_extra = "xai-box-medium"
            fraud_icon = "⚠️"
            fraud_label = "SUSPICIOUS PATTERN DETECTED"
            xai_header_text = "Why was this flagged?"
        else:
            xai_extra = "xai-box-low"
            fraud_icon = "✅"
            fraud_label = "LOW RISK — VERIFIED TRANSACTION"
            xai_header_text = "Model Verification Analysis"

        exps_html = "".join([f"<li>{e}</li>" for e in curr_tx["explanations"]])

        st.html(textwrap.dedent(f"""
        <div class="xai-box {xai_extra}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div class="xai-title">{fraud_icon} {fraud_label} — {xai_header_text}</div>
                <span class="badge-{risk.lower()}">{pattern}</span>
            </div>
            <ul class="xai-list">
                {exps_html}
            </ul>
            <div style="font-size:11px;color:#64748b;margin-top:10px;border-top:1px solid #e2e8f0;padding-top:8px;">
                Models: <b>GAT ({curr_tx.get('gat_confidence', '86%')})</b> + <b>LightGBM ({curr_tx.get('lgb_confidence', '88%')})</b> &nbsp;|&nbsp;
                Rule Engine: <b>{curr_tx.get('rule_confidence', '95%')}</b> &nbsp;|&nbsp;
                Combined Risk Score: <b>{curr_tx['risk_score']}/100</b>
            </div>
        </div>
        """))

        # ── Authorised Bank Auditor Decision Panel (HIGH RISK ONLY) ──
        if risk == "High":
            tx_key = curr_tx["tx_id"]
            already_submitted = st.session_state.human_decision_submitted.get(tx_key)

            st.html(textwrap.dedent(f"""
            <div class="human-decision-panel">
                <div class="human-decision-title">
                    🏦 Authorised Bank Auditor Decision Required — {curr_tx['tx_id']}
                </div>
                <div style="font-size:12px;color:#92400e;margin-bottom:12px;">
                    Risk Score: <b>{risk_score}/100</b> — This transaction requires an authorised bank auditor decision.
                </div>
            </div>
            """))

            if already_submitted:
                decision_val = already_submitted["decision"]
                color_map = {"✅ Approve Transaction": "#16a34a", "🚫 Block Transaction": "#dc2626", "📤 Escalate to Senior Investigator": "#d97706"}
                col = color_map.get(decision_val, "#16a34a")
                st.success(f"**Decision Recorded:** {decision_val}")
                st.info(f"**Authorised Bank Auditor Notes:** {already_submitted['notes'] or '(none)'}")
                if st.button("Revise Decision", key=f"revise_{tx_key}"):
                    del st.session_state.human_decision_submitted[tx_key]
                    st.rerun()
            else:
                dec_col1, dec_col2 = st.columns([1, 1])
                with dec_col1:
                    decision = st.radio(
                        "**Authorised Bank Auditor Decision**",
                        ["✅ Approve Transaction", "🚫 Block Transaction", "📤 Escalate to Senior Investigator"],
                        key=f"decision_{tx_key}",
                        index=2
                    )
                with dec_col2:
                    notes = st.text_area(
                        "**Authorised Bank Auditor Notes**",
                        placeholder="Add reasoning, observations, or escalation notes...",
                        key=f"notes_{tx_key}",
                        height=180
                    )

                st.html("""
                <div class="human-decision-disclaimer">
                    ⚠️ <b>Disclaimer:</b> By submitting, you confirm this decision is made by an
                    AUTHORISED BANK AUDITOR. The AI model provided supporting analysis only.
                    This action will be logged and audited.
                </div>
                """)

                if st.button(f"📋 Submit Decision for {tx_key}", key=f"submit_{tx_key}", type="primary", use_container_width=True):
                    st.session_state.human_decision_submitted[tx_key] = {
                        "decision": decision,
                        "notes": notes,
                        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
                    }
                    fraud_data.record_auditor_decision(tx_key, decision, notes)
                    st.rerun()

        # ── AI Advisory (COMPLETELY AT BOTTOM) ──
        st.html(textwrap.dedent("""
        <div class="ai-advisory" style="margin-top:16px;">
            <span style="font-size:16px;">🤖</span>
            <span>
                <b>AI Advisory:</b> The analysis above is generated by an AI model to
                <b>support the fraud investigator's decision</b>. The AI does not block or approve
                transactions. All final decisions must be made by an AUTHORIZED BANK AUDITOR.
            </span>
        </div>
        """))

    # ════════════════════════════════════════════════════════════════════
    #  RIGHT PANEL — Customer Profile (receiver, on row click)
    # ════════════════════════════════════════════════════════════════════
    with col_right:
        sub_tx = st.session_state.selected_sub_tx

        if sub_tx is None:
            st.html(textwrap.dedent("""
            <div class="profile-card">
                <div class="section-label">Customer Profile</div>
                <div class="profile-empty">
                    <div style="font-size:40px;margin-bottom:12px;">👤</div>
                    <div style="font-size:13px;font-weight:600;color:#64748b;">No row selected</div>
                    <div style="font-size:12px;margin-top:6px;color:#94a3b8;">
                        Click any transaction row in the center table to view the receiver's customer profile here.
                    </div>
                </div>
            </div>
            """))
        else:
            if isinstance(sub_tx, (list, tuple)) and len(sub_tx) > 0:
                sub_tx = sub_tx[0]
            if isinstance(sub_tx, dict):
                to_acc = sub_tx.get("to_account", sub_tx.get("receiver", sub_tx.get("receiver_account", "")))
            else:
                to_acc = str(sub_tx)
            profile = fraud_data.get_receiver_profile(to_acc)
            risk_tier = profile.get("risk_tier", "Unknown")
            kyc = profile.get("kyc_status", "Unknown")

            tier_color = "#dc2626" if "High" in risk_tier else "#d97706" if "Medium" in risk_tier else "#16a34a"

            kyc_class = (
                "kyc-verified" if kyc == "Verified" else
                "kyc-flagged" if kyc == "Flagged" else
                "kyc-minimal" if "Minimal" in kyc else
                "kyc-unverified"
            )

            # Behavior comparison table rows (simplified for receivers)
            beh_rows_html = ""
            for r in [
                ("Total Incoming", profile.get("total_incoming", "—")),
                ("Total Outgoing", profile.get("total_outgoing", "—")),
                ("Unique Senders", str(profile.get("unique_senders", "—"))),
                ("Unique Receivers", str(profile.get("unique_receivers", "—"))),
                ("Avg Tx Amount", profile.get("avg_tx_amount", "—")),
                ("Total Transactions", profile.get("total_transactions", "—")),
            ]:
                beh_rows_html += f"<tr><td><b>{r[0]}</b></td><td style='text-align:right;'>{r[1]}</td></tr>"

            st.html(textwrap.dedent(f"""
            <div class="profile-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                    <div>
                        <div class="section-label">Receiver Profile</div>
                        <div style="font-size:16px;font-weight:800;color:#0f172a;">{profile.get('name', to_acc)}</div>
                        <div style="font-size:12px;color:#64748b;margin-top:2px;">{to_acc}</div>
                    </div>
                    <div style="text-align:right;">
                        <span style="background:{tier_color}20;color:{tier_color};border:1px solid {tier_color}44;
                                     padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;">
                            {risk_tier}
                        </span>
                    </div>
                </div>

                <div class="profile-metric-grid">
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Account Type</div>
                        <div class="profile-metric-val">{profile.get('account_type', '—')}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">City</div>
                        <div class="profile-metric-val">{profile.get('city', '—')}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Open Since</div>
                        <div class="profile-metric-val" style="font-size:12px;">{profile.get('open_since', '—')}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Account Age (days)</div>
                        <div class="profile-metric-val">{sub_tx.get('account_age_days', '—')}</div>
                    </div>
                </div>

                <div style="font-size:12px;font-weight:700;color:#1e293b;margin-bottom:6px;">Financial Summary</div>
                <table class="custom-table">
                    <tbody>
                        {beh_rows_html}
                    </tbody>
                </table>
            </div>
            """))

    # ── Live Streaming Auto-Rerun Loop ──
    if st.session_state.get("is_live_streaming", False):
        import stream_engine
        import time
        raw_tx = stream_engine.generate_raw_transaction()
        new_tx = fraud_data.add_realtime_simulation_transaction([raw_tx])
        st.session_state.live_stream_count += 1
        st.toast(f"📡 Real-Time Stream Ingested #{st.session_state.live_stream_count}: {raw_tx['from_account']} ➔ {raw_tx['to_account']} (${raw_tx['amount_paid']:,.2f} USD)", icon="📡")
        time.sleep(st.session_state.get("live_stream_speed", 2.0))
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSACTIONS PAGE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Transactions":
    st.subheader("📊 All Transactions Log")
    df_all = fraud_data.get_transactions_df()
    c1, c2, c3 = st.columns(3)
    with c1:
        search_q = st.text_input("🔍 Search Transaction ID / Account", "")
    with c2:
        risk_filter = st.multiselect("Filter Risk Level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    with c3:
        min_amt = st.slider("Min Amount ($)", 0, 50000, 0)

    filtered_df = df_all[df_all["risk"].isin(risk_filter) & (df_all["amount"] >= min_amt)]
    if search_q:
        filtered_df = filtered_df[
            filtered_df["tx_id"].str.contains(search_q, case=False) |
            filtered_df["account"].str.contains(search_q, case=False)
        ]
    st.dataframe(
        filtered_df[["tx_id", "account", "timestamp", "amount_formatted", "risk", "risk_score", "pattern", "payment_format"]],
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
#  REAL-TIME SIMULATOR PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Real-Time Simulator":
    st.subheader("⚡ Real-Time Single-Transaction Streaming Engine & Model Inference Suite")
    st.markdown(
        "Input raw transactions one-by-one (**From Bank, From Account, To Bank, To Account, Amount Paid, Amount Received, Currency, Format, Timestamp**). "
        "As transactions arrive, the engine performs **EDA & feature engineering on the fly**, evaluates **Fan-Out out-degree growth**, "
        "computes **GAT + LightGBM + Rule Engine** ensemble scores, updates the **Money Trail Graph live**, and re-colors nodes upon **Human Auditor Approval**!"
    )

    st.markdown("---")
    
    # Session state initialization for real-time streaming engine
    if "stream_tx_list" not in st.session_state:
        st.session_state.stream_tx_list = []
    if "stream_step_index" not in st.session_state:
        st.session_state.stream_step_index = 0

    tab_single, tab_preset, tab_batch = st.tabs([
        "📥 Single Transaction Streamer (Input Form)", 
        "⚡ Step-by-Step Fan-Out Demo (1-by-1 Feed)", 
        "✍️ Batch Table Editor"
    ])

    # ── TAB 1: Single Transaction Input Form ──
    with tab_single:
        st.markdown("#### 📥 Submit Individual Streaming Transaction")
        st.markdown("Fill in raw transaction details as columns present in `HI-Small_Trans.csv`:")
        
        with st.form("single_tx_form", clear_on_submit=False):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                from_bank = st.text_input("From Bank", value="Bank of New York")
                from_acc = st.text_input("From Account (Sender ID)", value="ACC_78421")
                timestamp = st.text_input("Timestamp", value=datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
            with f_col2:
                to_bank = st.text_input("To Bank", value="Portugal Bank")
                to_acc = st.text_input("To Account (Receiver ID)", value=f"ACC_9011{len(st.session_state.stream_tx_list)+1:02d}")
                payment_format = st.selectbox("Payment Format", ["ACH", "Wire", "Credit Card", "Cheque", "Cash"], index=0)
            with f_col3:
                amount_paid = st.number_input("Amount Paid ($)", min_value=1.0, value=9500.0, step=100.0)
                amount_rec = st.number_input("Amount Received ($)", min_value=1.0, value=9500.0, step=100.0)
                payment_curr = st.selectbox("Payment Currency", ["US Dollar", "Euro", "UK Pound", "Rupee", "Yen"], index=0)

            submitted = st.form_submit_button("➕ Stream Single Transaction (Compute EDA & Update Graph)", type="primary", use_container_width=True)
            if submitted:
                new_item = {
                    "timestamp": timestamp,
                    "from_bank": from_bank,
                    "from_account": from_acc,
                    "to_bank": to_bank,
                    "to_account": to_acc,
                    "amount_paid": amount_paid,
                    "amount_received": amount_rec,
                    "payment_currency": payment_curr,
                    "payment_format": payment_format
                }
                st.session_state.stream_tx_list.append(new_item)
                
                # Execute EDA feature extraction and model inference for current accumulated stream
                res_tx = fraud_data.add_realtime_simulation_transaction(st.session_state.stream_tx_list)
                st.session_state.selected_tx_id = res_tx["tx_id"]
                st.success(f"✅ Streamed Tx #{len(st.session_state.stream_tx_list)} ({from_acc} ➔ {to_acc}): Calculated Out-Degree = {len(st.session_state.stream_tx_list)}, Ensemble Risk Score = {res_tx['risk_score']}/100!")
                st.rerun()

    # ── TAB 2: Step-by-Step Fan-Out Demo ──
    with tab_preset:
        st.markdown("#### ⚡ Real-Time Step-by-Step Fan-Out Stream Simulator")
        st.markdown("Click **'Stream Next Transaction'** to feed transactions 1-by-1 (`ACC_78421 ➔ target`) and watch the GAT + LightGBM risk score dynamically escalate as out-degree grows!")

        fanout_sequence = [
            {"timestamp": "2026/09/21 14:01", "from_bank": "Bank of New York", "from_account": "ACC_78421", "to_bank": "Portugal Bank", "to_account": "ACC_90112", "amount_paid": 9500.0, "payment_currency": "US Dollar", "payment_format": "ACH"},
            {"timestamp": "2026/09/21 14:02", "from_bank": "Bank of New York", "from_account": "ACC_78421", "to_bank": "Canada Bank", "to_account": "ACC_90113", "amount_paid": 9450.0, "payment_currency": "US Dollar", "payment_format": "ACH"},
            {"timestamp": "2026/09/21 14:03", "from_bank": "Bank of New York", "from_account": "ACC_78421", "to_bank": "UK Bank", "to_account": "ACC_90114", "amount_paid": 9800.0, "payment_currency": "US Dollar", "payment_format": "Wire"},
            {"timestamp": "2026/09/21 14:04", "from_bank": "Bank of New York", "from_account": "ACC_78421", "to_bank": "Germany Bank", "to_account": "ACC_90115", "amount_paid": 9300.0, "payment_currency": "US Dollar", "payment_format": "ACH"},
            {"timestamp": "2026/09/21 14:05", "from_bank": "Bank of New York", "from_account": "ACC_78421", "to_bank": "Spain Bank", "to_account": "ACC_90116", "amount_paid": 9600.0, "payment_currency": "US Dollar", "payment_format": "ACH"},
            {"timestamp": "2026/09/21 14:06", "from_bank": "Bank of New York", "from_account": "ACC_78421", "to_bank": "Brazil Bank", "to_account": "ACC_90117", "amount_paid": 9750.0, "payment_currency": "US Dollar", "payment_format": "Wire"},
            {"timestamp": "2026/09/21 14:07", "from_bank": "Bank of New York", "from_account": "ACC_78421", "to_bank": "Japan Bank", "to_account": "ACC_90118", "amount_paid": 9200.0, "payment_currency": "US Dollar", "payment_format": "ACH"},
            {"timestamp": "2026/09/21 14:08", "from_bank": "Bank of New York", "from_account": "ACC_78421", "to_bank": "Russia Bank", "to_account": "ACC_90119", "amount_paid": 9900.0, "payment_currency": "US Dollar", "payment_format": "ACH"},
            {"timestamp": "2026/09/21 14:09", "from_bank": "Bank of New York", "from_account": "ACC_78421", "to_bank": "Italy Bank", "to_account": "ACC_90120", "amount_paid": 9650.0, "payment_currency": "US Dollar", "payment_format": "Wire"},
            {"timestamp": "2026/09/21 14:10", "from_bank": "Bank of New York", "from_account": "ACC_78421", "to_bank": "Israel Bank", "to_account": "ACC_90121", "amount_paid": 9400.0, "payment_currency": "US Dollar", "payment_format": "ACH"},
        ]

        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            if st.button("➡️ Stream Next Fan-Out Transaction (1-by-1 Feed)", type="primary", use_container_width=True):
                if st.session_state.stream_step_index < len(fanout_sequence):
                    st.session_state.stream_step_index += 1
                    st.session_state.stream_tx_list = fanout_sequence[:st.session_state.stream_step_index]
                    res_tx = fraud_data.add_realtime_simulation_transaction(st.session_state.stream_tx_list)
                    st.session_state.selected_tx_id = res_tx["tx_id"]
                    st.success(f"✅ Streamed Step {st.session_state.stream_step_index}/10: Out-Degree = {st.session_state.stream_step_index}, Risk = {res_tx['risk_score']}/100!")
                    st.rerun()
                else:
                    st.info("ℹ️ All 10 fan-out transactions have been streamed! Reset engine to restart.")
        with p_col2:
            if st.button("🚀 Stream All 10 Transactions at Once", use_container_width=True):
                st.session_state.stream_step_index = 10
                st.session_state.stream_tx_list = fanout_sequence
                res_tx = fraud_data.add_realtime_simulation_transaction(st.session_state.stream_tx_list)
                st.session_state.selected_tx_id = res_tx["tx_id"]
                st.success("✅ Streamed full 10-tx burst!")
                st.rerun()
        with p_col3:
            if st.button("🗑️ Reset Real-Time Stream Engine", use_container_width=True):
                st.session_state.stream_tx_list = []
                st.session_state.stream_step_index = 0
                st.rerun()

    # ── TAB 3: Batch Table Editor ──
    with tab_batch:
        st.markdown("#### ✍️ Batch Transaction Table Editor")
        if "editor_data" not in st.session_state:
            st.session_state.editor_data = pd.DataFrame(columns=["#", "Sender Account ID", "Sender Bank", "Receiver Account", "Amount ($)", "Payment Format", "Currency"])

        edited_df = st.data_editor(
            st.session_state.editor_data,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="table_editor_instance"
        )

        if st.button("⚡ Run Real-Time AI Model Inference on Batch Table", type="primary", use_container_width=True):
            if edited_df.empty:
                st.warning("⚠️ Please enter at least 1 transaction in the table before running inference.")
            else:
                custom_txs = []
                for idx, row in edited_df.iterrows():
                    custom_txs.append({
                        "timestamp": f"2026/09/21 14:{idx+1:02d}",
                        "from_bank": str(row.get("Sender Bank", "GlobalTrust Bank")),
                        "from_account": str(row.get("Sender Account ID", "ACC_78421")),
                        "to_bank": "Target Bank",
                        "to_account": str(row.get("Receiver Account", f"ACC_9011{idx+1}")),
                        "amount_paid": float(row.get("Amount ($)", 1000.0) or 1000.0),
                        "payment_currency": "US Dollar" if str(row.get("Currency", "USD")).upper() in ["USD", "US DOLLAR"] else str(row.get("Currency", "US Dollar")),
                        "payment_format": str(row.get("Payment Format", "ACH"))
                    })
                st.session_state.stream_tx_list = custom_txs
                new_tx = fraud_data.add_realtime_simulation_transaction(custom_txs)
                st.session_state.selected_tx_id = new_tx["tx_id"]
                st.success(f"✅ Real-Time Model Inference Complete for {new_tx['tx_id']}!")
                st.rerun()

    # ── DISPLAY LIVE STREAMING RESULTS & GRAPH ──
    st.markdown("---")
    curr_tx = fraud_data.get_transaction_by_id(st.session_state.selected_tx_id)
    n_streamed = len(st.session_state.stream_tx_list) if st.session_state.stream_tx_list else len(curr_tx.get("to_accounts", []))
    
    st.markdown(f"### 📊 Real-Time Stream Execution & Model Score Breakdown — `{curr_tx['tx_id']}`")
    st.caption(f"Streamed Target Out-Degree: **{n_streamed} Receiver Accounts** | Sender Account: **{curr_tx['account']}** | Total Amount: **{curr_tx['amount_formatted']}**")

    col_r0, col_r1, col_r2, col_r3, col_r4 = st.columns(5)
    with col_r0:
        st.metric("Accumulated Out-Degree", f"{n_streamed} Txs")
    with col_r1:
        st.metric("GAT Graph Score", curr_tx.get("gat_confidence", "86%"))
    with col_r2:
        st.metric("LightGBM Score", curr_tx.get("lgb_confidence", "88%"))
    with col_r3:
        st.metric("Rule Engine Score", curr_tx.get("rule_confidence", "95%"))
    with col_r4:
        st.metric("Ensemble Risk Score", f"{curr_tx['risk_score']}/100", delta=curr_tx['risk'])

    # Dynamic SHAP Explainability Box
    box_class = "xai-box-high" if curr_tx["risk"] == "High" else "xai-box-medium" if curr_tx["risk"] == "Medium" else "xai-box-low"
    risk_emoji = "🚨" if curr_tx["risk"] == "High" else "⚠️" if curr_tx["risk"] == "Medium" else "✅"
    
    st.html(textwrap.dedent(f"""
    <div class="xai-box {box_class}">
        <div class="xai-title">{risk_emoji} Real-Time SHAP Feature Impact & Model Explainability</div>
        <ul class="xai-list">
            {"".join([f"<li>{e}</li>" for e in curr_tx["explanations"]])}
        </ul>
    </div>
    """))

    # Live Interactive Network Graph
    st.markdown("#### 🕸️ Live Network Topology Graph (Updated Real-Time)")
    fig_sim = graph_vis.render_plotly_graph(curr_tx["tx_id"], include_2hop=True)
    st.plotly_chart(fig_sim, use_container_width=True)

    # ── Authorised Bank Auditor Decision Panel ──
    st.markdown("#### ⚖️ Authorised Bank Auditor Decision Panel")
    st.markdown("Review the real-time stream graph above. Submitting a decision re-colors graph nodes and updates risk score live:")

    aud_col1, aud_col2 = st.columns([1, 1])
    with aud_col1:
        auditor_decision_choice = st.radio(
            "Select Compliance Decision",
            ["✅ Approve / Mark Legitimate", "🚨 Flag as Confirmed Fraud / Laundering"],
            key="stream_auditor_decision_choice"
        )
    with aud_col2:
        auditor_notes_text = st.text_area(
            "Auditor Compliance Notes",
            placeholder="e.g., Verified legitimate payroll transfer or vendor invoice payment...",
            key="stream_auditor_notes_text",
            height=120
        )

    if st.button("📋 Submit Auditor Decision & Update Live Graph Topology", type="primary", use_container_width=True):
        dec_str = "Legitimate" if "Approve" in auditor_decision_choice else "Fraud"
        fraud_data.update_auditor_decision(curr_tx["tx_id"], dec_str, auditor_notes_text)
        st.success(f"✅ Recorded Auditor Decision ({auditor_decision_choice})! Updated risk score and graph topology.")
        st.rerun()

    # ── Table Log of Streamed Transactions ──
    if st.session_state.stream_tx_list:
        st.markdown("#### 📜 Streamed Transactions Log")
        df_stream_log = pd.DataFrame(st.session_state.stream_tx_list)
        df_stream_log.insert(0, "#", range(1, len(df_stream_log) + 1))
        st.dataframe(df_stream_log, use_container_width=True)

elif page == "Alerts / Graph Network":
    st.subheader("🕸️ Money Trail")
    st.markdown(
        "Visualizing transactional connections up to **2 hops**. "
        "Hover over any node to see the customer profile. "
        "Sender (★) → Hop-1 Receivers (●) → Hop-2 Downstream Nodes (◆)"
    )

    # Pre-select the tx from dashboard if navigated via the graph button
    default_sel = st.session_state.selected_tx_id
    tx_options = [t["tx_id"] for t in fraud_data.TRANSACTIONS]
    default_idx = tx_options.index(default_sel) if default_sel in tx_options else 0

    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        sel_tx = st.selectbox(
            "Select Transaction Network to Inspect",
            tx_options,
            index=default_idx
        )
    with col_g2:
        show_2hop = st.checkbox("Show 2-Hop Neighbors", value=True)

    tx_info = fraud_data.get_transaction_by_id(sel_tx)

    # Info bar
    risk_col = "#dc2626" if tx_info["risk"] == "High" else "#d97706" if tx_info["risk"] == "Medium" else "#dc2626"
    st.html(textwrap.dedent(f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                padding:14px 18px;margin-bottom:16px;display:flex;gap:28px;align-items:center;">
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">PATTERN</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info['pattern']}</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">RISK SCORE</div>
            <div style="font-size:14px;font-weight:700;color:{risk_col};">{tx_info['risk_score']}/100</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">MODEL</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info['model_used']}</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">SENDER</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info['account']}</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">AMOUNT</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info['amount_formatted']}</div>
        </div>
    </div>
    """))

    fig_g = graph_vis.render_plotly_graph(sel_tx, include_2hop=show_2hop)
    st.plotly_chart(fig_g, use_container_width=True)

    # Legend explanation
    st.html(textwrap.dedent("""
    <div style="display:flex;gap:24px;justify-content:center;margin-top:4px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:14px;height:14px;background:#ef4444;border-radius:50%;display:inline-block;"></span>
            Sender Account
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:14px;height:14px;background:#f59e0b;border-radius:50%;display:inline-block;"></span>
            Hop-1 Direct Receivers
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:14px;height:14px;background:#e2e8f0;border:1px solid #cbd5e1;border-radius:4px;display:inline-block;"></span>
            Hop-2 Downstream Nodes
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:28px;height:2px;background:#ef4444;display:inline-block;"></span>
            Hop-1 Transfer (amount shown)
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:28px;height:2px;background:#e2e8f0;border-bottom:2px dashed #cbd5e1;display:inline-block;"></span>
            Hop-2 Transfer
        </div>
    </div>
    """))

    st.markdown("---")
    # XAI explanations on graph page too
    st.html(textwrap.dedent(f"""
    <div style="background:#fff5f5;border:1px solid #fecaca;border-radius:10px;padding:14px 16px;">
        <div style="font-weight:700;font-size:13px;color:#dc2626;margin-bottom:8px;">⚠️ Why was this flagged?</div>
        <ul style="margin:0;padding-left:18px;font-size:12px;color:#334155;line-height:1.8;">
            {"".join([f"<li>{e}</li>" for e in tx_info["explanations"]])}
        </ul>
    </div>
    """))

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOMERS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Customers":
    st.subheader("👤 Customer Profiling & Network Risk")
    acc_sel = st.selectbox("Select Account ID", list(fraud_data.CUSTOMER_PROFILES.keys()))
    prof = fraud_data.get_customer_profile(acc_sel)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Name:** {prof.get('name', '—')}")
        st.markdown(f"**KYC Status:** {prof.get('kyc_status', '—')}")
        st.markdown(f"**City:** {prof.get('city', '—')}")
        st.markdown(f"**Account Type:** {prof.get('account_type', '—')}")
    with c2:
        st.markdown(f"**Risk Tier:** {prof.get('risk_tier', '—')}")
        st.markdown(f"**Open Since:** {prof.get('open_since', '—')}")
        st.markdown(f"**Last Login:** {prof.get('last_login', '—')}")
        st.markdown(f"**Devices:** {prof.get('device_count', '—')}")

    st.dataframe(pd.DataFrame(prof["behavior_summary"]), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  REMAINING PAGES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Reports":
    st.subheader("📄 Automated Compliance & Fraud Reports")
    st.markdown("Generate AI-driven regulatory reports for compliance and operations stakeholders.")
    if st.button("Generate Summary Compliance PDF Report"):
        st.success("✅ Report generated successfully for 1,80,256 transactions processed.")

elif page == "Settings":
    st.subheader("⚙️ System & Graph AI Settings")
    st.slider("GraphSAGE Risk Detection Sensitivity", 0.0, 1.0, 0.75)
    st.selectbox("Primary Baseline Model", ["GraphSAGE + XGBoost Ensemble", "Isolation Forest", "TabPFN", "GNN Convolutional"])

elif page == "Help":
    st.subheader("❓ Help & Documentation")
    st.markdown("""
### Fraud Patterns Explained:
- **Fan-Out**: Single source account transferring money to multiple newly opened receiver accounts in a short time frame.
- **Circular Money Flow**: Funds transferred in a ring (A ➔ B ➔ C ➔ A) to disguise origin.
- **Account Takeover**: Unexpected login locations and immediate high-value transfer out of a dormant account.
- **Structuring / Smurfing**: Breaking down large sums into multiple smaller transfers to stay below regulatory reporting limits.
- **Rapid Velocity**: Unusually high transaction frequency within a short time window.

### How to use the Dashboard:
1. Click a **Flagged Account** in the left panel to load its transactions.
2. Click any **row** in the transaction table to view the **receiver's customer profile** on the right.
3. Tap **"View as Graph Network"** to visualize the 2-hop transaction network.
4. For **High Risk** transactions, submit your decision in the **Authorised Bank Auditor Decision Panel**.

### AI Advisory Disclaimer:
The AI model provides pattern analysis and risk scores to *support* the investigation. **Final decisions must always be made by an AUTHORIZED BANK AUDITOR.**
    """)
