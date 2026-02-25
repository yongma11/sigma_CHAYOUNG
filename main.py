import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import matplotlib.dates as mdates
import os
import json
from datetime import datetime, timedelta

# -----------------------------------------------------------
# 0. 기본 설정 & 스타일
# -----------------------------------------------------------
st.set_page_config(page_title="SOXL Sigma2 Trader", layout="wide", page_icon="🚀")

st.markdown("""
    <style>
    .big-metric { font-size: 26px !important; font-weight: bold; color: #1E88E5; }
    .order-box { text-align: center; padding: 20px; border-radius: 10px; color: white; font-weight: bold; }
    .stDataFrame { border: 1px solid #ddd; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 파일 입출력 및 설정 관리
# -----------------------------------------------------------
LOG_FILE = 'trade_log_v3.csv'
PROFIT_FILE = 'profit_log_v3.csv'
SETTINGS_FILE = 'settings.json'

def load_settings():
    default_settings = {'start_date': '2024-01-01', 'initial_capital': 10000}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f: return json.load(f)
        except: return default_settings
    return default_settings

def save_settings(start_date, initial_capital):
    settings = {'start_date': start_date.strftime('%Y-%m-%d'), 'initial_capital': initial_capital}
    with open(SETTINGS_FILE, 'w') as f: json.dump(settings, f)

def load_trade_log():
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            if not df.empty and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date', ascending=False)
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            return df
        except: pass
    return pd.DataFrame(columns=['Date', 'Type', 'Price', 'Qty', 'Tx_Value', 'Balance_Qty', 'Total_Cash', 'Total_Asset', 'Allocated_Cap'])

def save_trade_log(df):
    df.to_csv(LOG_FILE, index=False)

def load_profit_log():
    if os.path.exists(PROFIT_FILE):
        try: return pd.read_csv(PROFIT_FILE)
        except: pass
    return pd.DataFrame(columns=['Date', 'Total_Asset', 'Total_Profit', 'Return_Pct'])

def save_profit_log(df):
    df.to_csv(PROFIT_FILE, index=False)

# -----------------------------------------------------------
# 2. 데이터 및 로직 함수
# -----------------------------------------------------------
def calculate_growth_curve_precise(series, dates, window=1260):
    results = [np.nan] * len(series)
    date_nums = dates.map(pd.Timestamp.toordinal).values
    values = series.values
    for i in range(window, len(series)):
        y_train = values[i-window : i]
        x_train = date_nums[i-window : i]
        if np.any(y_train <= 0) or np.isnan(y_train).any(): continue
        try:
            fit = np.polyfit(x_train, np.log(y_train), 1)
            pred_log = fit[1] + fit[0] * date_nums[i]
            results[i] = np.exp(pred_log)
        except: pass
    return pd.Series(results, index=series.index)

@st.cache_data(ttl=300)
def get_market_data():
    try:
        # 미국 시장 마감 후 최신 데이터를 가져오기 위해 내일 날짜까지 요청
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = "2005-01-01" 
        
        df_soxl = yf.download("SOXL", start=start_date, end=end_date, progress=False, auto_adjust=True)
        df_qqq = yf.download("QQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if isinstance(df_soxl.columns, pd.MultiIndex): soxl_close = df_soxl['Close']['SOXL']
        else: soxl_close = df_soxl['Close']
        if isinstance(df_qqq.columns, pd.MultiIndex): qqq_close = df_qqq['Close']['QQQ']
        else: qqq_close = df_qqq['Close']

        df = pd.DataFrame({'SOXL': soxl_close, 'QQQ': qqq_close}).sort_index().reset_index()
        if 'Date' not in df.columns: df.rename(columns={'index':'Date'}, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        df['Growth'] = calculate_growth_curve_precise(df['QQQ'], df['Date'], window=1260)
        df['Eval'] = (df['QQQ'] / df['Growth']) - 1
        df['SOXL_Pct'] = df['SOXL'].pct_change()
        df['Sigma'] = df['SOXL_Pct'].rolling(window=2).std(ddof=0).shift(1)
        df['SOXL_Prev'] = df['SOXL'].shift(1)
        
        return df.dropna(subset=['Sigma','Eval']).reset_index(drop=True)
    except: return None

# [핵심] 시뮬레이션 엔진
def run_strategy_engine(df, start_date, initial_capital, params):
    sim_data = df[df['Date'] >= pd.to_datetime(start_date)].copy().reset_index(drop=True)
    
    cash = initial_capital
    shares = 0
    allocated_capital = initial_capital
    avg_price = 0
    
    detailed_logs = [] 
    simple_logs = [] 
    equity_history = [] 
    profit_records = []
    
    wins = []
    losses = []
    
    renewal = params.get('renewal', 7)
    split = params.get('split', 5)
    pcr = params.get('pcr', 0.95)
    lcr = params.get('lcr', 0.85)
    
    for i, row in sim_data.iterrows():
        price = row['SOXL']
        prev = row['SOXL_Prev']
        sigma = row['Sigma']
        mkt_eval = row['Eval']
        date_str = row['Date'].strftime('%Y-%m-%d')
        
        # 1. 리밸런싱
        if len(equity_history) > 0 and len(equity_history) % renewal == 0:
            if len(equity_history) >= renewal:
                profit = equity_history[-1] - equity_history[-renewal]
                factor = pcr if profit >= 0 else lcr
                allocated_capital += profit * factor
                allocated_capital = max(allocated_capital, 1000.0)

        # 2. 티어 판단
        tier_label = "MID"
        if mkt_eval < -0.10: tier_label = "ULOW"
        elif mkt_eval < -0.05: tier_label = "LOW"
        elif mkt_eval > 0.10: tier_label = "UHIGH"
        elif mkt_eval > 0.07: tier_label = "HIGH"
        
        # 3. 목표가 (Pivot)
        pivot = prev * (1 + 0.55 * sigma)
        t_buy = pivot
        t_sell = pivot
        
        # 4. 매매
        action = None
        qty = 0
        signed_qty = 0
        realized_pnl = 0
        
        if price > t_sell: 
            if shares > 0:
                ratio = 0.65
                if tier_label=='ULOW': ratio=0.55
                elif tier_label=='LOW': ratio=0.60
                elif tier_label=='HIGH': ratio=0.70
                elif tier_label=='UHIGH': ratio=0.75
                
                qty = int(shares * ratio)
                if qty > 0:
                    realized_pnl = qty * (price - avg_price)
                    if realized_pnl > 0: wins.append(realized_pnl)
                    else: losses.append(realized_pnl)
                    
                    shares -= qty
                    cash += qty * price
                    if shares == 0: avg_price = 0
                    action = 'Sell'
                    signed_qty = -qty
        
        elif price < t_buy:
            slot = allocated_capital / split
            if (shares * price < allocated_capital) and (cash > slot):
                can_buy = min(cash, slot)
                qty = int(can_buy / t_buy) 
                
                if qty > 0:
                    cost = qty * price 
                    avg_price = (avg_price*shares + cost) / (shares + qty)
                    shares += qty
                    cash -= cost
                    action = 'Buy'
                    signed_qty = qty
        
        # 5. 기록
        final_equity = cash + (shares * price)
        equity_history.append(final_equity)
        
        profit_records.append({
            'Date': row['Date'], 
            'Total_Asset': final_equity,
            'Total_Profit': final_equity - initial_capital,
            'Return_Pct': (final_equity - initial_capital)/initial_capital * 100
        })
        
        if action:
            simple_logs.append({
                'Date': date_str, 'Type': action, 'Price': round(price, 2),
                'Qty': signed_qty, 'Tx_Value': round(qty * price, 0),
                'Balance_Qty': shares, 'Total_Cash': round(cash, 0),
                'Total_Asset': round(final_equity, 0), 'Allocated_Cap': round(allocated_capital, 0)
            })
            
            detailed_logs.append({
                '날짜': date_str,
                '평가': f"{mkt_eval*100:.2f}% ({tier_label})",
                '1시그마': f"{sigma:.4f}",
                '종가': f"${price:.2f}",
                '일일투자금': f"${allocated_capital:,.0f}",
                '매수/매도 기준가': f"${pivot:.2f}",
                '주문수량': f"{signed_qty:+d}",
                '실현손익': f"${realized_pnl:,.0f}" if action == 'Sell' else "-",
                '총자산': f"${final_equity:,.0f}",
                '수익률': f"{(final_equity/initial_capital - 1)*100:.2f}%"
            })
            
    final_state = {
        'cash': cash, 'shares': shares, 'allocated_capital': allocated_capital,
        'cycle_day': (len(equity_history) % renewal) + 1, 'renewal_period': renewal
    }
    
    perf_metrics = {
        'wins': wins, 'losses': losses
    }
            
    return pd.DataFrame(simple_logs), pd.DataFrame(profit_records), pd.DataFrame(detailed_logs), final_state, perf_metrics

# -----------------------------------------------------------
# 2. 사이드바 설정 (설정 저장 기능)
# -----------------------------------------------------------
st.title("🚀 [시그마2] : SOXL for Chayoung")
st.sidebar.header("⚙️ 기본 설정")

saved_settings = load_settings()

with st.sidebar.form("settings_form"):
    start_date_val = pd.to_datetime(saved_settings['start_date'])
    init_cap_val = saved_settings['initial_capital']
    
    start_date = st.date_input("투자 시작일", start_date_val)
    init_cap = st.number_input("시작 원금 ($)", value=init_cap_val, step=1000)
    
    update_btn = st.form_submit_button("🔄 설정 저장 및 데이터 갱신")

if update_btn:
    save_settings(start_date, init_cap)
    st.sidebar.success("설정이 저장되었습니다!")

df_market = get_market_data()

if 'trade_log' not in st.session_state:
    st.session_state['trade_log'] = load_trade_log()
if 'profit_log' not in st.session_state:
    st.session_state['profit_log'] = load_profit_log()

# -----------------------------------------------------------
# 3. 메인 화면
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔥 실전 트레이딩", "📊 백테스트 분석", "📘 전략 로직"])

# ===========================================================
# TAB 1: 실전 트레이딩
# ===========================================================
with tab1:
    if df_market is not None:
        params = {'renewal':7, 'split':5, 'pcr':0.95, 'lcr':0.85}

        # 데이터 갱신 로직
        if update_btn or st.session_state['trade_log'].empty:
            with st.spinner("데이터 분석 및 복원 중..."):
                s_logs, s_profits, d_logs, state, _ = run_strategy_engine(df_market, start_date, init_cap, params)
                
                save_log_df = s_logs.copy()
                save_prof_df = s_profits.copy()
                
                if not save_prof_df.empty:
                    save_prof_df['Date'] = save_prof_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x)
                
                if not save_log_df.empty:
                    save_log_df['Date'] = pd.to_datetime(save_log_df['Date'])
                    save_log_df = save_log_df.sort_values('Date', ascending=False)
                    save_log_df['Date'] = save_log_df['Date'].dt.strftime('%Y-%m-%d')

                st.session_state['trade_log'] = save_log_df
                st.session_state['profit_log'] = save_prof_df
                st.session_state['current_state'] = state
                save_trade_log(save_log_df)
                save_profit_log(save_prof_df)
        
        # 현재 상태 계산
        if 'current_state' in st.session_state:
            state = st.session_state['current_state']
        else:
            _, _, _, state, _ = run_strategy_engine(df_market, start_date, init_cap, params)

        last_row = df_market.iloc[-1]
        prev_row_market = df_market.iloc[-2]
        
        # [SECTION 1] 정보바
        eval_val = last_row['Eval']
        tier_label = "MID (중립)"
        tier_color = "gray"
        if eval_val < -0.10: tier_label="ULOW (초저평가)"; tier_color="green"
        elif eval_val < -0.05: tier_label="LOW (저평가)"; tier_color="lightgreen"
        elif eval_val > 0.10: tier_label="UHIGH (초고평가)"; tier_color="red"
        elif eval_val > 0.07: tier_label="HIGH (고평가)"; tier_color="orange"
        
        cur_price = last_row['SOXL']
        price_chg = cur_price - prev_row_market['SOXL']
        price_pct = (price_chg / prev_row_market['SOXL']) * 100
        pct_color = "red" if price_pct > 0 else "blue"
        sign = "+" if price_pct > 0 else ""
        
        slot_val = state['allocated_capital'] / 5
        cycle_txt = f"{state['cycle_day']}일차 ({state['renewal_period']}일 주기)"
        
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; background:#f8f9fa; padding:15px; border-radius:10px; border:1px solid #ddd; margin-bottom:20px;">
            <div style="text-align:center; width:25%;">
                <div style="font-size:14px; color:#666;">시장 모드</div>
                <div style="font-size:22px; font-weight:bold; color:{tier_color};">{tier_label}</div>
                <div class="sub-text">평가율 {eval_val*100:.2f}%</div>
            </div>
            <div style="text-align:center; width:25%; border-left:1px solid #ddd;">
                <div style="font-size:14px; color:#666;">SOXL 현재가</div>
                <div style="font-size:22px; font-weight:bold;">${cur_price:.2f}</div>
                <div class="sub-text" style="color:{pct_color};">전일대비 {sign}{price_pct:.2f}%</div>
            </div>
            <div style="text-align:center; width:25%; border-left:1px solid #ddd;">
                <div style="font-size:14px; color:#666;">1티어 할당금(5분할)</div>
                <div style="font-size:22px; font-weight:bold;">${slot_val:,.0f}</div>
            </div>
            <div style="text-align:center; width:25%; border-left:1px solid #ddd;">
                <div style="font-size:14px; color:#666;">매매 사이클</div>
                <div style="font-size:22px; font-weight:bold; color:#0068c9;">{cycle_txt}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # [SECTION 2] 오늘 주문표 (오늘 날짜 + 예상 잔고 표시)
        today_date = datetime.now().strftime('%Y-%m-%d')
        st.subheader(f"📝 오늘 주문표 ({today_date})")
        
        c_ord1, c_ord2 = st.columns([1, 2])
        
        with c_ord1:
            manual_price = st.number_input("예상 종가 입력 ($)", value=float(cur_price), step=0.01)
        
        prev_row = df_market.iloc[-2]
        pivot = prev_row['SOXL'] * (1 + 0.55 * prev_row['Sigma'])
        
        with c_ord2:
            log_df_now = st.session_state['trade_log']
            if not log_df_now.empty:
                shares_now = log_df_now.iloc[0]['Balance_Qty']
                cash_now = log_df_now.iloc[0]['Total_Cash']
                alloc_now = log_df_now.iloc[0]['Allocated_Cap']
            else:
                shares_now = 0
                cash_now = init_cap
                alloc_now = init_cap

            decision_text = ""
            box_style = ""
            final_qty_str = ""
            
            if manual_price < pivot:
                # 매수
                if (shares_now * manual_price < alloc_now) and (cash_now > slot_val):
                    can_buy = min(cash_now, slot_val)
                    order_qty = int(can_buy / pivot) 
                    expected_final_qty = shares_now + order_qty
                    
                    decision_text = f"📉 매수 (BUY): 기준가 ${pivot:.2f} ({order_qty}주)"
                    final_qty_str = f"📦 매매 후 예상 잔고: {expected_final_qty:,} 주"
                    box_style = "background-color: #d1e7dd; color: #0f5132;"
                else:
                    decision_text = f"💤 관망 (HOLD): 매수 신호이나 자금 부족"
                    final_qty_str = f"📦 예상 잔고: {shares_now:,} 주 (변동 없음)"
                    box_style = "background-color: #eee; color: #666;"
            elif manual_price > pivot:
                # 매도
                if shares_now > 0:
                    r = 0.65
                    if "ULOW" in tier_label: r=0.55
                    elif "LOW" in tier_label: r=0.60
                    elif "HIGH" in tier_label: r=0.70
                    elif "UHIGH" in tier_label: r=0.75
                    order_qty = int(shares_now * r)
                    expected_final_qty = shares_now - order_qty
                    
                    decision_text = f"📈 매도 (SELL): 기준가 ${pivot:.2f} ({order_qty}주)"
                    final_qty_str = f"📦 매매 후 예상 잔고: {expected_final_qty:,} 주"
                    box_style = "background-color: #f8d7da; color: #842029;"
                else:
                    decision_text = "💤 관망 (HOLD): 매도 신호이나 잔고 없음"
                    final_qty_str = f"📦 예상 잔고: {shares_now:,} 주"
                    box_style = "background-color: #eee; color: #666;"
            else:
                decision_text = "💤 관망 (HOLD): 기준가 동일"
                final_qty_str = f"📦 예상 잔고: {shares_now:,} 주"
                box_style = "background-color: #eee; color: #666;"
            
            st.markdown(f"""
            <div class="order-box" style="{box_style}">
                <div style="font-size: 24px;">{decision_text}</div>
                <div style="font-size: 18px; margin-top: 5px; opacity: 0.9;">{final_qty_str}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # [SECTION 3] 계좌 현황
        st.subheader("💰 내 계좌 현황")
        cur_total_asset = cash_now + (shares_now * manual_price)
        total_pnl = cur_total_asset - init_cap
        total_ret = (total_pnl / init_cap) * 100
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("총 보유 수량", f"{shares_now:,} 주")
        k2.metric("보유 현금", f"${cash_now:,.0f}")
        k3.metric("총 평가 손익", f"${total_pnl:,.0f}", delta=f"{total_ret:.1f}%")
        k4.metric("현재 총 자산", f"${cur_total_asset:,.0f}")
        
        st.write("")

        # [SECTION 4] 트레이드 로그
        with st.expander("📋 매매 로그 (수정 가능)", expanded=False):
            edited_trade_log = st.data_editor(
                st.session_state['trade_log'],
                num_rows="dynamic",
                use_container_width=True
            )
            if st.button("💾 로그 저장"):
                if not edited_trade_log.empty:
                    edited_trade_log['Date'] = pd.to_datetime(edited_trade_log['Date'])
                    edited_trade_log = edited_trade_log.sort_values('Date', ascending=False)
                    edited_trade_log['Date'] = edited_trade_log['Date'].dt.strftime('%Y-%m-%d')
                st.session_state['trade_log'] = edited_trade_log
                save_trade_log(edited_trade_log)
                st.success("로그가 저장되었습니다.")
                st.rerun()

        st.write("")

        # [SECTION 5] 매매 수익 기록
        st.subheader("📈 매매 수익 기록")
        
        with st.expander("📝 수익 일지 (수정 가능)", expanded=False):
            profit_df_for_editor = st.session_state['profit_log'].copy()
            if not profit_df_for_editor.empty:
                profit_df_for_editor['Date'] = pd.to_datetime(profit_df_for_editor['Date'])
                profit_df_for_editor = profit_df_for_editor.sort_values('Date', ascending=False)

            edited_profits = st.data_editor(
                profit_df_for_editor,
                column_config={
                    "Total_Asset": st.column_config.NumberColumn("총 자산 ($)", format="$%.0f"),
                    "Total_Profit": st.column_config.NumberColumn("누적 수익금 ($)", format="$%.0f"),
                    "Return_Pct": st.column_config.NumberColumn("수익률 (%)", format="%.2f%%"),
                    "Date": st.column_config.DateColumn("날짜", format="YYYY-MM-DD")
                },
                num_rows="dynamic",
                use_container_width=True,
                height=300
            )
            
            if st.button("💾 수익 기록 저장"):
                save_df = edited_profits.copy()
                save_df['Date'] = save_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x)
                st.session_state['profit_log'] = save_df
                save_profit_log(save_df)
                st.toast("저장되었습니다!", icon="✅")
                st.rerun()

        st.markdown("##### 📊 자산 성장 그래프")
        chart_data = st.session_state['profit_log'].copy()
        if not chart_data.empty:
            chart_data['Date'] = pd.to_datetime(chart_data['Date'])
            chart_data = chart_data.sort_values('Date')
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(chart_data['Date'], chart_data['Total_Asset'], color='#C0392B', label='Total Asset ($)')
            ax.axhline(y=init_cap, color='gray', linestyle='--', label='Initial Capital')
            ax.set_ylabel("Total Asset ($)")
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='upper left')
            
            ax2 = ax.twinx()
            ax2.plot(chart_data['Date'], chart_data['Return_Pct'], color='blue', alpha=0.2, linestyle='-', label='Return (%)')
            ax2.set_ylabel("Return (%)")
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
            st.pyplot(fig)
        else:
            st.info("데이터가 없습니다.")

    else:
        st.error("데이터 로딩 실패")

# ===========================================================
# TAB 2: 백테스트
# ===========================================================
with tab2:
    st.markdown("### 📊 [시그마2] 전략 백테스트 분석")
    
    with st.form("backtest_form"):
        bc1, bc2, bc3 = st.columns(3)
        b_cap = bc1.number_input("초기 자본 ($)", 10000)
        b_start = bc2.date_input("검증 시작일", pd.to_datetime("2010-05-01"))
        b_end = bc3.date_input("검증 종료일", datetime.now())
        
        run_bt = st.form_submit_button("🔄 분석 실행")
    
    if run_bt:
        with st.spinner("15년치 정밀 분석 중..."):
            if df_market is not None:
                params_bt = {'renewal':7, 'split':5, 'pcr':0.95, 'lcr':0.85}
                bt_logs, profit_df, d_logs, _, metrics = run_strategy_engine(df_market, b_start, b_cap, params_bt)
                
                profit_df = profit_df[
                    (pd.to_datetime(profit_df['Date']) >= pd.to_datetime(b_start)) & 
                    (pd.to_datetime(profit_df['Date']) <= pd.to_datetime(b_end))
                ].copy()
                
                if not profit_df.empty:
                    final_eq = profit_df.iloc[-1]['Total_Asset']
                    total_ret = (final_eq / b_cap - 1) * 100
                    days = (pd.to_datetime(profit_df.iloc[-1]['Date']) - pd.to_datetime(profit_df.iloc[0]['Date'])).days
                    cagr = (final_eq / b_cap) ** (365/days) - 1 if days > 0 else 0
                    
                    profit_df['Peak'] = profit_df['Total_Asset'].cummax()
                    profit_df['DD'] = (profit_df['Total_Asset'] / profit_df['Peak']) - 1
                    mdd = profit_df['DD'].min()
                    calmar = cagr / abs(mdd) if mdd != 0 else 0
                    
                    wins = metrics['wins']
                    losses = metrics['losses']
                    win_rate = len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0
                    profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
                    
                    st.divider()
                    st.subheader("1. 종합 성과")
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("최종 수익금", f"${final_eq - b_cap:,.0f}", delta=f"{total_ret:.1f}%")
                    m2.metric("CAGR", f"{cagr*100:.1f}%")
                    m3.metric("MDD", f"{mdd*100:.2f}%")
                    m4.metric("Calmar", f"{calmar:.2f}")
                    m5.metric("승률", f"{win_rate:.1f}%")
                    m6.metric("손익비", f"{profit_factor:.2f}")
                    
                    st.subheader("2. 성과 차트")
                    fig2, ax1 = plt.subplots(figsize=(12, 5))
                    ax1.plot(pd.to_datetime(profit_df['Date']), profit_df['Total_Asset'], color='#1E88E5', label='Total Asset')
                    ax1.set_yscale('log')
                    ax1.set_ylabel("Total Asset ($)", color='#1E88E5')
                    ax1.grid(True, alpha=0.3)
                    
                    ax2 = ax1.twinx()
                    ax2.fill_between(pd.to_datetime(profit_df['Date']), profit_df['DD']*100, 0, color='gray', alpha=0.2, label='Drawdown')
                    ax2.set_ylabel("Drawdown (%)", color='gray')
                    ax2.set_ylim(-100, 10)
                    st.pyplot(fig2)
                    
                    st.subheader("3. 연도별 성과")
                    profit_df['Year'] = pd.to_datetime(profit_df['Date']).dt.year
                    yearly = []
                    for y, g in profit_df.groupby('Year'):
                        start_v = g['Total_Asset'].iloc[0]
                        end_v = g['Total_Asset'].iloc[-1]
                        ret = (end_v/start_v - 1)*100
                        y_mdd = g['DD'].min() * 100
                        yearly.append({'Year':y, 'Return':f"{ret:.1f}%", 'MDD':f"{y_mdd:.1f}%", 'End Asset':f"${end_v:,.0f}"})
                    st.dataframe(pd.DataFrame(yearly).set_index('Year').T, use_container_width=True)
                    
                    st.subheader("4. 백테스트 상세 매매 로그")
                    if not d_logs.empty:
                        d_logs['날짜'] = pd.to_datetime(d_logs['날짜'])
                        d_logs_filtered = d_logs[
                            (d_logs['날짜'] >= pd.to_datetime(b_start)) & 
                            (d_logs['날짜'] <= pd.to_datetime(b_end))
                        ]
                        d_logs_filtered['날짜'] = d_logs_filtered['날짜'].dt.strftime('%Y-%m-%d')
                        st.dataframe(d_logs_filtered.sort_values('날짜', ascending=False), use_container_width=True)

# ===========================================================
# TAB 3: 전략 로직
# ===========================================================
with tab3:
    st.markdown("""
    ### 📘 [시그마2] 전략 가이드 (상세)
    
    **"시그마2"**는 시장의 변동성을 역이용하지 않고 **순응(Breakout)**하여 추세를 타는 공격적인 스윙 전략입니다.
    
    ---
    
    #### 1. 핵심 매매 기준 (단일 피벗 기준선)
    매수와 매도의 기준선이 하나(Pivot)입니다. 이 선을 넘나들 때 즉각 대응합니다.
    
    * **기준선(Pivot) 공식:** $$ Pivot = 전일종가 \times (1 + 0.55 \times Sigma) $$
      *(Sigma = SOXL 전일 등락률의 2일 이동 표준편차)*
      
    * **📉 매수 (Buy):**
      * 조건: **현재가 < Pivot**
      * **[현실적용]** LOC 주문을 사용하므로, 수량 계산 시 **'현재가' 대신 '기준가(Pivot)'**를 사용하여 주문 수량을 산정합니다. 체결 시 실제 종가가 기준가보다 낮으므로 현금이 소폭 남는 안전한 방식입니다.
      
    * **📈 매도 (Sell):**
      * 조건: **현재가 > Pivot**
      * 의미: 주가가 기준선을 돌파하여 상승하면 '과열/수익실현 구간'으로 봅니다. 보유 물량을 매도하여 현금을 확보합니다.
    
    ---
    
    #### 2. 시장 평가 (Market Tier)
    QQQ(나스닥)의 위치를 파악하여 매도할 때 **'얼마나 팔지'**를 결정합니다.
    
    * **Eval 지표:** $(QQQ / Growth) - 1$
      *(Growth = QQQ 5년 지수 회귀 추세선)*
      
    * **티어별 매도 비율:**
      * **ULOW (Eval < -10%):** 시장이 매우 쌉니다. 상승 여력이 크므로 매도 시 **55%**만 팝니다.
      * **LOW (-10% ~ -5%):** 저평가 구간. **60%** 매도.
      * **MID (-5% ~ 7%):** 평범한 구간. **65%** 매도.
      * **HIGH (7% ~ 10%):** 고평가 구간. **70%** 매도.
      * **UHIGH (> 10%):** 매우 비쌉니다. 위험 관리를 위해 **75%** 대량 매도.
    
    ---
    
    #### 3. 자금 관리 (Money Management)
    * **5분할 매수:** 전체 투자금을 5개 슬롯으로 나누어, 한 번 매수 시 1개 슬롯만큼만 진입합니다. (리스크 분산)
    * **7일 주기 리밸런싱:**
      * 매 7일마다 계좌의 총 자산을 점검합니다.
      * **이익 발생 시:** 이익금의 **95%**를 투자 원금에 합산합니다. (적극적 복리)
      * **손실 발생 시:** 손실금의 **85%**만 반영하여 투자 원금을 줄입니다. (자산 방어)
    """)
