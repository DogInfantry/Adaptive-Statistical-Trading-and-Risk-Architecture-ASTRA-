import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION (The "Temple Gate")
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ASTRA | BNPP GMQR System",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Dark Mode" Finance Aesthetics
st.markdown("""
    <style>
    .metric-card {background-color: #1e1e1e; border-radius: 10px; padding: 20px; color: white;}
    .stAlert {background-color: #2b2b2b; color: #e0e0e0;}
    </style>
    """, unsafe_allow_html=True)

st.title("🦁 ASTRA: Adaptive Statistical Trading & Risk Architecture")
st.markdown("### **Global Markets Quantitative Research (GMQR) | Prime Strategist**")

# -----------------------------------------------------------------------------
# 2. SHARED DATA INGESTION ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def get_data(tickers, period="2y", interval="1d"):
    """Fetches data and automatically handles splits/dividends."""
    try:
        data = yf.download(tickers, period=period, interval=interval, auto_adjust=True)['Close']
        return data
    except Exception as e:
        st.error(f"Data Connection Failed: {e}")
        return pd.DataFrame()

# Sidebar Controls
st.sidebar.header("System Controls")
mode = st.sidebar.radio("Select Module", ["🛡️ Prime Brokerage (Inventory)", "⚡ Alpha Engine (Kalman Filter)"])

# -----------------------------------------------------------------------------
# MODULE A: PRIME BROKERAGE INVENTORY OPTIMIZER
# -----------------------------------------------------------------------------
if mode == "🛡️ Prime Brokerage (Inventory)":
    st.subheader("🛡️ Delta One Inventory & Funding Optimization")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Objective:** Minimize Portfolio Volatility under strict Balance Sheet constraints.")
    with col2:
        leverage_limit = st.slider("Gross Leverage Limit (x)", 1.0, 5.0, 2.0, 0.1)
    with col3:
        gamma_risk = st.slider("Risk Aversion Parameter (Gamma)", 0.1, 10.0, 1.0)

    # 1. Define Universe (Hardcoded 'The Book' for demo)
    prime_tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'GS', 'MS', 'TSLA', 'NVDA']
    with st.spinner(f"Connecting to Live Markets... Fetching {prime_tickers}"):
        df_prime = get_data(prime_tickers)
        
    if not df_prime.empty:
        # Calculate Returns
        returns = df_prime.pct_change().dropna()
        mu = returns.mean().values
        Sigma = returns.cov().values
        n = len(prime_tickers)

        # 2. CONVEX OPTIMIZATION (CVXPY)
        w = cp.Variable(n)
        portfolio_return = mu @ w
        portfolio_risk = cp.quad_form(w, Sigma)
        
        # Maximize Utility: Return - (Gamma * Risk)
        objective = cp.Maximize(portfolio_return - gamma_risk * portfolio_risk)
        
        # Constraints: 
        # 1. Gross Leverage <= Limit (The Bank's Balance Sheet)
        # 2. Net Exposure approx 0 (Delta Neutral)
        constraints = [
            cp.norm(w, 1) <= leverage_limit,
            cp.sum(w) >= -0.10, # Allow +/- 10% drift
            cp.sum(w) <= 0.10
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        # 3. OUTPUT
        weights = np.round(w.value, 4)
        
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Gross Leverage Used", f"{np.sum(np.abs(weights)):.2f}x", f"Limit: {leverage_limit}x")
        m2.metric("Net Exposure (Delta)", f"{np.sum(weights):.2%}", "Target: 0%")
        m3.metric("projected Annual Volatility", f"{np.sqrt(prob.value)*np.sqrt(252):.2%}")
        
        # Visualization
        results_df = pd.DataFrame({'Ticker': prime_tickers, 'Weight': weights})
        results_df['Type'] = results_df['Weight'].apply(lambda x: 'Long' if x > 0 else 'Short')
        
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
        sns.barplot(data=results_df, x='Ticker', y='Weight', hue='Type', palette={'Long': '#00ff00', 'Short': '#ff0000'}, dodge=False, ax=ax)
        ax.set_facecolor('#0e1117')
        ax.axhline(0, color='white', linewidth=1)
        ax.tick_params(colors='white')
        ax.set_title("Optimal Inventory Allocation (Convex Solution)", color='white')
        st.pyplot(fig)
        
        st.dataframe(results_df.style.background_gradient(cmap='RdYlGn', subset=['Weight']))

# -----------------------------------------------------------------------------
# MODULE B: KALMAN FILTER ALPHA ENGINE
# -----------------------------------------------------------------------------
elif mode == "⚡ Alpha Engine (Kalman Filter)":
    st.subheader("⚡ High-Frequency Statistical Arbitrage (Kalman Filter)")
    st.markdown("Dynamic Beta estimation using Bayesian Inference on Intraday Data.")

    c1, c2 = st.columns(2)
    with c1:
        asset_x = st.text_input("Asset X (Independent/Hedge)", "C")
    with c2:
        asset_y = st.text_input("Asset Y (Target/Dependent)", "JPM")
        
    # FETCH INTRADAY DATA
    if st.button("Initialize Kalman Algorithm"):
        with st.spinner("Ingesting Micro-structure Data (1-Minute Intervals)..."):
            # Fetch 5 days of 1-minute data
            dx = yf.download(asset_x, period="5d", interval="1m", auto_adjust=True)['Close']
            dy = yf.download(asset_y, period="5d", interval="1m", auto_adjust=True)['Close']
            
            # Align
            df_pair = pd.concat([dx, dy], axis=1).dropna()
            df_pair.columns = ['x', 'y']
            
            # -----------------------------------------------------------
            # THE KALMAN FILTER CLASS (Embedded for Production)
            # -----------------------------------------------------------
            class KalmanFilterReg:
                def __init__(self, delta=1e-5, R=1e-3):
                    self.P = np.zeros((2, 2)) 
                    self.R = R 
                    self.Q = np.eye(2) * delta 
                    self.theta = np.zeros(2) 

                def update(self, x, y):
                    H = np.array([1, x])
                    theta_pred = self.theta 
                    P_pred = self.P + self.Q
                    y_pred = H @ theta_pred
                    error = y - y_pred
                    S = H @ P_pred @ H.T + self.R
                    K = P_pred @ H.T / S
                    self.theta = theta_pred + K * error
                    self.P = (np.eye(2) - np.outer(K, H)) @ P_pred
                    return self.theta, error, np.sqrt(S)

            # Run Filter
            kf = KalmanFilterReg()
            betas = []
            z_scores = []
            
            progress_bar = st.progress(0)
            
            for t in range(len(df_pair)):
                theta, error, std = kf.update(df_pair['x'].iloc[t], df_pair['y'].iloc[t])
                betas.append(theta[1])
                z_scores.append(error / std)
                if t % 100 == 0:
                    progress_bar.progress(t / len(df_pair))
                    
            progress_bar.empty()
            
            # Plotting
            st.success(f"Algorithm Converged. Current Dynamic Beta: {betas[-1]:.4f}")
            
            # Dynamic Beta Chart
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, facecolor='#0e1117')
            
            ax1.plot(betas, color='#00ff00', linewidth=1)
            ax1.set_title(f"Real-Time Hedge Ratio (Beta)", color='white')
            ax1.set_facecolor('#0e1117')
            ax1.tick_params(colors='white')
            
            ax2.plot(z_scores, color='cyan', linewidth=0.8)
            ax2.axhline(2.0, color='red', linestyle='--')
            ax2.axhline(-2.0, color='red', linestyle='--')
            ax2.set_title("Z-Score Trading Signal (Mean Reversion)", color='white')
            ax2.set_facecolor('#0e1117')
            ax2.tick_params(colors='white')
            
            st.pyplot(fig2)

st.sidebar.markdown("---")
st.sidebar.info("Developed by: The 10th House Architect\nTarget: BNP Paribas GMQR")
