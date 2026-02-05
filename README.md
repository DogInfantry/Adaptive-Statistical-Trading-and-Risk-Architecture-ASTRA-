# Adaptive-Statistical-Trading-and-Risk-Architecture-ASTRA-
# 🦁 ASTRA: Adaptive Statistical Trading & Risk Architecture

### **Institutional Prime Brokerage & High-Frequency Alpha Engine**

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Optimization](https://img.shields.io/badge/Optimization-Convex_(CVXPY)-green) ![Alpha](https://img.shields.io/badge/Alpha-Bayesian_Kalman_Filter-orange) ![Risk](https://img.shields.io/badge/Risk-Monte_Carlo_VaR-red)

---

## 🚀 Executive Summary
**ASTRA** is a dual-engine quantitative framework designed to solve the two core challenges of a modern Prime Brokerage desk: **Inventory Optimization** and **Alpha Generation**.

In a high-rate environment (SOFR > 3.6%), "lazy" balance sheets bleed P&L. ASTRA minimizes this funding drag via **L1-Norm Convex Optimization** while simultaneously capturing uncorrelated alpha through a **High-Frequency Kalman Filter** that exploits mean-reversion in bank sector microstructure.

### **🏆 Key Performance Metrics (Live Backtest)**
| Metric | Result | Strategic Implication |
| :--- | :--- | :--- |
| **Alpha Generation** | **15.10 Sharpe Ratio** | Superior risk-adjusted returns via dynamic hedging. |
| **5-Day Return** | **+5.83%** | Rapid capital appreciation on 1-minute microstructure data. |
| **Risk Control** | **$118k VaR (99%)** | Solvency confirmed via 5,000 Monte Carlo simulations. |
| **Capital Efficiency** | **2.0x Leverage Cap** | Strict adherence to Balance Sheet constraints (L1 Regularization). |
| **Volatility** | **18.55%** | Minimized portfolio variance via covariance optimization. |

---

## 🧠 The Business Problem: The "Cost of Carry" Crisis
Legacy Prime Brokerage models often rely on static inventory management.
* **The Bleed:** Holding a long/short book costs money. With SOFR at ~3.6%, the **Daily Funding Cost** on an unoptimized book was identified as **-$41.49 per $2M**, creating an annualized drag of ~$15,000.
* **The Risk:** Static "Pairs Trading" (OLS Regression) fails during regime shifts, leaving desks exposed to widening spreads.

**ASTRA's Solution:** A mathematical engine that dynamically reallocates capital to "Self-Funding" assets and adapts hedge ratios in real-time.

---

## ⚙️ System Architecture

ASTRA operates as a unified command center driven by three distinct engines:

### **1. 🛡️ The Inventory Engine (Convex Optimization)**
* **Objective:** Minimize Portfolio Variance while strictly enforcing Gross Leverage limits.
* **Math:** Solves the convex problem:
    $$\text{minimize } w^T \Sigma w$$
    $$\text{subject to } ||w||_1 \le 2.0, \quad \sum w \approx 0$$
* **Outcome:** The model rejected an Equal-Weight allocation, identifying **JPM (37%)** and **MSFT (36%)** as the optimal collateral mix to minimize variance.

### **2. ⚡ The Alpha Engine (Bayesian Kalman Filter)**
* **Objective:** Capture mean-reversion profits in the **JPM vs. Citi** spread using 1-minute intraday bars.
* **Innovation:** Unlike static Beta (Linear Regression), the **Kalman Filter** treats the Hedge Ratio as a "Hidden State" that evolves stochastically.
* **Signal:**
    * **Dynamic Beta:** Observed shifting from **2.62 to 2.70** in real-time.
    * **Execution:** Short Spread when Z-Score > 2.0; Exit when Z-Score < 0.5.

### **3. 📉 The Risk Engine (Cholesky Monte Carlo)**
* **Objective:** Stress-test the book against "Black Swan" events.
* **Methodology:** 5,000 simulated market scenarios using **Cholesky Decomposition** to preserve asset correlations (e.g., Tech stocks moving together).
* **Safety:** Confirmed a **99% Value-at-Risk (VaR)** of **-11.86%**, validating the strategy fits within institutional risk appetites.

---

## 💻 Tech Stack & Installation

The project is built as a modular Python package with a **Streamlit** front-end for trader interaction.

### **Prerequisites**
* Python 3.10+
* Anaconda / Miniconda

### **Quick Start**
```bash
# 1. Clone the Repository
git clone [https://github.com/YourUsername/Project-ASTRA.git](https://github.com/YourUsername/Project-ASTRA.git)

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Launch the Command Center
streamlit run app.py
