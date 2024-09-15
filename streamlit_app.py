import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go

# Set Streamlit configuration
st.set_page_config(
    page_title='OJO Dashboard',
    page_icon=':bar_chart:',
    layout='wide'
)

# Function to load data from the CSV files in the 'data' folder
def load_data():
    conn = sqlite3.connect(":memory:")  # Use in-memory database
    cursor = conn.cursor()

    # Create income_statement table
    cursor.execute('''
    CREATE TABLE income_statement (
        Month TEXT,
        Sales REAL,
        Cost_of_Goods_Sold REAL,
        Gross_Profit REAL,
        Selling REAL,
        Administrative REAL,
        Total_Operating_Expenses REAL,
        Operating_Income REAL,
        Interest_Revenue REAL,
        Dividend_Revenue REAL,
        Interest_on_Loans REAL,
        PreTax_Income REAL,
        Income_Tax REAL,
        Net_Income REAL
    );''')

    # Load income_statement CSV from 'data' folder
    income_statement_file = Path(__file__).parent / 'data/income_statement_2324.csv'
    df_income = pd.read_csv(income_statement_file)
    df_income.to_sql('income_statement', conn, if_exists='append', index=False)

    # Create balance_sheet table
    cursor.execute('''
    CREATE TABLE balance_sheet (
        Month TEXT,
        Cash REAL,
        Accounts_Receivable REAL,
        Inventory REAL,
        Supplies REAL,
        Pre_Paids REAL,
        Total_Current_Assets REAL,
        Land REAL,
        Buildings REAL,
        Equipment REAL,
        Less_accum_depreciation REAL,
        Net_PPE REAL,
        Total_Assets REAL,
        Short_Term_Notes_Payable REAL,
        Accounts_Payable REAL,
        Total_Current_Liabilities REAL,
        Long_Term_Notes_Payable REAL,
        Total_Liabilities REAL,
        Paid_in_Capital REAL,
        Distributions REAL,
        Retained_Earnings REAL,
        Total_Stockholders_Equity REAL,
        Total_Liabilities_Equity REAL
    );''')

    # Load balance_sheet CSV from 'data' folder
    balance_sheet_file = Path(__file__).parent / 'data/balance_sheet_2324.csv'
    df_balance = pd.read_csv(balance_sheet_file)
    df_balance.to_sql('balance_sheet', conn, if_exists='append', index=False)

    # Create the ratios table
    cursor.execute("""
        CREATE TABLE Ratios AS
        SELECT
            income_statement.Month,
            (income_statement.Gross_Profit / income_statement.Sales) AS Profit_Margin,
            ((balance_sheet.Total_Current_Assets - balance_sheet.Inventory) / balance_sheet.Total_Current_Liabilities) AS Quick_Ratio,
            (balance_sheet.Total_Liabilities / balance_sheet.Total_Stockholders_Equity) AS Debt_to_Equity,
            (balance_sheet.Total_Current_Assets / balance_sheet.Total_Current_Liabilities) AS Working_Capital,
            ((balance_sheet.Retained_Earnings - balance_sheet.Distributions) / balance_sheet.Total_Stockholders_Equity) AS Return_on_Equity
        FROM income_statement
        JOIN balance_sheet ON income_statement.Month = balance_sheet.Month;
    """)

    return conn

# Load data
conn = load_data()

# Read the income statement, balance sheet, and ratios data
df_income = pd.read_sql_query("SELECT * FROM income_statement", conn)
df_balance = pd.read_sql_query("SELECT * FROM balance_sheet", conn)
df_ratios = pd.read_sql_query("SELECT * FROM Ratios", conn)

# Create the dashboard layout
st.title(':bar_chart: OJO Dashboard')

# Section 1: Display Ratios Table
st.subheader('Financial Ratios')
st.dataframe(df_ratios)

# Section 2: KPI Table
st.subheader('Key Performance Indicators (KPIs)')
latest_month = df_income['Month'].iloc[-1]
total_revenue = df_income['Sales'].iloc[-1]
total_cogs = df_income['Cost_of_Goods_Sold'].iloc[-1]
net_income = df_income['Net_Income'].iloc[-1]

cols = st.columns(3)
cols[0].metric("Total Revenue", f"${total_revenue:,.2f}")
cols[1].metric("Cost of Goods Sold", f"${total_cogs:,.2f}")
cols[2].metric("Net Income", f"${net_income:,.2f}")

# Section 3: Waterfall Chart
st.subheader('Waterfall Chart: Revenue Breakdown')
fig_waterfall = go.Figure(go.Waterfall(
    name="Revenue Breakdown",
    orientation="v",
    measure=["absolute", "relative", "relative", "relative"],
    x=["Sales", "Cost of Goods Sold", "Operating Income", "Net Income"],
    y=[total_revenue, -total_cogs, df_income['Operating_Income'].iloc[-1], net_income],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
))

fig_waterfall.update_layout(title="Waterfall Chart of Revenue", showlegend=True)
st.plotly_chart(fig_waterfall)

# Section 4: Sales and Operating Income Over Time
st.subheader('Sales and Operating Income Over Time')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_income['Month'], df_income['Sales'], label='Sales')
ax.plot(df_income['Month'], df_income['Operating_Income'], label='Operating Income')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.title('Trends in Sales and Operating Income')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)

# Section 5: Total Revenue and Budget Misses
st.subheader('Total Revenue and Budget Miss Analysis')
budget_target = 1.2 * total_revenue  # Assuming budget is 20% higher than actual sales
revenue_miss = budget_target - total_revenue

cols = st.columns(2)
cols[0].metric("Total Revenue", f"${total_revenue:,.2f}")
cols[1].metric("Budget Miss", f"${revenue_miss:,.2f}", delta_color="inverse")

# Section 6: Correlation Heatmap of Financial Metrics
st.subheader('Correlation Heatmap of Financial Metrics')
cols_to_analyze = ['Sales', 'Gross_Profit', 'Total_Operating_Expenses', 'Net_Income']
df_corr = df_income[cols_to_analyze].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Close connection after fetching data
conn.close()
