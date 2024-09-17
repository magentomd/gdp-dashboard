import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai_api_key"])

# Set Streamlit configuration
st.set_page_config(
    page_title='OJO Dashboard',
    page_icon=':bar_chart:',
    layout='wide'
)

# Cache the database loading logic using st.cache_data
@st.cache_data
def load_data():
    # Create an SQLite connection for each query
    def query_database(query):
        conn = sqlite3.connect(":memory:")
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

        # Execute the query
        result = pd.read_sql_query(query, conn)

        # Close the connection after each query
        conn.close()

        return result

    # Load dataframes
    df_income = query_database("SELECT * FROM income_statement")
    df_balance = query_database("SELECT * FROM balance_sheet")
    df_ratios = query_database("SELECT * FROM Ratios")

    return df_income, df_balance, df_ratios

# Function to get insights from OpenAI using ChatCompletion
def get_openai_insights(prompt, data_context):
    combined_prompt = f"Here is the financial data:\n{data_context}\n\n{prompt}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": combined_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# Load data
df_income, df_balance, df_ratios = load_data()

# Sidebar for filters
st.sidebar.header("Dashboard Filters")
selected_month = st.sidebar.selectbox("Select Month", df_income['Month'].unique())

# Filtered data based on the selected month
df_income_filtered = df_income[df_income['Month'] == selected_month]

# Sidebar for selecting metrics
st.sidebar.header("Metrics")
selected_metrics = st.sidebar.multiselect(
    "Select Metrics", df_income.columns[1:], default=["Sales", "Net_Income"])

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

# Section 3: Waterfall Chart with Custom Colors
st.subheader('Waterfall Chart: Revenue Breakdown')
fig_waterfall = go.Figure(go.Waterfall(
    name="Revenue Breakdown",
    orientation="v",
    measure=["absolute", "relative", "relative", "relative"],
    x=["Sales", "Cost of Goods Sold", "Operating Income", "Net Income"],
    y=[total_revenue, -total_cogs, df_income['Operating_Income'].iloc[-1], net_income],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    increasing={"marker": {"color": "green"}},
    decreasing={"marker": {"color": "red"}},
    totals={"marker": {"color": "blue"}}
))

fig_waterfall.update_layout(title="Waterfall Chart of Revenue", showlegend=True)
st.plotly_chart(fig_waterfall)

# Section 4: Sales and Operating Income Over Time (with user-selected metrics)
st.subheader('Trends Over Time')
fig, ax = plt.subplots(figsize=(10, 6))
for metric in selected_metrics:
    ax.plot(df_income['Month'], df_income[metric], label=metric)
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.title('Trends in Selected Metrics')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)

# Section 5: Total Revenue and Budget Miss Analysis
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

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', ax=ax, linewidths=1.5, linecolor='white', fmt='.2f')
ax.set_title('Correlation Heatmap of Financial Metrics', fontsize=16)
st.pyplot(fig)

# Analyze Correlation and Provide Insights
st.subheader('Insights')

# Calculate percent of revenue for each metric
df_income['Operating_Expenses_Percent'] = df_income['Total_Operating_Expenses'] / df_income['Sales'] * 100
df_income['Gross_Profit_Percent'] = df_income['Gross_Profit'] / df_income['Sales'] * 100

# Analyze the correlation between Sales and Operating Expenses
correlation = df_corr.loc['Sales', 'Total_Operating_Expenses']
if abs(correlation) < 0.5:
    st.write(f"The correlation between Sales and Total Operating Expenses is low ({correlation:.2f}), indicating that changes in sales do not strongly impact operating expenses. This suggests that your operating costs may be more fixed and not directly tied to revenue fluctuations.")

# Section 7: OpenAI Insights - User can query data and correlation
st.subheader("AI-Generated Insights")
user_input = st.text_area("Ask a question about the data or correlations (e.g., 'Show me the correlations between revenue and expenses'):")
if st.button("Get Insights"):
    # Prepare data context (e.g., the most relevant data to analyze)
    data_context = f"Total Revenue: ${total_revenue:,.2f}, Total Operating Expenses: ${df_income['Total_Operating_Expenses'].iloc[-1]:,.2f}, Correlation between Revenue and Expenses: {correlation:.2f}"
    
    insights = get_openai_insights(user_input, data_context)
    st.write(insights)

# Custom CSS for styling
st.markdown("""
    <style>
        .stMetricLabel {
            font-size: 20px;
            font-weight: bold;
        }
        .stMetricValue {
            color: #FF6347;
        }
        .stDataFrame {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)
