import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from openai import OpenAI
import plotly.express as px
import requests

# Set Streamlit configuration
st.set_page_config(
    page_title='OJO Dashboard',
    page_icon=':bar_chart:',
    layout='wide'
)

# QuickBooks API credentials
client_id = st.secrets["quickbooks_client_id"]
client_secret = st.secrets["quickbooks_client_secret"]
redirect_uri = "https://ojo-dashboard-tgftbskcp1m.streamlit.app/"  # Your correct redirect URI
company_id = st.secrets["quickbooks_company_id"]

# Step 1: Authorization URL
auth_url = f"https://appcenter.intuit.com/connect/oauth2" \
           f"?client_id={client_id}" \
           f"&response_type=code" \
           f"&scope=com.intuit.quickbooks.accounting" \
           f"&redirect_uri={redirect_uri}" \
           f"&state=some_state_value"

# Display the authorization link
st.write(f"[Click here to authorize QuickBooks]({auth_url})")

# Step 2: Capture the authorization code from query parameters
query_params = st.query_params  # Get the query parameters from the redirected URL

if 'code' in query_params:
    auth_code = query_params['code']  # Capture the authorization code
    st.success(f"Authorization code received: {auth_code}")

    # Step 3: Exchange authorization code for access and refresh tokens
    def exchange_code_for_tokens(auth_code):
        url = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        payload = {
            'grant_type': 'authorization_code',
            'code': auth_code,
            'redirect_uri': redirect_uri,
            'client_id': client_id,
            'client_secret': client_secret
        }
        
        response = requests.post(url, headers=headers, data=payload)
        
        if response.status_code == 200:
            token_data = response.json()
            st.success("Access token and refresh token obtained!")
            # Store tokens in session state for later use
            st.session_state['access_token'] = token_data['access_token']
            st.session_state['refresh_token'] = token_data['refresh_token']
        else:
            st.error(f"Error fetching access token: {response.status_code} - {response.text}")

    exchange_code_for_tokens(auth_code)

# Step 4: Function to refresh the access token
def refresh_access_token(refresh_token):
    url = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }
    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': client_id,
        'client_secret': client_secret
    }
    
    response = requests.post(url, headers=headers, data=payload)
    
    if response.status_code == 200:
        token_data = response.json()
        st.session_state['access_token'] = token_data['access_token']  # Update the access token
        st.session_state['refresh_token'] = token_data['refresh_token']  # Update refresh token
        st.success("Access token refreshed successfully!")
        return token_data['access_token']
    else:
        st.error(f"Error refreshing access token: {response.status_code} - {response.text}")
        return None

# Step 5: Fetch Company Info to validate connection
if 'access_token' in st.session_state:
    def fetch_company_info(access_token):
        url = f"https://sandbox-quickbooks.api.intuit.com/v3/company/{company_id}/companyinfo/{company_id}"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 401:  # Token expired, refresh it
            st.warning("Access token expired, refreshing token...")
            new_access_token = refresh_access_token(st.session_state['refresh_token'])
            if new_access_token:
                headers['Authorization'] = f'Bearer {new_access_token}'
                response = requests.get(url, headers=headers)  # Retry with new access token

        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame([data])
        else:
            st.error(f"Error fetching company info from QuickBooks: {response.status_code} - {response.text}")
            return pd.DataFrame()

    st.subheader("QuickBooks Company Info")
    df_info = fetch_company_info(st.session_state['access_token'])
    st.dataframe(df_info)

# Step 6: Fetch Balance Sheet report to test availability
    def fetch_balance_sheet_report(access_token):
        url = f"https://sandbox-quickbooks.api.intuit.com/v3/company/{company_id}/report/BalanceSheet"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 401:  # Token expired, refresh it
            st.warning("Access token expired, refreshing token...")
            new_access_token = refresh_access_token(st.session_state['refresh_token'])
            if new_access_token:
                headers['Authorization'] = f'Bearer {new_access_token}'
                response = requests.get(url, headers=headers)  # Retry with new access token

        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data['Rows']['Row'])
        else:
            st.error(f"Error fetching Balance Sheet report from QuickBooks: {response.status_code} - {response.text}")
            return pd.DataFrame()

    st.subheader("QuickBooks Balance Sheet Report")
    df_bs = fetch_balance_sheet_report(st.session_state['access_token'])
    st.dataframe(df_bs)

# Step 7: Fetch Profit and Loss report, handling token expiration
    def fetch_profit_loss_report(access_token):
        url = f"https://sandbox-quickbooks.api.intuit.com/v3/company/{company_id}/report/ProfitAndLoss"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 401:  # Token expired, refresh it
            st.warning("Access token expired, refreshing token...")
            new_access_token = refresh_access_token(st.session_state['refresh_token'])
            if new_access_token:
                headers['Authorization'] = f'Bearer {new_access_token}'
                response = requests.get(url, headers=headers)  # Retry with new access token

        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data['Rows']['Row'])
        else:
            st.error(f"Error fetching data from QuickBooks: {response.status_code} - {response.text}")
            return pd.DataFrame()

    st.subheader("QuickBooks Profit and Loss Report")
    df_pl = fetch_profit_loss_report(st.session_state['access_token'])
    st.dataframe(df_pl)
else:
    st.warning("Please authorize QuickBooks to fetch data.")


# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai_api_key"])



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
            Net_Income REAL,
            Business REAL,
            Region REAL,
            Brand REAL,
            Vice_President REAL,
            State REAL
        );''')

        # Create balance_sheet table


        # Load income_statement CSV from 'data' folder
        income_statement_file = Path(__file__).parent / 'data/income_statement_2324.csv'
        df_income = pd.read_csv(income_statement_file)
        df_income.to_sql('income_statement', conn, if_exists='append', index=False)

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
            Total_Liabilities_Equity REAL,
            Business REAL,
            Region REAL,
            Brand REAL,
            Vice_President REAL,
            State REAL
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

# Section 5: Total Revenue and Budget Miss Analysis
st.subheader('Total Revenue and Budget Miss Analysis')
budget_target = 1.2 * total_revenue  # Assuming budget is 20% higher than actual sales
revenue_miss = budget_target - total_revenue

cols = st.columns(2)
cols[0].metric("Total Revenue", f"${total_revenue:,.2f}")
cols[1].metric("Budget Miss", f"${revenue_miss:,.2f}", delta_color="inverse")

cols_to_analyze = ['Sales', 'Gross_Profit', 'Total_Operating_Expenses', 'Net_Income']
df_corr = df_income[cols_to_analyze].corr()
correlation = df_corr.loc['Sales', 'Total_Operating_Expenses']

# st.subheader("AI-Generated Insights")
# user_input = st.text_area("Ask a question about the data or correlations (e.g., 'Show me the correlations between revenue and expenses'):")
# if st.button("Get Insights"):
#     # Prepare data context (e.g., the most relevant data to analyze)
#     data_context = f"Total Revenue: ${total_revenue:,.2f}, Total Operating Expenses: ${df_income['Total_Operating_Expenses'].iloc[-1]:,.2f}, Correlation between Revenue and Expenses: {correlation:.2f}"
    
#     insights = get_openai_insights(user_input, data_context)
#     st.write(insights)
# Modify the AI Insights section
st.subheader("AI-Generated Insights")
user_input = st.text_area("Ask a question about the data or correlations (e.g., 'Show me the correlations between revenue and expenses'):")

if st.button("Get Insights"):
    # Prepare data context with information from various sections
    kpi_context = f"Total Revenue: ${total_revenue:,.2f}, Total Operating Expenses: ${df_income['Total_Operating_Expenses'].iloc[-1]:,.2f}, Correlation between Revenue and Expenses: {correlation:.2f}"
    
    # Waterfall Chart Data
    waterfall_context = f"Waterfall Chart Data: Sales: ${total_revenue:,.2f}, COGS: ${total_cogs:,.2f}, Operating Income: ${df_income['Operating_Income'].iloc[-1]:,.2f}, Net Income: ${net_income:,.2f}"

    # Financial Ratios Data
    ratios_summary = df_ratios.to_string(index=False)

    # Trends Over Time Data
    trends_summary = df_income_filtered[selected_metrics].to_string(index=False)

    # Tax Analysis Data
    average_tax = df_income['Income_Tax'].mean()
    tax_analysis_context = f"Average Income Tax: ${average_tax:,.2f}, Income Tax Over Time: {df_income[['Month', 'Income_Tax']].to_string(index=False)}"

    # Combine all data into a single context
    combined_context = (
        f"Here is the financial data:\n"
        f"{kpi_context}\n"
        f"{waterfall_context}\n"
        f"Financial Ratios:\n{ratios_summary}\n"
        f"Trends Over Time:\n{trends_summary}\n"
        f"{tax_analysis_context}\n\n{user_input}"
    )

    insights = get_openai_insights(user_input, combined_context)
    st.write(insights)



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

# Section 1: Display Ratios Table
st.subheader('Financial Ratios')
st.dataframe(df_ratios)

# # Section 4: Sales and Operating Income Over Time (with user-selected metrics)
# st.subheader('Trends Over Time')
# fig, ax = plt.subplots(figsize=(10, 6))
# for metric in selected_metrics:
#     ax.plot(df_income['Month'], df_income[metric], label=metric)
# plt.xlabel('Month')
# plt.ylabel('Amount ($)')
# plt.title('Trends in Selected Metrics')
# plt.xticks(rotation=45)
# plt.legend()
# st.pyplot(fig)

# Section 4: Sales and Operating Income Over Time (with user-selected metrics)
st.subheader('Trends Over Time')

# Using Plotly for styling consistency
fig_trends = go.Figure()
for metric in selected_metrics:
    fig_trends.add_trace(go.Scatter(x=df_income['Month'], y=df_income[metric], mode='lines+markers', name=metric))

fig_trends.update_layout(
    title='Trends in Selected Metrics',
    xaxis_title='Month',
    yaxis_title='Amount ($)',
    template='plotly_white',  # Matches the clean look
    title_font=dict(size=20),
    font=dict(size=12),
    xaxis=dict(tickangle=-45),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_trends)


# Section 6: Correlation Heatmap of Financial Metrics
# st.subheader('Correlation Heatmap of Financial Metrics')
cols_to_analyze = ['Sales', 'Gross_Profit', 'Total_Operating_Expenses', 'Net_Income']
df_corr = df_income[cols_to_analyze].corr()

fig, ax = plt.subplots(figsize=(12, 8))
# sns.heatmap(df_corr, annot=True, cmap='coolwarm', ax=ax, linewidths=1.5, linecolor='white', fmt='.2f')
# ax.set_title('Correlation Heatmap of Financial Metrics', fontsize=16)
# st.pyplot(fig)
# Section 6: Correlation Heatmap of Financial Metrics
# st.subheader('Correlation Heatmap of Financial Metrics')

# Convert seaborn heatmap to plotly for styling consistency

fig_heatmap = px.imshow(
    df_corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu",
    title="Correlation Heatmap of Financial Metrics"
)

fig_heatmap.update_layout(
    template="plotly_white",
    title_font=dict(size=20),
    font=dict(size=12),
    coloraxis_colorbar=dict(title="Correlation")
)

st.plotly_chart(fig_heatmap)


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
# st.subheader("AI-Generated Insights")
# user_input = st.text_area("Ask a question about the data or correlations (e.g., 'Show me the correlations between revenue and expenses'):")
# if st.button("Get Insights"):
#     # Prepare data context (e.g., the most relevant data to analyze)
#     data_context = f"Total Revenue: ${total_revenue:,.2f}, Total Operating Expenses: ${df_income['Total_Operating_Expenses'].iloc[-1]:,.2f}, Correlation between Revenue and Expenses: {correlation:.2f}"
    
#     insights = get_openai_insights(user_input, data_context)
#     st.write(insights)

# Calculate the average income tax
average_tax = df_income['Income_Tax'].mean()

# Section 8: Display Average Tax as a KPI
st.subheader('Tax Analysis')
st.metric("Average Income Tax", f"${average_tax:,.2f}")

# Section 9: Tax Breakdown Over Time (Line Chart)
st.subheader('Income Tax Over Time')
fig_tax, ax_tax = plt.subplots(figsize=(10, 6))
ax_tax.plot(df_income['Month'], df_income['Income_Tax'], label='Income Tax', color='purple', marker='o')
plt.xlabel('Month')
plt.ylabel('Tax Amount ($)')
plt.title('Income Tax Over Time')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig_tax)

# Section 10: Add a New Table Showing Total Income, PreTax Income, and Taxes Paid
st.subheader('Income Tax Breakdown Table')
tax_breakdown = df_income[['Month', 'PreTax_Income', 'Income_Tax', 'Net_Income']]
st.dataframe(tax_breakdown)

# Sidebar for business and breakdown filters
st.sidebar.header("Business Filters")

# Check for missing values in the columns to ensure proper dropdown population
# st.write("Checking for missing values in key filter columns:")
# st.write("Business missing:", df_income['Business'].isnull().sum())
# st.write("Region missing:", df_income['Region'].isnull().sum())
# st.write("Brand missing:", df_income['Brand'].isnull().sum())
# st.write("Vice President missing:", df_income['Vice_President'].isnull().sum())
# st.write("State missing:", df_income['State'].isnull().sum())

# Check that the columns exist and are populated
if 'Business' not in df_income.columns or df_income['Business'].isnull().all():
    st.error("The 'Business' column is missing or contains no data.")
if 'Region' not in df_income.columns or df_income['Region'].isnull().all():
    st.error("The 'Region' column is missing or contains no data.")
if 'Brand' not in df_income.columns or df_income['Brand'].isnull().all():
    st.error("The 'Brand' column is missing or contains no data.")
if 'Vice_President' not in df_income.columns or df_income['Vice_President'].isnull().all():
    st.error("The 'Vice_President' column is missing or contains no data.")
if 'State' not in df_income.columns or df_income['State'].isnull().all():
    st.error("The 'State' column is missing or contains no data.")

# Filter by Business
if df_income['Business'].notnull().any():
    selected_business = st.sidebar.selectbox("Select Business", df_income['Business'].dropna().unique())
else:
    selected_business = None

# Filter by Region
if df_income['Region'].notnull().any():
    selected_region = st.sidebar.selectbox("Select Region", df_income['Region'].dropna().unique())
else:
    selected_region = None

# Filter by Brand
if df_income['Brand'].notnull().any():
    selected_brand = st.sidebar.selectbox("Select Brand", df_income['Brand'].dropna().unique())
else:
    selected_brand = None

# Filter by Vice President
if df_income['Vice_President'].notnull().any():
    selected_vp = st.sidebar.selectbox("Select Vice President", df_income['Vice_President'].dropna().unique())
else:
    selected_vp = None

# Filter by State
if df_income['State'].notnull().any():
    selected_state = st.sidebar.selectbox("Select State", df_income['State'].dropna().unique())
else:
    selected_state = None

# Apply filters to the data
if selected_business and selected_region and selected_brand and selected_vp and selected_state:
    df_income_filtered = df_income[
        (df_income['Month'] == selected_month) &
        (df_income['Business'] == selected_business) &
        (df_income['Region'] == selected_region) &
        (df_income['Brand'] == selected_brand) &
        (df_income['Vice_President'] == selected_vp) &
        (df_income['State'] == selected_state)
    ]
    # st.write("Filtered Data:", df_income_filtered)
else:
    st.error("One or more filters are missing data, please check.")

# income_statement_file = Path(__file__).parent / 'data/income_statement_2324.csv'
# df_income = pd.read_csv(income_statement_file)
# st.write(df_income.head())  # Display the first few rows to verify the data


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
