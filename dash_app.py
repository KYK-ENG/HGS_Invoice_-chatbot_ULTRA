import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pd
import os
import plotly.express as px
import requests
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
# ----------------------------------------------------
# 1. Configuration and Data Loading
# ----------------------------------------------------
CSV_FILE_PATH = 'invoice_data.csv'
THEME = dbc.themes.BOOTSTRAP  # Using Bootstrap as base for custom styling

# --- Load Invoice Data ---
def load_data(file_path):
    """Loads the invoice data from the CSV file."""
    try:
        df = pd.read_csv(file_path)
        date_cols = ['Issue_Date', 'Due_Date', 'Approval_Date', 'Payment_Date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df_invoices = load_data(CSV_FILE_PATH)
if df_invoices.empty:
    print("FATAL ERROR: Data not loaded. Please ensure 'invoice_data.csv' is correctly placed.")
else:
    # DEBUG: Print actual status values to help with matching
    print(f"\n{'='*60}")
    print(f"DEBUG: Loaded {len(df_invoices)} invoices")
    print(f"DEBUG: Status values in CSV: {df_invoices['Status'].unique().tolist()}")
    print(f"DEBUG: Status counts: {df_invoices['Status'].value_counts().to_dict()}")
    print(f"{'='*60}\n")


# --- Groq API Function (Lightweight) ---
def query_groq(prompt):
    """Send query to Groq API directly."""
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not set in environment variables."
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: Groq API returned status {response.status_code}"
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"


# --- Data Visualization Function (The Tool Python Function) ---
def generate_bar_chart(df: pd.DataFrame, amount_column: str = 'Total_Amount'):
    """
    Groups invoice data by month and year, calculates the total amount,
    and returns a Plotly bar chart figure.
    """
    if df.empty:
        return None

    df = df.dropna(subset=['Issue_Date']).copy()
    if df.empty:
        return None

    df['Month_Year'] = df['Issue_Date'].dt.to_period('M')
    monthly_totals = df.groupby('Month_Year')[amount_column].sum().reset_index()
    monthly_totals['Date'] = monthly_totals['Month_Year'].apply(lambda x: x.start_time)
    
    fig = px.bar(
        monthly_totals, 
        x='Date', 
        y=amount_column, 
        title='Invoice Totals Grouped by Month',
        labels={'Date': 'Month', amount_column: 'Total Amount'},
        template='plotly_dark' 
    )
    
    fig.update_xaxes(dtick="M1", tickformat="%b\n%Y") 
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# --- RAG Initialization Functions (Cached and Isolated) ---

# Simplified query handler - no heavy dependencies
class SimpleQueryEngine:
    """Lightweight query engine using DataFrame and Groq API directly."""
    
    def __init__(self, df):
        self.df = df
    
    def query(self, query_str):
        """Process query using DataFrame lookups and Groq API."""
        query_lower = query_str.lower()
        
        # Check for chart requests
        chart_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualization', 'show me', 'display']
        if any(keyword in query_lower for keyword in chart_keywords):
            return type('Response', (), {
                'response': 'CHART_TRIGGERED:generate_bar_chart:column=Total_Amount'
            })()
        
        # Check for specific invoice ID
        import re
        invoice_match = re.search(r'INV-\d{4}-[A-Z]{2}', query_str, re.IGNORECASE)
        
        if invoice_match:
            invoice_id = invoice_match.group(0).upper()
            print(f"DEBUG: Looking up invoice: {invoice_id}")
            
            invoice_data = self.df[self.df['Invoice_ID'] == invoice_id]
            
            if not invoice_data.empty:
                row = invoice_data.iloc[0]
                
                # Build response based on query type
                if 'status' in query_lower:
                    response_text = (
                        f"Invoice {invoice_id} has a status of **{row['Status']}**.\n\n"
                        f"Details:\n"
                        f"• Company: {row['Company_Name']}\n"
                        f"• Vendor: {row['Vendor_Name']}\n"
                        f"• Amount: {row['Total_Amount']} {row['Currency']}\n"
                        f"• Due Date: {row['Due_Date']}"
                    )
                elif 'amount' in query_lower or 'cost' in query_lower:
                    response_text = (
                        f"Invoice {invoice_id} is for **{row['Total_Amount']} {row['Currency']}**.\n\n"
                        f"• Company: {row['Company_Name']}\n"
                        f"• Status: {row['Status']}\n"
                        f"• Due Date: {row['Due_Date']}"
                    )
                else:
                    response_text = (
                        f"Invoice {invoice_id} Information:\n\n"
                        f"• Status: {row['Status']}\n"
                        f"• Company: {row['Company_Name']}\n"
                        f"• Vendor: {row['Vendor_Name']}\n"
                        f"• Amount: {row['Total_Amount']} {row['Currency']}\n"
                        f"• Due Date: {row['Due_Date']}\n"
                        f"• Type: {row['Type']}"
                    )
                
                return type('Response', (), {'response': response_text})()
            else:
                return type('Response', (), {
                    'response': f"Invoice {invoice_id} not found in the system."
                })()
        
        # Handle aggregate queries
        aggregate_keywords = ['all', 'list', 'show', 'how many', 'overdue', 'pending', 'action']
        if any(keyword in query_lower for keyword in aggregate_keywords):
            
            if 'overdue' in query_lower:
                overdue = self.df[self.df['Status'] == 'Overdue']
                if not overdue.empty:
                    result = f"There are {len(overdue)} overdue invoices:\n\n"
                    for _, row in overdue.iterrows():
                        result += f"• {row['Invoice_ID']} - {row['Company_Name']} - {row['Total_Amount']} {row['Currency']}\n"
                    return type('Response', (), {'response': result})()
            
            elif 'pending' in query_lower or 'action' in query_lower:
                action_needed = self.df[self.df['Status'].isin(['Pending Approval', 'In Progress', 'Overdue'])]
                if not action_needed.empty:
                    result = f"There are {len(action_needed)} invoices needing action:\n\n"
                    for _, row in action_needed.iterrows():
                        result += f"• {row['Invoice_ID']} - {row['Status']} - {row['Total_Amount']} {row['Currency']}\n"
                    return type('Response', (), {'response': result})()
        
        # For general questions, use Groq API with DataFrame context
        context = f"Based on this invoice data summary: {len(self.df)} total invoices. "
        context += f"Status breakdown: {self.df['Status'].value_counts().to_dict()}. "
        context += f"\n\nUser question: {query_str}"
        
        groq_response = query_groq(context)
        return type('Response', (), {'response': groq_response})()


# Initialize the query engine
print("--- Initializing Lightweight Query Engine ---")
QUERY_ENGINE = SimpleQueryEngine(df_invoices)
print("✓ Query engine ready!")


# ----------------------------------------------------
# 2. Application Layout (The Professional UI)
# ----------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[THEME])
server = app.server  # Expose server for gunicorn deployment

# Custom CSS for Pure Black & White Minimal Design
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }
            
            body {
                background: #000000;
                color: #ffffff;
                margin: 0;
                padding: 0;
            }
            
            .main-container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 40px 20px;
            }
            
            /* Header Styling */
            .app-header {
                text-align: center;
                margin-bottom: 60px;
                padding: 40px 0;
            }
            
            .app-title {
                font-size: 48px;
                font-weight: 700;
                letter-spacing: -2px;
                color: #ffffff;
                margin-bottom: 12px;
            }
            
            .app-subtitle {
                font-size: 16px;
                font-weight: 400;
                color: #666666;
                letter-spacing: 0.5px;
            }
            
            /* Metric Cards */
            .metric-card {
                background: #ffffff;
                border: 1px solid #ffffff;
                border-radius: 2px;
                padding: 32px 28px;
                transition: all 0.2s ease;
                height: 100%;
            }
            
            .metric-card:hover {
                background: #f5f5f5;
                transform: translateY(-2px);
            }
            
            .metric-label {
                font-size: 11px;
                font-weight: 600;
                color: #666666;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                margin-bottom: 16px;
            }
            
            .metric-value {
                font-size: 40px;
                font-weight: 700;
                letter-spacing: -2px;
                color: #000000;
                margin-bottom: 8px;
                line-height: 1;
            }
            
            .metric-description {
                font-size: 13px;
                color: #999999;
                font-weight: 400;
            }
            
            /* Search Section */
            .search-section {
                background: #ffffff;
                border: 1px solid #ffffff;
                border-radius: 2px;
                padding: 40px 36px;
                margin: 40px 0;
            }
            
            .search-label {
                font-size: 13px;
                font-weight: 600;
                color: #666666;
                margin-bottom: 20px;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }
            
            .form-control-lg {
                background: #f5f5f5 !important;
                border: 1px solid #e0e0e0 !important;
                border-radius: 2px !important;
                color: #000000 !important;
                font-size: 15px !important;
                padding: 16px 20px !important;
                transition: all 0.2s ease !important;
                font-weight: 400 !important;
            }
            
            .form-control-lg:focus {
                background: #ffffff !important;
                border-color: #000000 !important;
                box-shadow: none !important;
                outline: none !important;
            }
            
            .form-control-lg::placeholder {
                color: #999999 !important;
            }
            
            .btn-search {
                background: #000000 !important;
                color: #ffffff !important;
                border: 1px solid #000000 !important;
                border-radius: 2px !important;
                font-weight: 600 !important;
                font-size: 13px !important;
                padding: 18px 32px !important;
                transition: all 0.2s ease !important;
                letter-spacing: 1px !important;
                text-transform: uppercase !important;
            }
            
            .btn-search:hover {
                background: #333333 !important;
                border-color: #333333 !important;
                transform: translateY(-1px);
            }
            
            .btn-search:active {
                transform: translateY(0);
            }
            
            /* Result Card */
            .result-card {
                background: #ffffff;
                border: 1px solid #ffffff;
                border-radius: 2px;
                padding: 0;
                overflow: hidden;
                margin-top: 40px;
            }
            
            .result-header {
                background: #f5f5f5;
                padding: 20px 32px;
                border-bottom: 1px solid #e0e0e0;
            }
            
            .result-title {
                font-size: 13px;
                font-weight: 600;
                color: #000000;
                margin: 0;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            
            .result-body {
                padding: 32px;
                background: #ffffff;
            }
            
            .query-text {
                font-size: 14px;
                font-weight: 600;
                color: #000000;
                margin-bottom: 20px;
                padding-bottom: 20px;
                border-bottom: 1px solid #e0e0e0;
            }
            
            .response-text {
                font-size: 15px;
                line-height: 1.8;
                color: #333333;
                white-space: pre-wrap;
                font-weight: 400;
            }
            
            /* Graph styling */
            .graph-card {
                background: #ffffff;
                border: 1px solid #ffffff;
                border-radius: 2px;
                padding: 32px;
                margin-top: 32px;
            }
            
            /* Footer */
            .app-footer {
                text-align: center;
                margin-top: 80px;
                padding: 40px 0;
                border-top: 1px solid #222222;
            }
            
            .footer-text {
                font-size: 11px;
                color: #666666;
                font-weight: 400;
                letter-spacing: 0.5px;
            }
            
            /* Divider */
            hr {
                border: none;
                height: 1px;
                background: #222222;
                margin: 60px 0;
            }
            
            /* Animation */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .fade-in {
                animation: fadeIn 0.4s ease-out;
            }
            
            /* Error state */
            .error-title {
                color: #000000 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Calculate metrics for the cards
total_invoices = len(df_invoices)
open_invoices = df_invoices[df_invoices['Status'].isin(['Pending Approval', 'In Progress', 'Overdue'])].shape[0]
total_value_usd = df_invoices[df_invoices['Currency'] == 'USD']['Total_Amount'].sum()
overdue_invoices = df_invoices[df_invoices['Status'] == 'Overdue'].shape[0]

# UI Components - Pure B&W metric cards
metric_cards = [
    dbc.Col(html.Div([
        html.Div("Total Invoices", className="metric-label"),
        html.Div(f"{total_invoices:,}", className="metric-value"),
        html.Div("Records in system", className="metric-description")
    ], className="metric-card"), md=3),
    
    dbc.Col(html.Div([
        html.Div("Need Action", className="metric-label"),
        html.Div(f"{open_invoices:,}", className="metric-value"),
        html.Div("Pending / In Progress", className="metric-description")
    ], className="metric-card"), md=3),
    
    dbc.Col(html.Div([
        html.Div("Overdue", className="metric-label"),
        html.Div(f"{overdue_invoices:,}", className="metric-value"),
        html.Div("Require attention", className="metric-description")
    ], className="metric-card"), md=3),
    
    dbc.Col(html.Div([
        html.Div("Total Value", className="metric-label"),
        html.Div(f"${total_value_usd:,.0f}", className="metric-value"),
        html.Div("USD invoices", className="metric-description")
    ], className="metric-card"), md=3)
]

# Main Layout - Redesigned with Apple aesthetic
app.layout = html.Div([
    dbc.Container([
        # Header
        html.Div([
            html.H1("Invoice Intelligence", className="app-title"),
            html.P("AI-Powered Query System for AP/AR Teams", className="app-subtitle")
        ], className="app-header fade-in"),
        
        # Metric Cards
        dbc.Row(metric_cards, className="mb-5 fade-in"),
        
        # Search Section
        html.Div([
            html.Div("Ask anything about your invoices", className="search-label"),
            dbc.Row([
                dbc.Col(dbc.Input(
                    id='query-input',
                    placeholder='e.g., What is the status of INV-1003-AP? Show overdue invoices...',
                    type='text',
                    className="form-control-lg",
                    n_submit=0
                ), width=9),
                dbc.Col(dbc.Button(
                    'Search', 
                    id='query-button', 
                    n_clicks=0,
                    className="btn-search w-100"
                ), width=3)
            ], className="align-items-center")
        ], className="search-section fade-in"),
        
        # Result Display Area
        html.Div(id='result-container', children=[
            html.Div([
                html.Div([
                    html.H4("AI Response", className="result-title")
                ], className="result-header"),
                html.Div([
                    html.Div(
                        id='query-output',
                        children="Enter a query above and press Enter or click Search to see results here.",
                        className="response-text"
                    )
                ], className="result-body")
            ], className="result-card fade-in")
        ]),
        
        # Graph Output Area
        html.Div(id='graph-output-area'),
        
        # Footer
        html.Div([
            html.P([
                "Powered by ",
                html.Span("LlamaIndex", style={"font-weight": "600"}),
                " • ",
                html.Span("Groq", style={"font-weight": "600"}),
                " • ",
                html.Span("ChromaDB", style={"font-weight": "600"})
            ], className="footer-text")
        ], className="app-footer")
    ], fluid=True, className="main-container")
], style={"minHeight": "100vh"})


# ----------------------------------------------------
# 3. Callbacks (The RAG Execution Logic)
# ----------------------------------------------------

@app.callback(
    [Output('query-output', 'children'),
     Output('graph-output-area', 'children')],
    [Input('query-button', 'n_clicks'),
     Input('query-input', 'n_submit')],  # Added: Listen for Enter key
    [State('query-input', 'value')]
)
def run_query(n_clicks, n_submit, user_query):
    # Initialize returns
    if (n_clicks is None or n_clicks == 0) and (n_submit is None or n_submit == 0):
        return dash.no_update, html.Div()
    
    if not user_query:
        return dash.no_update, html.Div()

    try:
        if not user_query:
            return "Please enter a valid query.", html.Div()
            
        print(f"\n{'='*60}")
        print(f"DEBUG: Executing query: {user_query}")
        print(f"{'='*60}")
        
        # Execute the RAG query
        response = QUERY_ENGINE.query(user_query)
        
        print(f"DEBUG: Response object: {response}")
        print(f"DEBUG: Response type: {type(response)}")
        print(f"DEBUG: Has response attr: {hasattr(response, 'response')}")
        
        if hasattr(response, 'response'):
            print(f"DEBUG: Response.response value: '{response.response}'")
            print(f"DEBUG: Response.response type: {type(response.response)}")
        
        if hasattr(response, 'source_nodes'):
            print(f"DEBUG: Number of source nodes: {len(response.source_nodes)}")
            for i, node in enumerate(response.source_nodes[:3]):
                print(f"DEBUG: Source {i+1}: {node.text[:100]}...")
        
        print(f"{'='*60}\n")

        # 1. Check if the LLM decided to call the chart tool
        response_text = str(response.response) if hasattr(response, 'response') else str(response)
        
        print(f"DEBUG: Final response_text: '{response_text}'")
        
        if "CHART_TRIGGERED:generate_bar_chart" in response_text:
            # The LLM triggered the chart. Extract column information if available.
            
            # --- Generate the Chart Figure ---
            fig = generate_bar_chart(df_invoices)
            
            if fig is None:
                 return html.Div([
                     html.Div([
                         html.Div([
                             html.H4("Error", className="result-title")
                         ], className="result-header"),
                         html.Div([
                             html.Div("Sorry, I cannot generate a chart. The data may be missing or incomplete.", className="response-text")
                         ], className="result-body")
                     ], className="result-card fade-in")
                 ]), html.Div()

            # Create the Plotly Graph component
            graph = dcc.Graph(figure=fig, config={'displayModeBar': False})
            
            # Return confirmation message and the graph
            return html.Div([
                html.Div([
                    html.Div([
                        html.H4("AI Response", className="result-title")
                    ], className="result-header"),
                    html.Div([
                        html.Div(f"Query: {user_query}", className="query-text"),
                        html.Div("Chart generated successfully showing monthly invoice totals.", className="response-text")
                    ], className="result-body")
                ], className="result-card fade-in")
            ]), html.Div(dcc.Graph(figure=fig, config={'displayModeBar': False}), className="graph-card fade-in")

        # 2. If the LLM did not call the tool, proceed with normal text response
        
        # Check if response is empty
        if not response_text or response_text.strip() == "" or response_text.lower() in ["none", "empty response"]:
            print("DEBUG: Response is empty, trying DataFrame fallback...")
            
            # Try DataFrame fallback
            import re
            invoice_match = re.search(r'INV-\d{4}-[A-Z]{2}', user_query, re.IGNORECASE)
            if invoice_match:
                invoice_id = invoice_match.group(0).upper()
                print(f"DEBUG: Found invoice ID in query: {invoice_id}")
                
                invoice_data = df_invoices[df_invoices['Invoice_ID'] == invoice_id]
                if not invoice_data.empty:
                    row = invoice_data.iloc[0]
                    print(f"DEBUG: Found invoice in DataFrame!")
                    fallback_response = (
                        f"Found invoice {invoice_id}:\n"
                        f"• Company: {row['Company_Name']}\n"
                        f"• Vendor: {row['Vendor_Name']}\n"
                        f"• Amount: {row['Total_Amount']} {row['Currency']}\n"
                        f"• Status: {row['Status']}\n"
                        f"• Due Date: {row['Due_Date']}\n"
                        f"• Type: {row['Type']}"
                    )
                    return html.Div([
                        html.Div([
                            html.Div([
                                html.H4("AI Response", className="result-title")
                            ], className="result-header"),
                            html.Div([
                                html.Div(f"Query: {user_query}", className="query-text"),
                                html.Pre(fallback_response, className="response-text")
                            ], className="result-body")
                        ], className="result-card fade-in")
                    ]), html.Div()
                else:
                    print(f"DEBUG: Invoice {invoice_id} not found in DataFrame")
            
            return html.Div([
                html.Div([
                    html.Div([
                        html.H4("No Results", className="result-title")
                    ], className="result-header"),
                    html.Div([
                        html.Div(f"Query: {user_query}", className="query-text"),
                        html.Div("I couldn't find any information matching your query. Please try rephrasing or check if the invoice ID exists.", className="response-text")
                    ], className="result-body")
                ], className="result-card fade-in")
            ]), html.Div()
        
        # Return the AI's response formatted nicely
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("AI Response", className="result-title")
                ], className="result-header"),
                html.Div([
                    html.Div(f"Query: {user_query}", className="query-text"),
                    html.Div(response_text, className="response-text")
                ], className="result-body")
            ], className="result-card fade-in")
        ]), html.Div()

    except Exception as e:
        print(f"ERROR during RAG query: {e}")
        import traceback
        traceback.print_exc()
        # Return a clear error message to the user
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Error", className="result-title error-title")
                ], className="result-header"),
                html.Div([
                    html.Div(f"An error occurred: {str(e)}", className="response-text")
                ], className="result-body")
            ], className="result-card fade-in")
        ]), html.Div()


# ----------------------------------------------------
# 4. Run the App
# ----------------------------------------------------

if __name__ == '__main__':
    # NOTE: Ensure your GROQ_API_KEY is set as an environment variable
    # For local: app.run(debug=True)
    # For production: gunicorn will handle the server
    import os
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
