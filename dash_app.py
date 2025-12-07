#backup 2 dash wth black and white apple UI
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pd
import os
import chromadb
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- LlamaIndex Imports (FINAL STABLE VERSION) ---
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq 
from llama_index.core.tools import FunctionTool, QueryEngineTool
# ----------------------------------------------------
# 1. Configuration and Data Loading
# ----------------------------------------------------
PERSIST_DIR = "./invoice_index_storage"
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

# FIX: Function to cache the Chroma client (uses lru_cache)
@lru_cache(maxsize=1)
def get_chroma_client():
    """Initializes and returns the ChromaDB client."""
    print("--- Initializing ChromaDB Client ---")
    return chromadb.Client()

# FIXED: Removed @lru_cache decorator because DataFrames are not hashable
def setup_rag(df):
    """Sets up the standard RAG system and returns the query engine."""
    print("--- Starting RAG Setup ---")
    chroma_client = get_chroma_client()

    # 1. Define the LLM (Groq)
    # TEMPORARY: Hardcoded API key for testing (REMOVE BEFORE PRODUCTION!)
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or "PASTE_YOUR_GROQ_API_KEY_HERE"
    
    if not GROQ_API_KEY or GROQ_API_KEY == "PASTE_YOUR_GROQ_API_KEY_HERE":
        print("=" * 60)
        print("ERROR: GROQ_API_KEY environment variable is not set!")
        print("Please set it using: $env:GROQ_API_KEY = 'your_key_here'")
        print("=" * 60)
        raise ValueError("GROQ_API_KEY is required but not set in environment variables")
    
    print(f"✓ API Key found: {GROQ_API_KEY[:8]}...{GROQ_API_KEY[-4:]}")
    
    llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.1) 
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # --- ALWAYS REBUILD INDEX FOR DEBUGGING ---
    print("--- Building FRESH data index...")
    documents = []
    for index, row in df.iterrows():
        # Enhanced document text with more context for better retrieval
        invoice_text = (
            f"Invoice ID: {row['Invoice_ID']} "
            f"Company: {row['Company_Name']} "
            f"Vendor: {row['Vendor_Name']} "
            f"Type: {row['Type']} "
            f"Amount: {row['Total_Amount']} {row['Currency']} "
            f"Status: {row['Status']} "
            f"Issue Date: {row['Issue_Date']} "
            f"Due Date: {row['Due_Date']} "
            f"Approval Date: {row['Approval_Date']} "
            f"Payment Date: {row['Payment_Date']} "
            f"Approval Required By: {row['Approval_Required_By']} "
            f"Description: {row['Invoice_Description']}"
        )
        doc = Document(text=invoice_text, doc_id=str(row['Invoice_ID']))
        documents.append(doc)
        print(f"  Added document: {row['Invoice_ID']}")

    print(f"--- Total documents to index: {len(documents)} ---")

    # Use a fresh collection name for testing
    collection_name = "invoice_collection_v2"
    try:
        chroma_client.delete_collection(collection_name)
        print(f"--- Deleted old collection: {collection_name} ---")
    except:
        pass
    
    chroma_collection = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("--- Creating vector index (this may take a moment)... ---")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )
    
    print("--- Index created successfully! ---")
    
    # Test the collection
    collection_count = chroma_collection.count()
    print(f"--- ChromaDB collection has {collection_count} documents ---")

    # Create query engine with verbose output
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
        verbose=True
    )
    
    # Test query
    print("\n--- Testing query engine with sample query... ---")
    test_response = query_engine.query("What is invoice INV-1003-AP?")
    print(f"Test response: {test_response.response}")
    print("--- Test complete ---\n")
    
    return query_engine

# --- Tool Integration ---

# Wrapper function for the LLM to call the chart tool
def tool_generate_bar_chart(amount_column: str = 'Total_Amount') -> str:
    """
    Generates and saves a Plotly bar chart from invoice data, grouped by month. 
    It focuses on visualizing a specific 'amount_column' (defaulting to Total_Amount).
    The actual chart rendering is handled by the Dash callback.
    """
    # NOTE: The LLM calls this function. We return a unique string 
    # that the Dash callback can use to know a chart should be displayed.
    return f"CHART_TRIGGERED:generate_bar_chart:column={amount_column}"

# 2. Initialize the tool for LlamaIndex/Groq
CHART_TOOL = FunctionTool.from_defaults(fn=tool_generate_bar_chart, name="generate_chart")

# 3. Create a simple wrapper that handles both query engine and tools
def setup_rag_with_tools(query_engine, tools):
    """Wraps the RAG engine with chart tool capability."""
    # We'll handle tool routing in the callback instead
    # Return both the query engine and tools as a simple wrapper
    class SimpleRAGWrapper:
        def __init__(self, query_engine, tools):
            self.query_engine = query_engine
            self.tools = tools
            self.llm = Settings.llm
        
        def query(self, query_str):
            # Simple logic: check if user is asking for a chart/visualization
            chart_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualization', 'show me', 'display']
            
            if any(keyword in query_str.lower() for keyword in chart_keywords):
                # Return a special response to trigger chart generation
                return type('Response', (), {
                    'response': 'CHART_TRIGGERED:generate_bar_chart:column=Total_Amount'
                })()
            
            # Check for aggregate/list queries that need full DataFrame scan
            aggregate_keywords = [
                'all', 'list', 'show', 'how many', 'total', 'count',
                'overdue', 'pending', 'need action', 'in progress',
                'paid', 'closed', 'approved'
            ]
            
            query_lower = query_str.lower()
            is_aggregate_query = any(keyword in query_lower for keyword in aggregate_keywords)
            
            # HYBRID APPROACH 1: Handle specific invoice ID queries
            import re
            invoice_match = re.search(r'INV-\d{4}-[A-Z]{2}', query_str, re.IGNORECASE)
            
            if invoice_match and not is_aggregate_query:
                invoice_id = invoice_match.group(0).upper()
                print(f"DEBUG: Detected specific invoice query: {invoice_id}")
                
                # Try exact DataFrame lookup first
                from __main__ import df_invoices
                invoice_data = df_invoices[df_invoices['Invoice_ID'] == invoice_id]
                
                if not invoice_data.empty:
                    row = invoice_data.iloc[0]
                    print(f"DEBUG: Found exact match in DataFrame for {invoice_id}")
                    
                    # Create a contextualized query for LLM
                    context = (
                        f"Based on the invoice data:\n"
                        f"Invoice ID: {row['Invoice_ID']}\n"
                        f"Company: {row['Company_Name']}\n"
                        f"Vendor: {row['Vendor_Name']}\n"
                        f"Type: {row['Type']}\n"
                        f"Amount: {row['Total_Amount']} {row['Currency']}\n"
                        f"Status: {row['Status']}\n"
                        f"Issue Date: {row['Issue_Date']}\n"
                        f"Due Date: {row['Due_Date']}\n"
                        f"Approval Date: {row['Approval_Date']}\n"
                        f"Payment Date: {row['Payment_Date']}\n"
                        f"Approval Required By: {row['Approval_Required_By']}\n"
                        f"Description: {row['Invoice_Description']}\n\n"
                        f"User question: {query_str}\n\n"
                        f"Provide a natural, conversational answer."
                    )
                    
                    custom_response = self.llm.complete(context)
                    return type('Response', (), {
                        'response': str(custom_response),
                        'source_nodes': []
                    })()
            
            # HYBRID APPROACH 2: Handle aggregate/list queries with DataFrame
            if is_aggregate_query:
                print(f"DEBUG: Detected aggregate query, using DataFrame analysis")
                from __main__ import df_invoices
                
                # Analyze the DataFrame based on query intent
                result_text = ""
                
                if 'overdue' in query_lower:
                    overdue = df_invoices[df_invoices['Status'] == 'Overdue']
                    if not overdue.empty:
                        result_text = f"There are {len(overdue)} overdue invoices:\n\n"
                        for _, row in overdue.iterrows():
                            result_text += (
                                f"• {row['Invoice_ID']} - {row['Company_Name']} - "
                                f"{row['Total_Amount']} {row['Currency']} - Due: {row['Due_Date']}\n"
                            )
                    else:
                        result_text = "No overdue invoices found."
                
                elif 'pending' in query_lower or 'need action' in query_lower or 'in progress' in query_lower or 'action' in query_lower:
                    # Get all unique status values to ensure we're matching correctly
                    print(f"DEBUG: All statuses in DataFrame: {df_invoices['Status'].unique().tolist()}")
                    
                    action_needed = df_invoices[df_invoices['Status'].isin(['Pending Approval', 'In Progress', 'Overdue'])]
                    
                    print(f"DEBUG: Found {len(action_needed)} invoices needing action")
                    print(f"DEBUG: Status breakdown: {action_needed['Status'].value_counts().to_dict()}")
                    
                    if not action_needed.empty:
                        result_text = f"There are {len(action_needed)} invoices needing action:\n\n"
                        
                        # Group by status for better readability
                        for status in ['Overdue', 'Pending Approval', 'In Progress']:
                            status_invoices = action_needed[action_needed['Status'] == status]
                            if not status_invoices.empty:
                                result_text += f"\n**{status} ({len(status_invoices)}):**\n"
                                for _, row in status_invoices.iterrows():
                                    result_text += (
                                        f"• {row['Invoice_ID']} - {row['Company_Name']} - "
                                        f"{row['Total_Amount']} {row['Currency']} - Due: {row['Due_Date']}\n"
                                    )
                    else:
                        result_text = "No invoices need action."
                
                elif 'paid' in query_lower or 'closed' in query_lower:
                    paid = df_invoices[df_invoices['Status'] == 'Paid/Closed']
                    if not paid.empty:
                        result_text = f"There are {len(paid)} paid/closed invoices:\n\n"
                        for _, row in paid.head(10).iterrows():  # Limit to first 10
                            result_text += (
                                f"• {row['Invoice_ID']} - {row['Company_Name']} - "
                                f"{row['Total_Amount']} {row['Currency']}\n"
                            )
                        if len(paid) > 10:
                            result_text += f"\n... and {len(paid) - 10} more."
                    else:
                        result_text = "No paid/closed invoices found."
                
                elif 'how many' in query_lower or 'count' in query_lower or 'total' in query_lower:
                    total = len(df_invoices)
                    by_status = df_invoices['Status'].value_counts().to_dict()
                    result_text = f"Total invoices: {total}\n\nBreakdown by status:\n"
                    for status, count in by_status.items():
                        result_text += f"• {status}: {count}\n"
                
                if result_text:
                    return type('Response', (), {
                        'response': result_text,
                        'source_nodes': []
                    })()
            
            # HYBRID APPROACH 3: Use regular RAG for other queries
            print("DEBUG: Using regular RAG query")
            response = self.query_engine.query(query_str)
            
            # If RAG response is weak, try DataFrame search
            if not response.response or str(response.response).strip() in ["", "Unknown.", "None"]:
                print("DEBUG: RAG returned weak response, trying DataFrame fallback...")
                if invoice_match:
                    invoice_id = invoice_match.group(0).upper()
                    from __main__ import df_invoices
                    invoice_data = df_invoices[df_invoices['Invoice_ID'] == invoice_id]
                    if not invoice_data.empty:
                        row = invoice_data.iloc[0]
                        fallback_response = (
                            f"Invoice {invoice_id} Details:\n"
                            f"• Company: {row['Company_Name']}\n"
                            f"• Vendor: {row['Vendor_Name']}\n"
                            f"• Amount: {row['Total_Amount']} {row['Currency']}\n"
                            f"• Status: {row['Status']}\n"
                            f"• Due Date: {row['Due_Date']}\n"
                            f"• Type: {row['Type']}"
                        )
                        return type('Response', (), {'response': fallback_response})()
            
            return response
    
    return SimpleRAGWrapper(query_engine, tools)

# --- Initialize RAG once globally ---
STANDARD_RAG_ENGINE = setup_rag(df_invoices)
QUERY_ENGINE = setup_rag_with_tools(
    STANDARD_RAG_ENGINE, 
    tools=[CHART_TOOL]
)

# ----------------------------------------------------
# 2. Application Layout (The Professional UI)
# ----------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[THEME])

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
    # Run the application: http://127.0.0.1:8050/
    app.run(debug=True)