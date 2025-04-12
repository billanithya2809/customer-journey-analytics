import sys
print("Python version:", sys.version)
print("Starting app initialization...")
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots
from datetime import datetime
from io import StringIO

# Initialize Dash app with a modern theme
# Initialize the Dash app with a modern theme
app = dash.Dash(
    __name__, 
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css'
    ]
)
app.title = "Customer Journey Analytics"

# Add this line - this exposes the server variable that gunicorn needs
server = app.server

# Define color scheme for consistency
colors = {
    "primary": "#2c3e50",
    "secondary": "#3498db", 
    "accent": "#e74c3c",
    "background": "#f9f9f9",
    "card": "#ffffff",
    "text": "#333333",
    "text_light": "#7f8c8d",
    "border": "#eaeaea",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "info": "#3498db",
    "funnel_colors": ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c"]
}

# Define CSS styles
styles = {
    "page_container": {
        "margin": "0 auto",
        "padding": "20px",
        "max-width": "1500px",
        "font-family": "'Open Sans', sans-serif",
        "background-color": colors["background"],
        "min-height": "100vh",
    },
    "header": {
        "text-align": "center",
        "padding": "20px 0",
        "margin-bottom": "30px",
        "border-bottom": f"1px solid {colors['border']}",
        "color": colors["primary"],
    },
    "controls_container": {
        "display": "flex",
        "flex-wrap": "wrap",
        "gap": "15px",
        "margin-bottom": "30px",
        "background-color": colors["card"],
        "padding": "20px",
        "border-radius": "8px",
        "box-shadow": "0 2px 6px rgba(0, 0, 0, 0.05)",
    },
    "control_item": {
        "flex": "1",
        "min-width": "200px",
    },
    "section_title": {
        "margin-top": "40px",
        "margin-bottom": "20px",
        "color": colors["primary"],
        "font-weight": "600",
        "border-bottom": f"1px solid {colors['border']}",
        "padding-bottom": "10px",
    },
    "card": {
        "background-color": colors["card"],
        "padding": "20px",
        "border-radius": "8px",
        "box-shadow": "0 2px 6px rgba(0, 0, 0, 0.05)",
        "margin-bottom": "25px",
    },
    "stats_container": {
        "display": "flex",
        "flex-wrap": "wrap",
        "gap": "20px",
        "margin-bottom": "30px",
    },
    "stat_card": {
        "flex": "1 1 200px",
        "background-color": colors["card"],
        "border-radius": "8px",
        "padding": "20px",
        "text-align": "center",
        "box-shadow": "0 2px 6px rgba(0, 0, 0, 0.05)",
    },
    "stat_value": {
        "font-size": "24px",
        "font-weight": "700",
        "color": colors["secondary"],
        "margin": "10px 0",
    },
    "stat_label": {
        "font-size": "14px",
        "color": colors["text_light"],
        "text-transform": "uppercase",
    },
    "chart_grid": {
        "display": "grid",
        "grid-template-columns": "repeat(auto-fit, minmax(500px, 1fr))",
        "gap": "25px",
        "margin-bottom": "40px",
    },
    "chart_card": {
        "background-color": colors["card"],
        "border-radius": "8px",
        "padding": "15px",
        "box-shadow": "0 2px 6px rgba(0, 0, 0, 0.05)",
    },
    "footer": {
        "text-align": "center",
        "padding": "20px",
        "margin-top": "50px",
        "border-top": f"1px solid {colors['border']}",
        "color": colors["text_light"],
    },
    "tab_selected": {
        "border-top": f"3px solid {colors['secondary']}",
        "background-color": colors["card"],
    },
    "tab": {
        "padding": "15px 20px",
        "border-radius": "5px 5px 0 0",
    }
}

# Function to load and process data
def load_data():
    # Load the dataset
    df = pd.read_csv('online_retail_dataset.csv')
    
    # Convert date column to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Add YearMonth column for easier aggregation
    df['YearMonth'] = df['InvoiceDate'].dt.strftime('%Y-%m')
    
    # Calculate revenue for purchase events
    df['Revenue'] = np.where(df['Stage'] == 'Purchase', df['Quantity'] * df['UnitPrice'], 0)
    
    return df

# Load the dataset
df = load_data()

# Get date range for filters
min_date = df['InvoiceDate'].min().date()
max_date = df['InvoiceDate'].max().date()

# Prepare stage and country options for filters
stage_options = [{'label': 'All Stages', 'value': 'all'}] + [
    {'label': stage, 'value': stage} for stage in sorted(df['Stage'].unique())
]

country_options = [{'label': 'All Countries', 'value': 'all'}] + [
    {'label': country, 'value': country} for country in sorted(df['Country'].unique())
]

# App layout
app.layout = html.Div(
    style=styles["page_container"],
    children=[
        # Header
        html.Div(
            style=styles["header"],
            children=[
                html.H1("Customer Journey Analytics Dashboard", style={"margin-bottom": "5px"}),
                html.P(
                    "Analyze conversion funnel, customer behavior patterns and journey metrics",
                    style={"color": colors["text_light"]}
                ),
            ]
        ),
        
        # Control panel
        html.Div(
            style=styles["controls_container"],
            children=[
                html.Div(
                    style=styles["control_item"],
                    children=[
                        html.Label("Date Range:", style={"font-weight": "600", "margin-bottom": "8px", "display": "block"}),
                        dcc.DatePickerRange(
                            id='date-range',
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            initial_visible_month=max_date,
                            start_date=min_date,
                            end_date=max_date,
                            display_format='YYYY-MM-DD',
                        ),
                    ]
                ),
                html.Div(
                    style=styles["control_item"],
                    children=[
                        html.Label("Stage:", style={"font-weight": "600", "margin-bottom": "8px", "display": "block"}),
                        dcc.Dropdown(
                            id='stage-filter',
                            options=stage_options,
                            value='all',
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    style=styles["control_item"],
                    children=[
                        html.Label("Country:", style={"font-weight": "600", "margin-bottom": "8px", "display": "block"}),
                        dcc.Dropdown(
                            id='country-filter',
                            options=country_options,
                            value='all',
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    style=styles["control_item"],
                    children=[
                        html.Label("Update:", style={"font-weight": "600", "margin-bottom": "8px", "display": "block"}),
                        html.Button(
                            'Apply Filters',
                            id='apply-button',
                            style={
                                "background-color": colors["secondary"],
                                "color": "white",
                                "border": "none",
                                "padding": "10px 15px",
                                "border-radius": "4px",
                                "cursor": "pointer",
                                "width": "100%",
                                "font-weight": "600",
                            }
                        ),
                    ]
                ),
            ]
        ),
        
        # KPI Summary Cards
        html.Div(
            style=styles["stats_container"],
            children=[
                html.Div(
                    style=styles["stat_card"],
                    children=[
                        html.Div(style={"display": "flex", "align-items": "center", "justify-content": "center"},
                            children=[
                                html.I(className="fas fa-users", style={"color": colors["info"], "font-size": "20px", "margin-right": "10px"}),
                                html.P("Total Customers", style=styles["stat_label"]),
                            ]
                        ),
                        html.H3(id="total-customers", style=styles["stat_value"]),
                    ]
                ),
                html.Div(
                    style=styles["stat_card"],
                    children=[
                        html.Div(style={"display": "flex", "align-items": "center", "justify-content": "center"},
                            children=[
                                html.I(className="fas fa-chart-line", style={"color": colors["success"], "font-size": "20px", "margin-right": "10px"}),
                                html.P("Conversion Rate", style=styles["stat_label"]),
                            ]
                        ),
                        html.H3(id="conversion-rate", style=styles["stat_value"]),
                    ]
                ),
                html.Div(
                    style=styles["stat_card"],
                    children=[
                        html.Div(style={"display": "flex", "align-items": "center", "justify-content": "center"},
                            children=[
                                html.I(className="fas fa-shopping-cart", style={"color": colors["warning"], "font-size": "20px", "margin-right": "10px"}),
                                html.P("Total Revenue", style=styles["stat_label"]),
                            ]
                        ),
                        html.H3(id="total-revenue", style=styles["stat_value"]),
                    ]
                ),
                html.Div(
                    style=styles["stat_card"],
                    children=[
                        html.Div(style={"display": "flex", "align-items": "center", "justify-content": "center"},
                            children=[
                                html.I(className="fas fa-dollar-sign", style={"color": colors["accent"], "font-size": "20px", "margin-right": "10px"}),
                                html.P("Average Order Value", style=styles["stat_label"]),
                            ]
                        ),
                        html.H3(id="avg-order-value", style=styles["stat_value"]),
                    ]
                ),
            ]
        ),
        
        # Funnel Analysis Section
        html.H2("Funnel Analysis", style=styles["section_title"]),
        
        html.Div(
            style=styles["chart_grid"],
            children=[
                # Funnel Overview
                html.Div(
                    style=styles["chart_card"],
                    children=[
                        html.H3("Customer Journey Funnel", style={"margin-top": "0", "margin-bottom": "15px"}),
                        dcc.Graph(id='funnel-chart', config={'displayModeBar': False})
                    ]
                ),
                
                # Conversion Rates Over Time
                html.Div(
                    style=styles["chart_card"],
                    children=[
                        html.H3("Conversion Rates Trend", style={"margin-top": "0", "margin-bottom": "15px"}),
                        dcc.Graph(id='conversion-trend-chart', config={'displayModeBar': False})
                    ]
                ),
            ]
        ),
        
        # Customer Journey Metrics Section
        html.H2("Customer Journey Metrics", style=styles["section_title"]),
        
        html.Div(
            style=styles["chart_grid"],
            children=[
                # Stage Volumes Over Time
                html.Div(
                    style=styles["chart_card"],
                    children=[
                        html.H3("Journey Stage Volumes", style={"margin-top": "0", "margin-bottom": "15px"}),
                        dcc.Graph(id='stage-volumes-chart', config={'displayModeBar': False})
                    ]
                ),
                
                # Funnel Drop-off Analysis
                html.Div(
                    style=styles["chart_card"],
                    children=[
                        html.H3("Funnel Drop-off Analysis", style={"margin-top": "0", "margin-bottom": "15px"}),
                        dcc.Graph(id='dropoff-chart', config={'displayModeBar': False})
                    ]
                ),
            ]
        ),
        
        # Additional Analysis Section
        html.H2("Additional Insights", style=styles["section_title"]),
        
        html.Div(
            style=styles["chart_grid"],
            children=[
                # Sankey Diagram of Customer Flow
                html.Div(
                    style=styles["chart_card"],
                    children=[
                        html.H3("Customer Journey Flow", style={"margin-top": "0", "margin-bottom": "15px"}),
                        dcc.Graph(id='sankey-chart', config={'displayModeBar': False})
                    ]
                ),
                
                # Monthly Performance Heatmap
                html.Div(
                    style=styles["chart_card"],
                    children=[
                        html.H3("Monthly Conversion Heatmap", style={"margin-top": "0", "margin-bottom": "15px"}),
                        dcc.Graph(id='heatmap-chart', config={'displayModeBar': False})
                    ]
                ),
            ]
        ),
        
        # Footer
        html.Div(
            style=styles["footer"],
            children=[
                html.P(f"Data Range: {min_date} to {max_date}"),
                html.P("Customer Journey Analytics Dashboard © 2025")
            ]
        ),

        # Store filtered data
        dcc.Store(id='filtered-data'),
    ]
)

# Callback to filter data and store it
@app.callback(
    Output('filtered-data', 'data'),
    [Input('apply-button', 'n_clicks')],
    [
        State('date-range', 'start_date'),
        State('date-range', 'end_date'),
        State('stage-filter', 'value'),
        State('country-filter', 'value'),
    ]
)
def filter_data(n_clicks, start_date, end_date, stage, country):
    # Create a copy of the original dataframe
    filtered_df = df.copy()
    
    # Apply date range filter
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['InvoiceDate'] >= start_date) & 
                                 (filtered_df['InvoiceDate'] <= end_date)]
    
    # Apply stage filter
    if stage and stage != 'all':
        filtered_df = filtered_df[filtered_df['Stage'] == stage]
    
    # Apply country filter
    if country and country != 'all':
        filtered_df = filtered_df[filtered_df['Country'] == country]
    
    # Return the filtered dataframe as JSON
    return filtered_df.to_json(date_format='iso', orient='split')

# Callback to update KPI metrics
@app.callback(
    [
        Output('total-customers', 'children'),
        Output('conversion-rate', 'children'),
        Output('total-revenue', 'children'),
        Output('avg-order-value', 'children'),
    ],
    [Input('filtered-data', 'data')]
)
def update_kpis(json_data):
    if not json_data:
        return "N/A", "N/A", "N/A", "N/A"
    
    # Load the filtered dataframe from JSON
    filtered_df = pd.read_json(StringIO(json_data), orient='split')
    
    # Calculate KPIs
    total_customers = filtered_df['CustomerID'].nunique()
    
    # Overall conversion rate (Browse to Purchase)
    browse_count = filtered_df[filtered_df['Stage'] == 'Browse']['CustomerID'].nunique()
    purchase_count = filtered_df[filtered_df['Stage'] == 'Purchase']['CustomerID'].nunique()
    
    conversion_rate = (purchase_count / browse_count) * 100 if browse_count > 0 else 0
    
    # Total revenue from purchases
    purchase_df = filtered_df[filtered_df['Stage'] == 'Purchase']
    total_revenue = (purchase_df['Quantity'] * purchase_df['UnitPrice']).sum()
    
    # Average order value
    if not purchase_df.empty:
        # Group by invoice number and sum the revenue
        orders = purchase_df.groupby('InvoiceNo').apply(
            lambda x: (x['Quantity'] * x['UnitPrice']).sum()
        )
        avg_order_value = orders.mean()
    else:
        avg_order_value = 0
    
    # Format the KPIs
    total_customers_str = f"{total_customers:,}"
    conversion_rate_str = f"{conversion_rate:.1f}%"
    total_revenue_str = f"${total_revenue:,.2f}"
    avg_order_value_str = f"${avg_order_value:.2f}"
    
    return total_customers_str, conversion_rate_str, total_revenue_str, avg_order_value_str

# Callback for Funnel Chart
@app.callback(
    Output('funnel-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_funnel_chart(json_data):
    if not json_data:
        return go.Figure()
    
    # Load the filtered dataframe from JSON
    filtered_df = pd.read_json(StringIO(json_data), orient='split')
    
    # Get unique customers at each stage
    stages = ['Browse', 'View', 'Cart', 'Checkout', 'Purchase']
    stage_counts = []
    
    for stage in stages:
        count = filtered_df[filtered_df['Stage'] == stage]['CustomerID'].nunique()
        stage_counts.append(count)
    
    # Create the funnel chart
    fig = go.Figure(go.Funnel(
        y=stages,
        x=stage_counts,
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.8,
        marker={"color": colors["funnel_colors"]},
        connector={"line": {"color": "royalblue", "width": 1}}
    ))
    
    # Add conversion rate annotations
    annotations = []
    for i in range(len(stages)-1):
        current_stage = stages[i]
        next_stage = stages[i+1]
        
        if stage_counts[i] > 0:
            conversion_rate = stage_counts[i+1] / stage_counts[i]
            
            annotations.append(dict(
                x=0.5,
                y=(i + i+1)/2,  # Position between stages
                xref='paper',
                text=f"{conversion_rate:.1%}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#636363",
                arrowsize=1,
                arrowwidth=2,
                ax=-30,
                ay=0
            ))
    
    fig.update_layout(
        title="",
        annotations=annotations,
        margin=dict(l=60, r=20, t=30, b=30),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for Conversion Trend Chart
@app.callback(
    Output('conversion-trend-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_conversion_trend(json_data):
    if not json_data:
        return go.Figure()
    
    # Load the filtered dataframe from JSON
    filtered_df = pd.read_json(StringIO(json_data), orient='split')
    
    # Add YearMonth column for monthly aggregation
    filtered_df['YearMonth'] = pd.to_datetime(filtered_df['InvoiceDate']).dt.strftime('%Y-%m')
    
    # Define stages
    stages = ['Browse', 'View', 'Cart', 'Checkout', 'Purchase']
    
    # Create dataframe for monthly stage counts
    monthly_data = []
    
    for year_month in sorted(filtered_df['YearMonth'].unique()):
        month_df = filtered_df[filtered_df['YearMonth'] == year_month]
        
        row = {'YearMonth': year_month}
        
        # Count unique customers at each stage
        for stage in stages:
            row[stage] = month_df[month_df['Stage'] == stage]['CustomerID'].nunique()
        
        # Calculate conversion rates
        if row['Browse'] > 0:
            row['Browse_to_View'] = row['View'] / row['Browse']
            row['Overall_Conversion'] = row['Purchase'] / row['Browse']
        else:
            row['Browse_to_View'] = 0
            row['Overall_Conversion'] = 0
            
        if row['View'] > 0:
            row['View_to_Cart'] = row['Cart'] / row['View']
        else:
            row['View_to_Cart'] = 0
            
        if row['Cart'] > 0:
            row['Cart_to_Checkout'] = row['Checkout'] / row['Cart']
        else:
            row['Cart_to_Checkout'] = 0
            
        if row['Checkout'] > 0:
            row['Checkout_to_Purchase'] = row['Purchase'] / row['Checkout']
        else:
            row['Checkout_to_Purchase'] = 0
        
        monthly_data.append(row)
    
    # Convert to dataframe
    monthly_df = pd.DataFrame(monthly_data)
    
    if monthly_df.empty:
        return go.Figure()
    
    # Convert YearMonth to datetime for better axis formatting
    monthly_df['Date'] = pd.to_datetime(monthly_df['YearMonth'] + '-01')
    
    # Create line chart for conversion rates
    fig = go.Figure()
    
    # Add each conversion rate as a line
    conversion_rates = [
        ('Browse_to_View', 'Browse → View', colors["funnel_colors"][0]),
        ('View_to_Cart', 'View → Cart', colors["funnel_colors"][1]),
        ('Cart_to_Checkout', 'Cart → Checkout', colors["funnel_colors"][2]),
        ('Checkout_to_Purchase', 'Checkout → Purchase', colors["funnel_colors"][3]),
        ('Overall_Conversion', 'Overall (Browse → Purchase)', colors["accent"])
    ]
    
    for rate, label, color in conversion_rates:
        fig.add_trace(go.Scatter(
            x=monthly_df['Date'],
            y=monthly_df[rate],
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate='%{x|%b %Y}: %{y:.1%}<extra>' + label + '</extra>'
        ))
    
    # Update layout
    fig.update_layout(
        yaxis=dict(
            tickformat='.0%',
            title="Conversion Rate",
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=False
        ),
        xaxis=dict(
            title="",
            tickformat='%b %Y',
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=20, t=30, b=40),
        height=400,
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for Stage Volumes Chart
@app.callback(
    Output('stage-volumes-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_stage_volumes(json_data):
    if not json_data:
        return go.Figure()
    
    # Load the filtered dataframe from JSON
    filtered_df = pd.read_json(StringIO(json_data), orient='split')
    
    # Add YearMonth column for monthly aggregation
    filtered_df['YearMonth'] = pd.to_datetime(filtered_df['InvoiceDate']).dt.strftime('%Y-%m')
    
    # Define stages
    stages = ['Browse', 'View', 'Cart', 'Checkout', 'Purchase']
    
    # Create dataframe for monthly stage counts
    monthly_data = []
    
    for year_month in sorted(filtered_df['YearMonth'].unique()):
        month_df = filtered_df[filtered_df['YearMonth'] == year_month]
        
        row = {'YearMonth': year_month}
        
        # Count unique customers at each stage
        for stage in stages:
            row[stage] = month_df[month_df['Stage'] == stage]['CustomerID'].nunique()
        
        monthly_data.append(row)
    
    # Convert to dataframe
    monthly_df = pd.DataFrame(monthly_data)
    
    if monthly_df.empty:
        return go.Figure()
    
    # Convert YearMonth to datetime for better axis formatting
    monthly_df['Date'] = pd.to_datetime(monthly_df['YearMonth'] + '-01')
    
    # Create line chart for stage volumes
    fig = go.Figure()
    
    # Add each stage as a line
    for i, stage in enumerate(stages):
        fig.add_trace(go.Scatter(
            x=monthly_df['Date'],
            y=monthly_df[stage],
            mode='lines+markers',
            name=stage,
            line=dict(color=colors["funnel_colors"][i], width=2),
            marker=dict(size=6),
            hovertemplate='%{x|%b %Y}: %{y:,} customers<extra>' + stage + '</extra>'
        ))
    
    # Update layout
    fig.update_layout(
        yaxis=dict(
            title="Number of Customers",
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=False
        ),
        xaxis=dict(
            title="",
            tickformat='%b %Y',
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=20, t=30, b=40),
        height=400,
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for Drop-off Analysis Chart
@app.callback(
    Output('dropoff-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_dropoff_chart(json_data):
    if not json_data:
        return go.Figure()
    
    # Load the filtered dataframe from JSON
    filtered_df = pd.read_json(StringIO(json_data), orient='split')
    
    # Define stages
    stages = ['Browse', 'View', 'Cart', 'Checkout', 'Purchase']
    
    # Count unique customers at each stage
    stage_counts = []
    for stage in stages:
        count = filtered_df[filtered_df['Stage'] == stage]['CustomerID'].nunique()
        stage_counts.append(count)
    
    # Calculate drop-offs
    dropoffs = []
    dropoff_percentages = []
    
    for i in range(len(stages) - 1):
        if stage_counts[i] > 0:
            dropoff = stage_counts[i] - stage_counts[i+1]
            dropoff_percentage = (dropoff / stage_counts[i]) * 100
        else:
            dropoff = 0
            dropoff_percentage = 0
            
        dropoffs.append(dropoff)
        dropoff_percentages.append(dropoff_percentage)
    
    # Create labels for transitions
    transition_labels = [f"{stages[i]} → {stages[i+1]}" for i in range(len(stages) - 1)]
    
    # Create the bar chart
    fig = go.Figure()
    
    # Add drop-off bars
    fig.add_trace(go.Bar(
        x=transition_labels,
        y=dropoff_percentages,
        text=[f"{p:.1f}%" for p in dropoff_percentages],
        textposition='auto',
        marker_color=colors["accent"],
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>Drop-off: %{y:.1f}%<br>(%{customdata:,} customers)<extra></extra>',
        customdata=dropoffs
    ))
    
    # Update layout
    fig.update_layout(
        yaxis=dict(
            title="Drop-off Percentage",
            tickformat='.0%',
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=False,
            range=[0, max(dropoff_percentages) * 1.1 if dropoff_percentages else 100]
        ),
        xaxis=dict(
            title="",
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        margin=dict(l=40, r=20, t=30, b=40),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for Sankey Chart
@app.callback(
    Output('sankey-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_sankey_chart(json_data):
    if not json_data:
        return go.Figure()
    
    # Load the filtered dataframe from JSON
    filtered_df = pd.read_json(StringIO(json_data), orient='split')
    
    # Define stages
    stages = ['Browse', 'View', 'Cart', 'Checkout', 'Purchase']
    
    # Count unique customers at each stage
    stage_counts = []
    for stage in stages:
        count = filtered_df[filtered_df['Stage'] == stage]['CustomerID'].nunique()
        stage_counts.append(count)
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = stages,
            color = colors["funnel_colors"]
        ),
        link = dict(
            source = [0, 1, 2, 3],  # Browse, View, Cart, Checkout
            target = [1, 2, 3, 4],  # View, Cart, Checkout, Purchase
            value = [
                stage_counts[1],  # Browse to View
                stage_counts[2],  # View to Cart
                stage_counts[3],  # Cart to Checkout
                stage_counts[4]   # Checkout to Purchase
            ],
            color = ["rgba(52, 152, 219, 0.4)", "rgba(46, 204, 113, 0.4)", 
                     "rgba(243, 156, 18, 0.4)", "rgba(231, 76, 60, 0.4)"]
        )
    )])
    
    fig.update_layout(
        margin=dict(l=40, r=20, t=30, b=40),
        height=400,
        font=dict(size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback for Heatmap Chart
@app.callback(
    Output('heatmap-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_heatmap_chart(json_data):
    if not json_data:
        return go.Figure()
    
    # Load the filtered dataframe from JSON
    filtered_df = pd.read_json(StringIO(json_data), orient='split')
    
    # Add YearMonth column for monthly aggregation
    filtered_df['YearMonth'] = pd.to_datetime(filtered_df['InvoiceDate']).dt.strftime('%Y-%m')
    
    # Define stages
    stages = ['Browse', 'View', 'Cart', 'Checkout', 'Purchase']
    
    # Create dataframe for monthly stage counts
    monthly_data = []
    
    for year_month in sorted(filtered_df['YearMonth'].unique()):
        month_df = filtered_df[filtered_df['YearMonth'] == year_month]
        
        row = {'YearMonth': year_month}
        
        # Count unique customers at each stage
        for stage in stages:
            row[stage] = month_df[month_df['Stage'] == stage]['CustomerID'].nunique()
        
        # Calculate conversion rates
        if row['Browse'] > 0:
            row['Browse_to_View'] = row['View'] / row['Browse']
            row['Overall_Conversion'] = row['Purchase'] / row['Browse']
        else:
            row['Browse_to_View'] = 0
            row['Overall_Conversion'] = 0
            
        if row['View'] > 0:
            row['View_to_Cart'] = row['Cart'] / row['View']
        else:
            row['View_to_Cart'] = 0
            
        if row['Cart'] > 0:
            row['Cart_to_Checkout'] = row['Checkout'] / row['Cart']
        else:
            row['Cart_to_Checkout'] = 0
            
        if row['Checkout'] > 0:
            row['Checkout_to_Purchase'] = row['Purchase'] / row['Checkout']
        else:
            row['Checkout_to_Purchase'] = 0
        
        monthly_data.append(row)
    
    # Convert to dataframe
    monthly_df = pd.DataFrame(monthly_data)
    
    if monthly_df.empty or len(monthly_df) < 2:
        # Return empty figure with message if not enough data
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for heatmap visualization",
            showarrow=False,
            font=dict(size=14),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5
        )
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # Get last 12 months (or all if less than 12)
    max_months = min(12, len(monthly_df))
    monthly_df = monthly_df.iloc[-max_months:]
    
    # Prepare data for heatmap
    conversion_rates = ['Browse_to_View', 'View_to_Cart', 'Cart_to_Checkout', 
                        'Checkout_to_Purchase', 'Overall_Conversion']
    conversion_labels = ['Browse → View', 'View → Cart', 'Cart → Checkout', 
                         'Checkout → Purchase', 'Overall']
    
    # Create heatmap data
    z_data = []
    for rate in conversion_rates:
        z_data.append(monthly_df[rate].values)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=monthly_df['YearMonth'],
        y=conversion_labels,
        colorscale='blues',
        zmin=0,
        zmax=1,
        text=[[f"{v:.1%}" for v in row] for row in z_data],
        texttemplate="%{text}",
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis=dict(
            title="Month",
            tickangle=45
        ),
        yaxis=dict(
            title="Conversion Type",
            autorange="reversed"  # To have overall at the bottom
        ),
        margin=dict(l=40, r=20, t=30, b=80),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Run the server
# Run the server
# Run the app
if __name__ == '__main__':
    # For local development
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))