import dash
from dash import html, dcc, Output, Input, State
import requests

# Initialize Dash
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Stock Information Dashboard", style={'text-align': 'center'}),
    dcc.Input(
        id='input-ticker', 
        type='text', 
        placeholder='Enter Ticker (e.g., MDIA)',
        n_submit=0,  # Track enter key presses
        style={'width': '300px', 'padding': '10px', 'margin': '10px'}
    ),
    html.Button(
        'Get Stock Info', 
        id='button-submit', 
        n_clicks=0,
        style={'padding': '10px', 'margin': '10px'}
    ),
    html.Div(id='container-output', style={'margin': '20px', 'padding': '10px'})
], style={'text-align': 'center', 'font-family': 'Arial, sans-serif'})

# Callback for updating stock info
@app.callback(
    Output('container-output', 'children'),
    [Input('button-submit', 'n_clicks'), Input('input-ticker', 'n_submit')],
    [State('input-ticker', 'value')]
)
def update_output(n_clicks, n_submit, ticker):
    # Check if the button was clicked or enter key was pressed
    if n_clicks > 0 or n_submit > 0:
        if ticker:
            ticker = ticker.strip().upper()
            url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey=YOUR_API_KEY'
            response = requests.get(url)
            data = response.json()

            categories = ['top_gainers', 'top_losers', 'most_actively_traded']
            for category in categories:
                for stock in data[category]:
                    if stock['ticker'] == ticker:
                        return html.Div([
                            html.H3(f"{category.replace('_', ' ').title()} - {stock['ticker']}"),
                            html.P(f"Price: {stock['price']}"),
                            html.P(f"Change Amount: {stock['change_amount']}"),
                            html.P(f"Change Percentage: {stock['change_percentage']}"),
                            html.P(f"Volume: {stock['volume']}")
                        ], style={'border': '1px solid #ddd', 'padding': '15px', 'border-radius': '5px', 'background-color': '#f9f9f9'})

    return 'Enter a ticker!'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
