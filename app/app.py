import dash
from dash import html, dcc, Output, Input, State
import requests

# Initialize Dash
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Stock Information Lookup", style={'text-align': 'center'}),
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
    if n_clicks > 0 or n_submit > 0:
        if ticker:
            ticker = ticker.strip().upper()
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=YOUR_API_KEY'
            response = requests.get(url)
            data = response.json()

            try:
                daily_data = data['Time Series (Daily)']
                recent_dates = sorted(daily_data.keys(), reverse=True)[:7]  # Last 7 days

                def calculate_change(previous, current):
                    return round(current - previous, 2), round((current - previous) / previous * 100, 2)

                historical_info = []
                for i, date in enumerate(recent_dates):
                    close_price = float(daily_data[date]['4. close'])
                    volume = daily_data[date]['5. volume']

                    if i < len(recent_dates) - 1:
                        prev_close = float(daily_data[recent_dates[i + 1]]['4. close'])
                        change_amt, change_pct = calculate_change(prev_close, close_price)
                    else:
                        change_amt, change_pct = 0, 0

                    historical_info.append({
                        "date": date,
                        "close_price": close_price,
                        "change_amount": change_amt,
                        "change_percentage": change_pct,
                        "volume": volume
                    })

                return html.Div([
                    html.H3(f"Stock Data for {ticker} - Last 7 Days"),
                    html.Table([
                        html.Thead(html.Tr([html.Th("Date", style={'padding': '0 15px', 'text-align': 'left'}), html.Th("Price", style={'padding': '0 15px', 'text-align': 'left'}), html.Th("Change Amt", style={'padding': '0 15px', 'text-align': 'left'}), html.Th("Change %", style={'padding': '0 15px', 'text-align': 'left'}), html.Th("Volume", style={'padding': '0 15px', 'text-align': 'left'})])),
                        html.Tbody([html.Tr([html.Td(info["date"], style={'padding': '0 15px', 'text-align': 'left'}), html.Td(info["close_price"], style={'padding': '0 15px', 'text-align': 'left'}), html.Td(info["change_amount"], style={'padding': '0 15px', 'text-align': 'left'}), html.Td(f"{info['change_percentage']}%", style={'padding': '0 15px', 'text-align': 'left'}), html.Td(info["volume"], style={'padding': '0 15px', 'text-align': 'left'})]) for info in historical_info])
                    ], style={'margin-left': 'auto', 'margin-right': 'auto', 'border-collapse': 'collapse'})
                ], style={'text-align': 'center', 'padding': '15px', 'border-radius': '5px', 'background-color': '#f9f9f9', 'width': '80%', 'margin': 'auto'})
            except KeyError:
                return 'Stock data not available or invalid ticker symbol entered.'

    return 'Enter a ticker and click the button or press Enter.'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
