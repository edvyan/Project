import dash
from dash import html, dcc, Input, Output, State, ClientsideFunction, callback_context
import dash_bootstrap_components as dbc
from flask import Flask

# Import generate_response from chatbot
from bot import generate_response

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# initial message
initial_bot_greeting = "Hello! I'm your financial advisor. How can I assist you today?\n"

app.layout = dbc.Container([
    dbc.Row(html.H2("Financial Advisor Chatbot", className="text-center mb-4")), 

    dbc.Row(
        dcc.Textarea(
            id='chat-area',
            value=initial_bot_greeting,
            style={'width': '100%', 'height': '800px', 'overflowY': 'auto'}, 
            readOnly=True
        ),
        className="mb-3" 
    ),

    dbc.Row(
        [
            dcc.Input(
                id='user-input',
                type='text',
                placeholder='Type your message here...',
                style={'width': '90%', 'height': '50px'}, 
                n_submit=0,
                value=''
            ),
            html.Button('Send', id='send-button', n_clicks=0, style={'height': '50px', 'width': '10%'}), 
        ],
        className="mb-3" 
    ),

    html.Div(id='company-context', style={'display': 'none'}),
    html.Div(id='ticker-context', style={'display': 'none'}),
    html.Div(id='dummy-div', style={'display': 'none'}) 
], fluid=True)


# handle sending messages
@app.callback(
    [Output('chat-area', 'value'), Output('user-input', 'value')],
    [Input('send-button', 'n_clicks'), Input('user-input', 'n_submit')],
    [State('user-input', 'value'), State('chat-area', 'value')]
)
def update_chat(send_clicks, enter_presses, input_text, chat_value):
    ctx = callback_context
    if not ctx.triggered:
        return chat_value, input_text  
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if (button_id == 'send-button' or button_id == 'user-input') and input_text:
        response = generate_response(input_text)
        new_chat = f"{chat_value or ''}You: {input_text}\nBot: {response}\n\n"
        return new_chat, ''  
    return chat_value, input_text  

# Clientside callback to handle auto-scrolling (if you prefer directly embedding JavaScript)
app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='scrollToBottom'
    ),
    Output('dummy-div', 'children'), 
    Input('chat-area', 'value')  # Triggered whenever the chat-area text changes
)

if __name__ == '__main__':
    app.run_server(debug=True)
