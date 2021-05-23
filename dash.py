import dash
import matplotlib as plt
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

df = pd.read_csv("./Data/GOOG.csv")

#---------------------------------------------------------


app.layout = html.Div([

    html.H1("Stock Prediction Interactive Dashboard", style={'text-align': 'center'}),

    dcc.Dropdown(id="Stock_Choose",
                    options=[
                        {"label": "Text 1", "Value": 1},
                        {"label": "Text 2", "Value": 2},
                        {"label": "Text 3", "Value": 3},
                        {"label": "Text 4", "Value": 4},
                        {"label": "Text 5", "Value": 5}],
                    multi=False,
                    value=1,
                    style={'width': "40%"}
                    ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my_stock_data', figure={})


])
