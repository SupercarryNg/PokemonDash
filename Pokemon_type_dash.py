import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px

from PIL import Image

from dash import Dash, dcc, html
from dash.dependencies import Input, Output

df = pd.read_csv('pokemon.csv')


app = Dash(__name__)

app.layout = html.Div([
    html.H1('Pokemon Type Plot', style={'text-align': 'center'}),
    dcc.Dropdown(id='Type',
                 value='bug',
                 options=[
                     {'label': 'bug', 'value': 'bug'},
                     {'label': 'dark', 'value': 'dark'},
                     {'label': 'dragon', 'value': 'dragon'},
                     {'label': 'electric', 'value': 'electric'},
                     {'label': 'fairy', 'value': 'fairy'},
                     {'label': 'fight', 'value': 'fight'},
                     {'label': 'fire', 'value': 'fire'},
                     {'label': 'flying', 'value': 'flying'},
                     {'label': 'ghost', 'value': 'ghost'},
                     {'label': 'grass', 'value': 'grass'},
                     {'label': 'ground', 'value': 'ground'},
                     {'label': 'ice', 'value': 'ice'},
                     {'label': 'normal', 'value': 'normal'},
                     {'label': 'poison', 'value': 'poison'},
                     {'label': 'psychic', 'value': 'psychic'},
                     {'label': 'rock', 'value': 'rock'},
                     {'label': 'steel', 'value': 'steel'},
                     {'label': 'water', 'value': 'water'},
                 ]),
    html.Br(),
    dcc.Graph(id='Pokemon pic', style={'width': '50%'})
])


@app.callback(
    Output(component_id='Pokemon pic', component_property='figure'),
    [Input(component_id='Type', component_property='value')]
)
def update_output(Type):
    dff_p = df.copy()

    dff_p = dff_p[dff_p['type1'] == Type] # get the target pokemon info

    df_p = dff_p[['attack', 'defense', 'speed', 'sp_attack', 'sp_defense']].copy(deep=True)
    df_p['total_ability'] = df_p.apply(lambda x: x.sum(), axis=1)
    df_cp = dff_p[['capture_rate']].copy(deep=True)
    df_cp_ability = pd.concat([df_cp, df_p['total_ability']], axis=1)
    fig = px.density_heatmap(df_cp_ability, x="capture_rate", y="total_ability", nbinsx=30, nbinsy=30)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)