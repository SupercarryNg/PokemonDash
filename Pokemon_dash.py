import pandas as pd
import numpy as np
import plotly.express as px

from PIL import Image

from dash import Dash, dcc, html
from dash.dependencies import Input, Output

ROOT = 'pokemon_jpg/'
df = pd.read_csv('pokemon.csv')
df_p = df[['pokedex_number', 'name', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense']]
Types = ['against_bug', 'against_dark', 'against_dragon', 'against_electric',
         'against_fairy', 'against_fight', 'against_fire', 'against_flying',
         'against_ghost', 'against_grass', 'against_ground', 'against_ice',
         'against_normal', 'against_poison', 'against_psychic', 'against_rock',
         'against_steel', 'against_water'
         ]
df_t = df[Types]
df_t = pd.concat([df['name'], df_t], axis=1)
# add picture

app = Dash(__name__)

# Design the layout of the dash board
app.layout = html.Div([
    html.H1('Pokemon ability Plot', style={'text-align': 'center'}),
    dcc.Input(
            id="Pokemon_Name",
            placeholder="Input a Pokemon Name",
            value='Bulbasaur',
            type='text'
        ),
    html.Br(),
    html.Div(children=[dcc.Graph(id='Pokemon pic', style={'width': '33%', 'display': 'inline-block'}),
                       dcc.Graph(id='Radar Plot', figure={}, style={'width': '33%', 'display': 'inline-block'}),
                       dcc.Graph(id='Against Type', figure={}, style={'width': '33%', 'display': 'inline-block'})
                       ])
])


@app.callback(
    [Output(component_id='Pokemon pic', component_property='figure'),
     Output(component_id='Radar Plot', component_property='figure'),
     Output(component_id='Against Type', component_property='figure')],
    [Input(component_id='Pokemon_Name', component_property='value')]
)
def update_output(Pokemon):
    dff_p = df_p.copy()

    dff_p = dff_p[dff_p['name'] == Pokemon] # get the target pokemon info

    dff_index = int(dff_p['pokedex_number'].values) # get the index of pokemon info
    img_path = ROOT + str(dff_index) + '.jpg'

    img = np.array(Image.open(img_path))
    fig = px.imshow(img)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    tmp_r = pd.DataFrame(dict(
         r=dff_p.values[0][2:],
         theta=dff_p.columns[2:]))
    fig_r = px.line_polar(tmp_r, r='r', theta='theta', line_close=True)
    fig_r.update_traces(fill='toself')

    dff_t = df_t.copy()
    dff_t = dff_t[dff_t['name'] == Pokemon]
    tmp_t = pd.DataFrame(dict(
         Against=dff_t.values[0][1:],
         Type=dff_t.columns[1:]))
    fig_t = px.bar_polar(tmp_t, r="Against", theta="Type",color="Against")
    # fig_t = px.line_polar(tmp_t, r='r', theta='theta', line_close=True)
    # fig_t.update_traces(fill='toself')

    return fig, fig_r, fig_t


if __name__ == '__main__':
    app.run_server(debug=True)


# Pikachu
# Charizard