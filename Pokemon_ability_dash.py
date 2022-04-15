import os
from PIL import Image

import umap

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# basic information
ROOT = 'pokemon_jpg/'
df = pd.read_csv('pokemon.csv')
# config for plots
configs = {
    'staticPlot': False,
    'scrollZoom': True,
    'doubleClick': 'reset'
}

df_p = df[['hp', 'attack', 'speed', 'sp_attack', 'sp_defense', 'defense']]
reducer = umap.UMAP(min_dist=0.9, random_state=42, n_components=2, n_neighbors=200)
# reducer = sklearn.manifold.MDS()
embedding = reducer.fit_transform(df_p)
df_embedding = pd.DataFrame(embedding)
df_embedding = df_embedding.reset_index()
df_embedding = pd.concat([df_embedding, df['type1']], axis=1)
df_embedding.columns = ['idx', 'embedding_0', 'embedding_1', 'Primary Type']

df_gb = df[['hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense']].groupby(df['type1']).mean()

# Analysis on target types for warrior, tank, assassin
dff = df.copy()
df_gb_class = dff.pivot_table(index=['type1'],
                              values=['name', 'capture_rate'],
                              aggfunc={'name': len, 'capture_rate': np.mean})
df_gb_class = df_gb_class.reset_index()

recommend_group = ['dragon', 'ground', 'steel', 'ghost', 'flying', 'electric', 'psychic']
recommend = []
for type_ in df_gb_class['type1']:
    if type_ in recommend_group:
        recommend.append('Recommend Group')
    else:
        recommend.append('Not Recommend Group')

df_gb_class['recommend'] = recommend
df_gb_class.columns = ['Primary Type', 'Capture_rate', 'Count', 'Recommend']


def get_top5(attribute):  # Get the Top5 types for the select attribute(hp, attack,...)
    attr = df_gb[attribute]
    attr = attr.sort_values(ascending=False)
    return attr


def get_bins(ability) -> list:  # input attack, defense etc.
    tmp_df = pd.read_csv('pokemon.csv')
    tmp_ab = tmp_df[[ability]].describe()  # only concern about the target column
    bins = [tmp_ab.at['min', ability], tmp_ab.at['25%', ability], tmp_ab.at['50%', ability],
            tmp_ab.at['75%', ability], tmp_ab.at['max', ability]]
    return bins


def get_labels(ability, bins) -> list:  # input the ability and bins divided by get_bins
    tmp_df = pd.read_csv('pokemon.csv').copy()
    lbs = pd.cut(tmp_df[ability], bins=bins, labels=False)
    return lbs


def img_open_trans(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = np.array(img)
    return img


def show_img(img):  # show images in dash, input a np array
    fig = px.imshow(img)
    fig.update_layout(coloraxis_showscale=False, hovermode=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


app = Dash(__name__)

app.layout = html.Div([
    html.H1('Pokemon Ability & Types Analysis', style={'text-align': 'center', 'font-family': 'Book Antiqua'}),
    html.Div(
        children=[dcc.Graph(id='ability_1', style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'}),
                  dcc.Graph(id='Top5', style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'})
                  ]),
    dcc.RadioItems(
        id='RadioItems',
        options=[
            {'label': 'Sp Defense', 'value': 'sp_defense'},
            {'label': 'Defense', 'value': 'defense'},
            {'label': 'Hp', 'value': 'hp'},
            {'label': 'Attack', 'value': 'attack'},
            {'label': 'Speed', 'value': 'speed'},
            {'label': 'Sp Attack', 'value': 'sp_attack'}
        ],
        value='sp_defense',
        inline=True,
        style={'text-align': 'center', 'font-family': 'Book Antiqua'}
    ),
    html.Br(),
    html.Hr(),
    html.Div(children=[dcc.Graph(id='mds_res1', style={'width': '50%', 'display': 'inline-block'}, config=configs),
                       dcc.Graph(id='mds_res2', style={'width': '50%', 'display': 'inline-block'}, config=configs)
                       ]),
    html.Div(children=[dcc.Graph(id='mds_res3', style={'width': '50%', 'display': 'inline-block'}, config=configs),
                       dcc.Graph(id='cor_matrix', style={'width': '50%', 'display': 'inline-block'})
                       ]),
    html.Hr(),
    html.Div(children=[dcc.Graph(id='class_table', style={'width': '50%', 'display': 'inline-block'}),
                       dcc.Graph(id='treemap', style={'width': '50%', 'display': 'inline-block'})
                       ]),
    # dcc.Markdown('''
    #                                 We classify Pokemon as three class as below:
    #                                 + high Hp, Attack Pokemon as Warrior.
    #                                 + high Defense, Sp_defense Pokemon as Tank.
    #                                 + high Speed, Sp_attack Pokemon as Assassin.
    #                                 Then we pick Pokemon types that ranks top5 in both attributes
    #                                 (For example Ground rank Top5 in Hp & Attack).
    #                                 ''',
    #              style={'width': '40%', 'display': 'inline-block'}
    #              )
    html.Div(children=[dcc.Dropdown(id='Type',
                                    value='dragon',
                                    options=[
                                             {'label': 'Warrior: Dragon', 'value': 'dragon'},
                                             {'label': 'Warrior: Ground', 'value': 'ground'},
                                             {'label': 'Tank: Dragon', 'value': 'dragon'},
                                             {'label': 'Tank: Steel', 'value': 'steel'},
                                             {'label': 'Tank: Ghost', 'value': 'ghost'},
                                             {'label': 'Assassin: Electric', 'value': 'electric'},
                                             {'label': 'Assassin: Flying', 'value': 'flying'},
                                             {'label': 'Assassin: Psychic', 'value': 'psychic'},
                                             {'label': 'Assassin: Dragon', 'value': 'dragon'}
                                            ],
                                    style={'width': '40%', 'display': 'inline-block',
                                           'font-family': 'Book Antiqua'})]),
    html.Div(
        children=[
            dcc.Graph(id='scatter_plot', style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'}),
            dcc.Graph(id='Pokemon_pic', style={'width': '40%', 'display': 'inline-block', 'text-align': 'center'})
        ]),
    html.Br(),
    html.Div(
        children=[
            dcc.Graph(id='Radar_plot', style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'}),
            dcc.Graph(id='Capture_rate', style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'})
        ])
])


@app.callback(
    [Output(component_id='ability_1', component_property='figure'),
     Output(component_id='Top5', component_property='figure'),
     Output(component_id='mds_res1', component_property='figure'),
     Output(component_id='mds_res2', component_property='figure'),
     Output(component_id='mds_res3', component_property='figure'),
     Output(component_id='cor_matrix', component_property='figure'),
     Output(component_id='class_table', component_property='figure'),
     Output(component_id='treemap', component_property='figure')],
    [Input(component_id='RadioItems', component_property='value')]
)
def update_output(ability):
    # Umap decrease dimension
    bins = get_bins(ability)
    lbs = get_labels(ability, bins)
    fig = go.Figure()  # define a graph
    for i in range(len(np.unique(lbs))):
        fig.add_trace(
            go.Scatter(x=embedding[:, 0][lbs == i], y=embedding[:, 1][lbs == i],
                       name='{}-{}%'.format(i*25, (i+1)*25), mode='markers', marker=dict(size=10))
        )
    fig.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.9,
                      title_text="<b>UMAP Multidimensional Scaling</b>")
    fig.update_layout(title_font_family="Book Antiqua")

    # print bar chart
    df_top5 = get_top5(ability)
    color_top5 = ['orange'] * 5
    color_rest = ['lightblue'] * 13
    color_top5.extend(color_rest)
    fig_top = px.bar(df_top5)
    fig_top.update_traces(marker_color=color_top5)
    fig_top.update_layout(
        xaxis_title="Pokemon Primary Type",
        yaxis_title="{} Points".format(ability.capitalize()),
        legend_title="Top5 Types",
        font=dict(
            family="Book Antiqua",
        )
    )
    fig_top.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.95,
                          title_text="<b>{} Top 5 Types</b>".format(ability.capitalize()))
    fig_top.update_layout(
        title_font_family="Book Antiqua",
    )


    # print the mds results got from R
    mds_results = os.listdir('mds_results')

    img_path = 'mds_results/' + mds_results[0]
    img = img_open_trans(img_path)
    fig_img1 = show_img(img)
    fig_img1.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.95,
                      title_text='<b>Defense & Sp Defense Trend on Mds Plot</b>')
    fig_img1.update_layout(
        title_font_family="Book Antiqua",
    )


    img_path = 'mds_results/' + mds_results[1]
    img = img_open_trans(img_path)
    fig_img2 = show_img(img)
    fig_img2.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.95,
                      title_text='<b>HP & Attack Trend on Mds Plot</b>')
    fig_img2.update_layout(
        title_font_family="Book Antiqua",
    )

    img_path = 'mds_results/' + mds_results[2]
    img = img_open_trans(img_path)
    fig_img3 = show_img(img)
    fig_img3.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.95,
                      title_text='<b>Speed & Sp Attack Trend on Mds Plot</b>')
    fig_img3.update_layout(
        title_font_family="Book Antiqua",
    )

    # plot the correlation matrix
    cor = df_p.corr()
    cor_matrix_fig = px.imshow(cor, text_auto=True)
    cor_matrix_fig.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.95,
                      title_text='<b>Correlation Matrix Between Abilities</b>')
    cor_matrix_fig.update_layout(
        title_font_family="Book Antiqua",
    )
    # plot class table: Warrior, Tank, Assassin
    fig_table = go.Figure(data=[go.Table(header=dict(values=['<b>Warrior</b>(High Hp&Attack)',
                                                             '<b>Tank</b>(High Defense&Sp_Defense)',
                                                             '<b>Assassin</b>(High Speed&Sp_Attack)']),
                                         cells=dict(values=[['<b>Dragon</b>', 'Ground', '', ''],
                                                            ['Steel', '<b>Dragon</b>', 'Ghost', ''],
                                                            ['Flying', 'Electric', 'Psychic', '<b>Dragon</b>']]))
                                ])
    fig_table.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.95,
                      title_text='<b>Recommend Groups & Types</b>')
    fig_table.update_layout(
        title_font_family="Book Antiqua",
    )

    # plot treemap
    fig_treemap = px.treemap(df_gb_class, path=[px.Constant("All Types"), 'Recommend', 'Primary Type'],
                             values='Count',
                             color='Capture_rate',
                             color_continuous_scale='blues')
    fig_treemap.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.95,
                      title_text='<b>Pokemon Types Treemap</b>')
    fig_treemap.update_layout(
        title_font_family="Book Antiqua",
    )

    return fig, fig_top, fig_img1, fig_img2, fig_img3, cor_matrix_fig, fig_table, fig_treemap


@app.callback(
    Output(component_id='scatter_plot', component_property='figure'),
    [Input(component_id='Type', component_property='value')]
)
def update_output(type_):
    df_embedding['lbs'] = np.where(df_embedding['Primary Type'] == type_, 1, 0).astype(str)
    fig = px.scatter(df_embedding, x='embedding_0', y='embedding_1', color='lbs', custom_data=['idx'],
                     color_discrete_map={'0': 'lightblue', '1': 'orange'})
    fig.update_layout(showlegend=False)
    fig.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.95,
                      title_text="<b>{} Pokemon Distribution</b>".format(type_.capitalize()))
    fig.update_layout(
        title_font_family="Book Antiqua",
    )

    return fig


@app.callback(
    [Output(component_id='Pokemon_pic', component_property='figure'),
     Output(component_id='Radar_plot', component_property='figure'),
     Output(component_id='Capture_rate', component_property='figure')],
    [Input(component_id='scatter_plot', component_property='clickData')]
)
def update_side_graph(idx):
    if idx is None:
        poke_idx = 24
    else:
        poke_idx = idx['points'][0]['customdata'][0]
        print(idx)
    img_path = ROOT + str(poke_idx + 1) + '.jpg'
    img = np.array(Image.open(img_path))
    poke_name = df.iloc[[poke_idx]]['name'].values[0]
    poke_type = df.iloc[[poke_idx]]['type1'].values[0]
    fig = px.imshow(img)
    fig.update_layout(coloraxis_showscale=False, hovermode=False)
    fig.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.1,
                      title_text="<b>{}, Type: {}</b>".format(poke_name, poke_type.capitalize()))
    fig.update_layout(
        title_font_family="Book Antiqua",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Plot the radar plot
    tmp = df_p.iloc[[poke_idx]]
    tmp_r = pd.DataFrame(dict(
        r=tmp.values[0],
        theta=tmp.columns))
    fig_radar = px.line_polar(tmp_r, r='r', theta='theta', line_close=True)
    fig_radar.update_traces(fill='toself')
    fig_radar.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.95,
                      title_text='<b>{} Ability Radar Chart</b>'.format(poke_name))
    fig_radar.update_layout(
        title_font_family="Book Antiqua",
    )
    # Plot the capture rate
    df_capture_rate = df['capture_rate']
    capture_rate = df_capture_rate.iloc[[poke_idx]].values[0]
    fig_cp = px.violin(df_capture_rate)
    fig_cp.add_trace(go.Scatter(x=['capture_rate', 'capture_rate'], y=[capture_rate, capture_rate],
                                mode="markers+text", name="", showlegend=False))
    fig_cp.update_layout(coloraxis_showscale=False, title_x=0.5, title_y=0.95,
                      title_text='<b>{} Capture Rate Violin Plot</b>'.format(poke_name))
    fig_cp.update_layout(
        title_font_family="Book Antiqua",
    )
    return fig, fig_radar, fig_cp


if __name__ == '__main__':
    app.run_server(debug=True, port=8052)

#  Things to update
# 1. legend of the scatter plot >>> Done
# 2. titles >>> Done
# 3. divide lines between different parts >>> Done
# 4. Highlight in the table >>> Done
# 5. Basic information for the Pokemon image >>> Done
# 6. Type Radar plot, also show the base total below the radar plot
# 7. Use Battle data to show the winning rates
# 8. Determine the Pokemon that will be showed
