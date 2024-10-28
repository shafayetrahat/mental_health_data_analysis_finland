from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import dash_bootstrap_components as dbc

data = pd.read_csv('/home/shafayetrahat/intro_data_course/mini_project/web_app/important_features.csv')

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions = True)
# Load the HTML file content
html_file_path = 'assests/finland_map_total.html'

# Read the HTML file's contents
with open(html_file_path, "r", encoding="utf-8") as file:
    html_content = file.read()

app.layout = dbc.Container([
    dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Source Code", href="https://github.com/shafayetrahat/mental_health_data_analysis_finland")),
    ],
    brand="Mental Health Analysis",
    brand_href="#",
    color="primary",
    dark=True,
)
    ,
    dbc.Container([
                   html.Br(),
                   html.Br(),
                   html.Br(),
                   dbc.Row([
                   dbc.Col(html.Div(""), width=3),
                   dbc.Col(html.Div("Mental health problems have a profound effect on an individual's life. Our solution can help people from administration to decide based on mental health."),
                           width=6,
                           style={"fontSize": "20px"},
                           ),
                   dbc.Col(html.Div(""), width=3),
                   ]),
                   html.Br(),
                   html.Br(),
                   html.Br(),
                   html.Br()
                   ]),
    dbc.Container([
    dbc.Row([
        dbc.Col(html.Div(""), width=1),
        dbc.Col(html.Iframe(srcDoc=html_content,
                            style={"width": "100%", "height": "500px", "border": "none"}
                            ),
                width=10),
        dbc.Col(html.Div(""), width=1),
    ]
        ),
     html.Br(),
                   html.Br(),
                   html.Br(),
     dbc.Row([
         dbc.Col(html.Div(""), width=4),
        dbc.Col(html.Div("Fig: Overall Mental health indicator all over finland. Municipality-based map"), width=7),
        dbc.Col(html.Div(""), width=1),
    ]
        ),
        html.Br(),
        html.Br(),
    dbc.Row([dbc.Col(html.H4("Mental Health Analysis For a single Municipality.", className="text-center mb-4"), width=12)]),
    dbc.Row([
        dbc.Col(html.Div(""), width=4),
        dbc.Col(
            dcc.Dropdown(
                data.municipality.unique(),
                'Akaa',
                id='dropdown-selection',
                className="text-center mb-4"
            ),
            width=4
        ),
        dbc.Col(html.Div(""), width=4),
    ]),
     dbc.Row([
        dbc.Col(dcc.Graph(id='graph4'), width=12),
    ]),
      dbc.Row([
        dbc.Col(html.Br(), width=12),
    ]),
     dbc.Row([
        dbc.Col(dcc.Graph(id='graph1'), width=4),
        dbc.Col(dcc.Graph(id='graph2'), width=4),
        dbc.Col(dcc.Graph(id='graph3'), width=4),
    ]),
    dbc.Row([
        dbc.Col(html.Div(""), width=2),
        dbc.Col(dcc.Graph(id='graph0'), width=8),
        dbc.Col(html.Div(""), width=2),
    ])
    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    dbc.Row([
         dbc.Col(html.Div(""), width=4),
        dbc.Col(html.Div("Fig: Mental health index Ranking (Municipality-based). It will show you which parameters are more important than the others influencing mental health indicator."), width=7),
        dbc.Col(html.Div(""), width=1),
    ]
        ),
], fluid=True)  # fluid=True makes it full-width

@callback(
    [
        Output('graph4', 'figure'),
        Output('graph1', 'figure'),
        Output('graph2', 'figure'),
        Output('graph3', 'figure'),
        Output('graph0', 'figure')
        ],
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    municipality_data = data[data['municipality'] == value]
    y_values = data[data['year']==2022]
    y_values = y_values[['year', 'municipality', 'depression', 'involuntary_care', 'psychotic']]
    df = y_values[['depression', 'involuntary_care', 'psychotic']]
    y_values[['depression', 'involuntary_care', 'psychotic']] = (df-df.min())/(df.max()-df.min())
    y_targets = [municipality_data['depression'], municipality_data['involuntary_care'], municipality_data['psychotic']]
    municipality_data = municipality_data.drop(['year', 'municipality', 'depression', 'involuntary_care', 'psychotic'], axis=1)
    X = municipality_data
    importance_dict = {}
    targetname=['depression_index','involuntary_care_index','psychotic_index']
    for i, target in enumerate(y_targets):
        model = RandomForestRegressor()
        model.fit(X, target)
        importance_dict[f'{targetname[i]}'] = model.feature_importances_

    importance_df = pd.DataFrame(importance_dict, index=X.columns)
    importance_df['Average_MH_index'] = importance_df.mean(axis=1)
    targetname = importance_df.columns
    figs = []
    val =y_values[y_values['municipality']==value]
    df_long = val.iloc[:,2:].melt(var_name="Mental_Health_Factor", value_name="Value")
    fig = px.bar(
            df_long,
            x='Mental_Health_Factor',
            y='Value',
            orientation='v',
            title=f'MH Index values for {value}',
            labels={'y': 'Features', 'x': 'target'}
        )
    fig.update_layout(template="simple_white", margin=dict(l=20, r=20, t=40, b=20))
    figs.append(fig)
    for target in importance_df:
        importance_df = importance_df.sort_values(by=target, ascending=True)
        fig = px.bar(
            importance_df,
            y=importance_df.index,
            x=target,
            orientation='h',
            title=f'Feature Importance for municipality: {value}',
            labels={'y': 'Features', 'x': 'target'}
        )
        fig.update_layout(template="simple_white", margin=dict(l=20, r=20, t=40, b=20))
        figs.append(fig)

    return figs

if __name__ == "__main__":
    app.run_server(debug=True)
