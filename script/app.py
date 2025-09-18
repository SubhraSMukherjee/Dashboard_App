#python -m pip install pandas,plotly,dash_bootstrap_components


import plotly.graph_objs as go
import plotly.figure_factory as figure
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from dash.dependencies import Output
from dash.dependencies import Input
from dash import Dash, dcc, html
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from datetime import datetime
from plotly.tools import mpl_to_plotly

series = pd.Series(['76.25', '77.41', '71.84', '79.07', '76.31', '77.02', '76.72','72.6', '69.74', '69.3', '64.33', '73.05', '66.74', '73.28','71.57', '66.23', '66.24', '62.95', '63.24', '69.52', '70.25','65.96', '66.1', '68', '66.62', '66.27', '69.57', '66.28', '67.25','65.58', '65.32', '63.37', '72.69', '78.99', '77.74', '69.11','72.62', '73.59', '72.04', '65.48', '71.45', '71.5', '68.83','66.03', '60.77', '64.79', '64.23', '67.59', '70.92', '66.01','62.45', '60.95', '63.92', '65.19', '64.57', '68.53', '64.33','64.5', '71.73', '68.68', '68.36', '69.99', '78.57', '73.76','69.78', '76.81', '74.39', '76.09', '100', '83.92', '84.83','86.76', '75.99', '71.74', '71.98', '75.82', '76.22', '68.42','68.36', '64.87', '65.95', '66.55', '72.8', '74.27', '70.19','65.48', '66.39', '73.58', '80.06', '76.14', '80.22', '81.99','85.4', '83.73', '87.79', '83.62', '88.31', '83.42', '74.7','75.19', '69.39', '71.66', '68.29', '66.91', '71.54', '71.33',
'66.15', '63.51', '66.52', '71.71', '73.08', '83.44', '82.63','83.7', '85.23', '78.04', '83.43', '86.57', '88.87', '78.74','77.8', '71.85', '75.74', '76.52', '79.71', '79.17', '76.92','75.89', '78.49', '77.12', '73.06', '74.47', '80.24', '76.55','76.88', '77.79', '80.9', '80.33', '81.82', '78.22', '78.26','85.55', '90.09', '85.7', '88.17', '84.99', '85.14', '79.65','79.17', '83.69', '87.19', '91.4', '93.18', '94.28', '80.59','87.61', '88.38', '83.69', '83.95', '91.48', '87.21', '85.29','77.41', '79.08', '76.93', '86.52', '82.4', '80.31', '79.76','78.36', '76.56', '73.63', '77.86', '82.93', '86.29', '78.65','70.91', '68.86', '64.47', '66.02', '70.54', '72.56', '72.44','71.35', '73.33', '72.32', '77.85', '79.92', '81.98', '77.49','77.38', '78.62', '75.24', '77.88', '82.18', '83.48', '81.99','80.74', '77.31', '79.36', '81.81', '87.56', '82.88', '80.74','82.18', '79.88', '81.85', '77.06', '80.95', '78.06', '76.94','75.63', '77.95', '81.01', '79.71', '79.31'])

plt.rcParams['axes.linewidth'] = 0.1
fig_Time_Series, (ax0, ax1, ax2, ax3) = plt.subplots(4, figsize=(17, 10))

series = series.astype('float')
dates = pd.date_range(datetime.strptime('2024-01-01', '%Y-%m-%d').date(), periods=216)

#result = seasonal_decompose(series, model ='additive', period = 30, extrapolate_trend = 0, two_sided = True)
result = STL(series, period=30, seasonal_deg = 12, trend_deg=0, low_pass_deg=0, robust=True).fit()

ax0.plot(dates, series, linewidth=1)
ax0.get_xaxis().set_visible(False)
ax0.set_facecolor('whitesmoke')
ax0.set_ylabel('Raw Time Series')
#ax0.set_title(label=" Time Series Decomposition", loc = 'left', pad = 12.0,
          #fontsize=20,
          #color="black")


ax1.plot(dates, result.trend , linewidth=1.5)
ax1.get_xaxis().set_visible(False)
ax1.set_ylabel('Trend')


ax2.plot(dates, result.seasonal , linewidth=1)
ax2.get_xaxis().set_visible(False)
ax2.set_ylabel('Seasonal')


ax3.plot(dates, result.resid ,  marker="o", linestyle="none", color='grey')
#ax3.get_xaxis().set_visible(False)
ax3.set_xlabel('Year to Date')
ax3.set_ylabel('Residual')


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

#Page 1
page1_content = html.Div([
    html.H3(children='Page 1', style={'color': "LightGray",'padding': "10px", 'fontFamily': "Arial", 'margin-right': '50em'}),
    #Dropdown Div
    html.Div(
            children=[
            html.Div(
                children=[
                    html.Div(children="Country",
                             style = {'display': 'inline-block', 'margin': '5px 5px 5px 5px', 'font-size':'20px'}),


                    html.Div(title=("Select Country of Choice"),
                             children="[?]",
                             style = {'display': 'inline-block', 'border-bottom': '1px dotted black'}),

                    # the actual dropdown menu to select a word to display graphs for
                    dcc.Dropdown([{"label": country, "value": country}
                            for country in ['Canada', 'US', 'Netherlands','Korea', 'Japan']],
                        value = 'Canada',
                        id='dropdown_selection',
                        clearable=True,
                        className="dropdown",
                        style = {'width':'98%','display': 'inline-block', 'margin': '0 5px auto'}
                    ),

                ]
            ),
            ],
            style={"display": 'inline-block', 'width':'25%'},
            className="card",
            ),

    html.Hr(),
    #Tabs
    dcc.Tabs(id="tabs", vertical = False, children=[
        dcc.Tab(label='Tab 1', style = {'margin-right': '0px'} , children=[
            html.Div(
                    children = dcc.Graph(
                    id='Install_Rates',
                    config={"displayModeBar": False},
                    #figure=fig,
                    #style={"width":"50vh"}
                )
              ),
            html.Hr(style = {'width': '100%', 'color':'gray'}),
            html.Div(
                    children = dcc.Graph(
                    id='In_App_Purchases',
                    config={"displayModeBar": False},
                    #figure=fig,
                    #style={"width":"50vh"}
                )
              )
            ]),
        dcc.Tab(label='Tab 2', children=[
            html.Div([
                html.H1("Tab 2"),
                html.P("Graph goes here!")
            ])
        ]),
        dcc.Tab(label='Tab 3', children=[
            html.Div([
                html.H1("Tab 3"),
                html.P("Graph goes here!")
            ])
        ]),
    ],
    style={
        'fontFamily': 'Arial',
        'backgroundColor': "DodgerBlue",
        'padding-left': '0em',
        'margin': '0 auto' ,
        'align-items': 'left',
        'justify-content': 'left',
        'padding':'0px'
    },
        content_style={
        "margin-left": "0rem",
        "margin-right": "0rem",
        "padding": "0rem 0rem",

    },
        parent_style={
        'maxWidth': '2000px',
        'margin': '0 auto',
        'padding-left': '0em',
        'align-items': 'left'
    }

    )
])

#Page 2
page2_content = html.Div(
    [
      html.H2(children='Page 2', style={'color': "LightGray",'padding': "5px", 'font-size': '2em', 'fontFamily': "Arial", 'margin-right': '40em'}),
      html.Hr(),
      dcc.Graph(id='Time_Series_Decomposition', style={'width':'100%',  'padding-left': '0em', 'align-items': 'left', 'align' : "left"}, config={"displayModeBar": False}, figure=mpl_to_plotly(fig_Time_Series))
    ],
    style={'width':'100%', 'justify-content': 'left','margin': '0 auto', 'padding-left': '0em', 'align-items': 'left', 'align' : "left"}
    )

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#dae7f5", # #f8f9fa
}

sidebar = html.Div(
    [
        html.H3("My Opensource Visualization Tool", className="display-4", style={'fontFamily': "Menlo", 'font-size': '3em'}),
        html.Hr(),
        html.P("Pages", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Page 1", href="/", active="exact"),
                dbc.NavLink("Page 2", href= "/page2_url", active="exact"),
                dbc.NavLink("Page 3", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    #'height' : '200vh',   # To Control HTML height
    'overflow-y': 'hidden'
}

content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# Page Link
@app.callback(Output("page-content", "children"),[Input("url", "pathname")])
def render_page(pathname):
    if pathname == "/":
        return page1_content
    elif pathname == "/page2_url":
        return page2_content
    return html.Div(
        [
            html.H1("No Page Found!!", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} does not exist yet..."),
        ],
        className="p-3 bg-light rounded-3",
    )


# Page 1 Tab 1
@app.callback(
    [Output('Install_Rates', 'figure'),
     Output('In_App_Purchases', 'figure')],
    [Input('dropdown_selection', "value")],
     prevent_initial_call=False
)
def update_charts(value):
  if value == 'Canada':
    mean = 1
  else:
    mean = 2

  df = pd.DataFrame(np.column_stack([list(range(101)), list(range(101)), np.random.normal(0.35, 0.01, 101).tolist(),
                    np.random.normal(0.07*mean, 0.05, 101).tolist(),np.random.normal(0.30, 0.02, 101).tolist(),np.random.normal(0.7, 0.03, 101).tolist()]),
                    columns=['days_after_release', 'days_after_release_actual', 'MetricRate', 'MetricRate_5thPercentile','MetricRate_Median', 'MetricRate_95thPercentile'])

  df['days_after_release'] = df['days_after_release'].astype(int)
  df['days_after_release_actual'] = df['days_after_release_actual'].astype(int)
  df['MetricRate'] = df['MetricRate'].astype(float)
  df['MetricRate_5thPercentile'] = df['MetricRate_5thPercentile'].astype(float)
  df['MetricRate_Median'] = df['MetricRate_Median'].astype(float)
  df['MetricRate_95thPercentile'] = df['MetricRate_95thPercentile'].astype(float)

  fig = go.Figure([go.Scatter(name='Average Install Rate', x=df['days_after_release_actual'], y=df['MetricRate'] , mode='lines', line=dict(color = 'rgb(31,119,180)',width=3)),go.Scatter(name='Install Rate Median', x=df['days_after_release_actual'], y=df['MetricRate_Median'] , mode='lines', line=dict(color = 'rgb(250,128,114)')) ,go.Scatter(name = '95th Percentile Install Rate', x=df['days_after_release_actual'], y=df['MetricRate_95thPercentile'], mode='lines', marker=dict(color='#444'), line=dict(width=1),showlegend=True), go.Scatter(name = '5th Percentile Install Rate', x=df['days_after_release_actual'], y=df['MetricRate_5thPercentile'] , mode='lines', fillcolor='rgba(68,68,68,0.05)', fill='tonexty', showlegend=True)])

  fig.update_layout(title="Install Conversion Rates for App Soft Launches", yaxis = dict(title = dict(text = ' Install Conversion Rates')),  hovermode = 'x', xaxis = dict(title = dict(text = 'Days After Release'),dtick = 2), yaxis_range=[0,0.8], plot_bgcolor = 'rgba(0, 0, 0, 0)', legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
  ), width=2000, height=500)


  x1 = np.random.chisquare(5*mean, size=10000)
  x2 = np.random.gamma(10*mean,1.5, size=10000)
  x3 = np.random.gamma(25/mean,0.3, size=10000)
  x4 = np.random.chisquare(30/mean, size=10000)

  list_data = [x1, x2, x3, x4]

  group_labels = ['18-25', '35-60', '60+', '25-35']

  # Create distplot with custom bin_size
  fig_1 = figure.create_distplot(list_data, group_labels, bin_size=.2)
  fig_1.update_layout(title="In App Purchases by Demography", yaxis = dict(title = dict(text = 'No of Purchases (Relative)')),
                      hovermode = 'x', xaxis = dict(title = dict(text = 'Purchase Size'),dtick = 1), legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
  ), width=2000, height=600)
  #fig.show()

  return fig,fig_1


if __name__ == '__main__':
    app.run(debug=False)
