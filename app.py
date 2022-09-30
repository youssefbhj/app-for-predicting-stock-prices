import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import pmdarima as pm
from pmdarima.arima import auto_arima
import dash
from fbprophet.plot import plot_plotly, plot_components_plotly
from pmdarima.arima import auto_arima
from fbprophet import Prophet
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, dash_table

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math
from statsmodels.graphics.gofplots import qqplot



########################""
def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    if result[1] <= 0.05:
        return (
            "Augmented Dickey-Fuller Test:\n strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        return (
            "Augmented Dickey-Fuller Test:\n weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


###########################################################"

def create_corr_plot(series, plot_pacf=False):
    corr_array = pacf(series.dropna(), nlags=40, alpha=0.05) if plot_pacf else acf(series.dropna(), nlags=40,
                                                                                   alpha=0.05)
    lower_y = corr_array[1][:, 0] - corr_array[0]
    upper_y = corr_array[1][:, 1] - corr_array[0]

    fig = go.Figure()
    [fig.add_scatter(x=(x, x), y=(0, corr_array[0][x]), mode='lines', line_color='#3f3f3f')
     for x in range(len(corr_array[0]))]
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                    marker_size=12)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines', fillcolor='rgba(32, 146, 230,0.3)',
                    fill='tonexty', line_color='rgba(255,255,255,0)')
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1, 42])
    fig.update_yaxes(zerolinecolor='#000000')

    title = 'Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig.update_layout(title=title)
    return fig


#########################""""



app = Dash(__name__, suppress_callback_exceptions=True)





tabs_styles = {
    'height': '40px'
}
tab_style = {
    'align': 'center',
    'fontWeight': 'bold'
}
tab_selected_style = {
    'align': 'center',
    'fontWeight': 'bold'
}

app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Div(dcc.Input(id='inputonsubmit', type='text', placeholder="ticker")),
                        width=2),
                dbc.Col(html.Div(dcc.DatePickerRange(
                    id='my-date-picker-range',
                    min_date_allowed=date(1995, 8, 5),
                    max_date_allowed=date(2022, 6, 16),
                    initial_visible_month=date(2022, 6, 16),
                    end_date=date(2022, 6, 16),
                    display_format='D-M-Y'
                )), width=3),
                dbc.Col(
                    dbc.Button(
                        "SUBMIT", id="submit-button", n_clicks=0
                    ), width=1, className="d-grid gap-2"),
                dbc.Col(
                    dcc.Tabs(id="tabs-example-graph", value='', style=tabs_styles,
                             children=[
                                 dcc.Tab(label='tables', value='tab-1-example-graph', style=tab_style,
                                         selected_style=tab_selected_style),
                                 dcc.Tab(label='charts', value='tab-2-example-graph', style=tab_style,
                                         selected_style=tab_selected_style),
                                 dcc.Tab(label='decomposition', value='tab-3-example-graph', style=tab_style,
                                         selected_style=tab_selected_style),
                                 dcc.Tab(label='modelling', value='tab-4-example-graph', style=tab_style,
                                         selected_style=tab_selected_style),
                                 dcc.Tab(label='accuracy/resid', value='tab-5-example-graph', style=tab_style,
                                         selected_style=tab_selected_style),
                                 dcc.Tab(label='forecast', value='tab-6-example-graph', style=tab_style,
                                         selected_style=tab_selected_style)
                             ]), width=6)

            ]),

        ])
    ),
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Dropdown(['ARIMA', 'auto-ARIMA', 'SARIMA', 'Prophet'], 'model', id='demodropdown'),
                        html.Div(id='dd-output-container')
                    ])

                ], width=2),
                dbc.Col([
                    html.Div([
                        dcc.Slider(min=0, max=100, step=5, value=80, id='myslider'),
                    ])

                ], width=7),
                dbc.Col(
                    dbc.Button(
                        "TRAIN", id="TRAINY", n_clicks=0
                    ), width=1, className="d-grid gap-2"),

                dbc.Col(
                    dcc.Input(id='forecast_steps', type='text', placeholder="steps"), width=1, className="d-grid gap-2"),

                dbc.Col(
                    dbc.Button(
                        "Forecast", id="Forecast_button", n_clicks=0
                    ), width=1, className="d-grid gap-2")

            ])
        ])
    ), html.Div(id='dfoutput', style={'display': 'none'}),
    html.Div(id='ARIMA'),
    html.Div(id='SARIMA'),

    html.Div(id='decompparameters'),
    html.Div(id='plot_viz'),
    html.Div(id='drop')
    ,
    html.Div(id='daaaamn'),
    html.Div(id='tabs'),
    html.Div(id='tabstable'),

    html.Div(id='modelllllll'),
    html.Div(id='auto-ARIMA_train'),
    html.Div(id='ARIMA_train'),
html.Div(id='SARIMA_train'),


    html.Div(id='SARIMA_viz'),
    html.Div(id='ARIMA_viz'),

    html.Div(id='PACF_ACF_plot')

])


###########################
# TICKER SLECT
@app.callback(
    Output('dfoutput', 'children'),
    Input('submit-button', 'n_clicks'),
    State('inputonsubmit', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),

)
def update_output(n_clicks, inputonsubmit_value, start_date, end_date):
    if n_clicks > 0:
        global df, dg
        df = yf.download(str(inputonsubmit_value), start=pd.to_datetime(start_date), end=pd.to_datetime(end_date),
                         adjusted=True, progress=False)
        dg = yf.download(str(inputonsubmit_value), start=pd.to_datetime(start_date), end=pd.to_datetime(end_date),
                         adjusted=True, progress=False)
        df = df.asfreq('b')
        df = df.fillna(method='ffill')
        dg = dg.asfreq('b')
        dg = dg.fillna(method='ffill')
        dg = dg.reset_index()
        dg = dg[['Date', 'Close']]
        dg.columns = ['ds', 'y']
        js = df.to_json(orient='records')
        return js


###################################3333
@app.callback(
    Output('dd-output-container', 'children'),
    Input('demodropdown', 'value'),
    Input('myslider', 'value')
)
def dropdown(demodropdown_value, myslider_value):
    return f'You have selected {demodropdown_value} , train {myslider_value}% of the data set '


#########################""

@app.callback(
    Output('ARIMA', 'children'),
    Input('demodropdown', 'value'),

)
def model_output(value):
    if value == 'ARIMA':
        return dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Div(dcc.Input(id='porder', type='text', placeholder="P")),
                            width={'size': 1, 'offset': 0}),
                    dbc.Col(html.Div(dcc.Input(id='iorder', type='text', placeholder="I")),
                            width={'size': 1, 'offset': 1}),
                    dbc.Col(html.Div(dcc.Input(id='qorder', type='text', placeholder="Q")),
                            width={'size': 1, 'offset': 1})
                ])

            ]))


#########################""
@app.callback(
    Output('SARIMA', 'children'),
    Input('demodropdown', 'value'),

)
def smodel_output(value):
    if value == 'SARIMA':
        return dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Div(dcc.Input(id='sporder', type='text', placeholder="p")),
                            width={'size': 1, 'offset': 0}),
                    dbc.Col(html.Div(dcc.Input(id='siorder', type='text', placeholder="i")),
                            width={'size': 1, 'offset': 1}),
                    dbc.Col(html.Div(dcc.Input(id='sqorder', type='text', placeholder="q")),
                            width={'size': 1, 'offset': 1}),
                ],    justify="center", align="center"),
                dbc.Row([html.P("seasonal")], align="center"),
                dbc.Row([
                    dbc.Col(html.Div(dcc.Input(id='bsporder', type='text', placeholder="Ps")),
                            width={'size': 1, 'offset': 0}),
                    dbc.Col(html.Div(dcc.Input(id='bsiorder', type='text', placeholder="Is")),
                            width={'size': 1, 'offset': 1}),
                    dbc.Col(html.Div(dcc.Input(id='bsqorder', type='text', placeholder="Qs")),
                            width={'size': 1, 'offset': 1}),
                    dbc.Col(html.Div(dcc.Input(id='morder', type='text', placeholder="M")),
                            width={'size': 1, 'offset': 1})
                ], justify="center", align="center")

            ]))


#####################
@app.callback(Output('tabs', 'children'),
              Input('tabs-example-graph', 'value'),
              Input('submit-button', 'n_clicks'))
def tab_para(tab, n_clicks):
    if tab == 'tab-1-example-graph' and n_clicks > 0:
        return dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.RangeSlider(1, len(df), id='tabslider', step=1, marks=None, value=[0, len(df)],
                                            tooltip={"placement": "bottom", "always_visible": True}, persistence=True),
                            width=6)
                    , dbc.Col(dbc.Button('show', id='tabshow', n_clicks=0), width=2, className="d-grid gap-2")
                ],   justify="center", align="center")]))


@app.callback(Output('tabstable', 'children'),
              Input('tabs-example-graph', 'value'),
              Input('tabslider', 'value'),
              Input('tabshow', 'n_clicks'))
def tabshow(tab, value, n_clicks):
    if tab == 'tab-1-example-graph' and n_clicks > 0:
        return dbc.Container([
            dbc.Label('Click a cell in the table:'),
            dash_table.DataTable(df[value[0] - 1:value[1] - 1].reset_index().to_dict('records'),
                                 [{"name": i, "id": i} for i in df.reset_index().columns], id='tbl'
                                 ,
                                 style_data={
                                     'color': 'black',
                                     'backgroundColor': 'white',
                                     'textAlign': 'center',
                                     'font_size': '12px',
                                     'fontWeight': '500',
                                     'border': '1px solid white'
                                 },
                                 style_data_conditional=[
                                     {
                                         'if': {'row_index': 'odd'},
                                         'backgroundColor': 'rgb(171, 226, 251)'
                                         ,
                                         'font_size': '12px',
                                         'fontWeight': '500',
                                         'textAlign': 'center',

                                     }
                                 ],
                                 style_header={
                                     'backgroundColor': 'rgb(13, 110, 253)',
                                     'color': 'white',
                                     'font_size': '15px',
                                     'fontWeight': 'bold',
                                     'textAlign': 'center',
                                     'border': '1px solid white'
                                 }

                                 ), ])


#####################
@app.callback(Output('plot_viz', 'children'),
              Input('tabs-example-graph', 'value'),
              Input('submit-button', 'n_clicks'))
def decomp_para(tab, n_clicks):
    if tab == 'tab-2-example-graph' and n_clicks > 0:
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False)
        return dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Div([dcc.Graph(id='timeseries',
                                                config={'displayModeBar': False},
                                                animate=True,
                                                figure=px.line(df[['Close', 'Open', 'High', 'Low', 'Adj Close']]
                                                               )
                                                ),
                                      dcc.Graph(id='lopooo',
                                                config={'displayModeBar': False},
                                                animate=True,
                                                figure=px.bar(df['Volume'])),
                                      dcc.Graph(id='candlestick',
                                                config={'displayModeBar': False},
                                                animate=True,
                                                figure=fig)
                                      ]), width=8)

                ],    justify="center", align="center")

            ])
        )


#########################""
@app.callback(Output('decompparameters', 'children'),
              Input('tabs-example-graph', 'value'))
def decomp_para(tab):
    if tab == 'tab-3-example-graph':
        return dbc.Card(
            dbc.CardBody([
                dbc.Row([dbc.Col(html.Div(html.H4('select the period:')),
                                 width=1),
                         dbc.Col(html.Div(dcc.Input(id='period_num', type='text', persistence=True)),
                                 width={'size': 2, 'offset': 0}),
                         dbc.Col(html.Div(html.H4('select the decomposition model:')),
                                 width={'size': 3, 'offset': 1}),
                         dbc.Col(html.Div(dcc.Dropdown(['additive ', 'multiplicative'], 'model', id='model_dropdown',
                                                       persistence=True)),
                                 width=2),
                         dbc.Col(
                             dbc.Button('decompose', id='decomp', n_clicks=0), width=2, className="d-grid gap-2"),

                         ], align='center')
            ]))


#########################
@app.callback(Output('drop', 'children'),
              Input('tabs-example-graph', 'value'),
              Input('decomp', 'n_clicks'),
              State('model_dropdown', 'value'),
              State('period_num', 'value'), )
def render_content(tab, n_clicks, model_dropdown_value, period_num_value):
    if tab == 'tab-3-example-graph' and n_clicks > 0:
        decomposition_results = seasonal_decompose(df.Close, model=str(model_dropdown_value),
                                                   period=int(period_num_value), extrapolate_trend=10)
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
        fig.add_trace(
            go.Scatter(x=decomposition_results.seasonal.index, y=decomposition_results.observed, mode="lines"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=decomposition_results.trend.index, y=decomposition_results.trend, mode="lines"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=decomposition_results.seasonal.index, y=decomposition_results.seasonal, mode="lines"),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=decomposition_results.resid.index, y=decomposition_results.resid, mode="lines"),
            row=4,
            col=1,
        )
        fig.update_layout(
            height=900, title="Seasonal Decomposition", margin=dict(t=100), title_x=0.5, showlegend=False
        )
        return dbc.Row(dbc.Col(dcc.Graph(id='sheeesh', figure=fig), width=8), justify="center", align="center")


#####################""""""


@app.callback(
    Output('modelllllll', 'children'),
    Input('demodropdown', 'value'),
    Input('tabs-example-graph', 'value'),
    Input('Forecast_button', 'n_clicks'),
    Input('forecast_steps', 'value'),
)
def update_output(demodropdown_value, tab, n_clicks, forecast_steps_value):
    if tab == 'tab-6-example-graph' and n_clicks > 0:
        if demodropdown_value == 'Prophet':
            model = Prophet(daily_seasonality=True)
            model.fit(dg)
            future_dates = model.make_future_dataframe(periods=int(forecast_steps_value))
            prediction = model.predict(future_dates)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prediction['ds'], y=prediction['yhat_upper'],
                                     fill=None,
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(
                x=prediction['ds'],
                y=prediction['yhat_lower'],
                fill='tonexty',  # fill area between trace0 and trace1
                mode='lines', line_color='lightblue', name='lower hand'))
            fig.add_trace(go.Scatter(x=prediction['ds'], y=prediction['yhat'],
                                     fill=None,
                                     mode='lines',
                                     line_color='blue', name='forecast'
                                     ))
            fig.add_trace(
                go.Scatter(x=dg['ds'], y=dg['y'], mode='markers', marker_color='black', marker_size=3,
                           name='actual value')
            )
            return dcc.Graph(id='sheeesh', figure=fig, className='eight columns offset-by-two column')
        if demodropdown_value == 'auto-ARIMA':
            model = auto_arima(df['Close'], start_p=1, start_q=1,
                               test='adf',
                               max_p=5, max_q=5,
                               m=1,
                               d=1,
                               seasonal=False,
                               start_P=0,
                               D=None,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
            prediction, confint = model.predict(n_periods=int(forecast_steps_value), return_conf_int=True)
            cf = pd.DataFrame(confint)
            pre = pd.DataFrame(prediction)
            future_dates = pd.date_range(df.index[-1], periods=int(forecast_steps_value), freq='B')
            cf = cf.set_index(future_dates)
            pre = pre.set_index(future_dates)
            cf['forecast'] = pre
            cf.columns = ['lower', 'upper', 'forecast']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cf.index, y=cf['upper'],
                                     fill=None,
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(x=cf.index, y=cf['lower'],
                                     fill='tonexty',
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines', line_color='blue', name='forecast'))

            fig.add_trace(go.Scatter(x=cf.index, y=cf['forecast'],
                                     fill=None,
                                     mode='lines',
                                     line_color='orange', name='forecast'
                                     ))
            return dcc.Graph(id='sheeesh', figure=fig, className='eight columns offset-by-two column')


#####################""""""
@app.callback(
    Output('ARIMA_viz', 'children'),
    Input('demodropdown', 'value'),
    Input('tabs-example-graph', 'value'),
    Input('Forecast_button', 'n_clicks'),
    Input('forecast_steps', 'value'),
    Input('porder', 'value'),
    Input('iorder', 'value'),
    Input('qorder', 'value'),

)
def update_output(demodropdown_value, tab, n_clicks, forecast_steps_value, porder_value, iorder_value, qorder_value):
    if tab == 'tab-6-example-graph' and n_clicks > 0:
        if demodropdown_value == 'ARIMA':
            model = SARIMAX(df.Close, order=(int(porder_value), int(iorder_value), int(qorder_value)))
            results = model.fit()
            forecast_values = results.get_forecast(steps=int(forecast_steps_value))
            pred_ci = forecast_values.conf_int()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 1],
                                     fill=None,
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 0],
                                     fill='tonexty',
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines', line_color='blue', name='forecast'))

            fig.add_trace(go.Scatter(x=forecast_values.predicted_mean.index, y=forecast_values.predicted_mean,
                                     fill=None,
                                     mode='lines',
                                     line_color='orange', name='forecast'
                                     ))
            return dcc.Graph(id='sheeesh', figure=fig, className='eight columns offset-by-two column')


#####################""""""

@app.callback(
    Output('SARIMA_viz', 'children'),
    Input('demodropdown', 'value'),
    Input('tabs-example-graph', 'value'),
    Input('Forecast_button', 'n_clicks'),
    Input('forecast_steps', 'value'),
    Input('sporder', 'value'),
    Input('siorder', 'value'),
    Input('sqorder', 'value'),

    Input('bsporder', 'value'),
    Input('bsiorder', 'value'),
    Input('bsqorder', 'value'),
    Input('morder', 'value'),
)
def update_output(demodropdown_value, tab, n_clicks, forecast_steps_value, sporder_value, siorder_value, sqorder_value,
                  bsporder_value, bsiorder_value, bsqorder_value, morder_value):
    if tab == 'tab-6-example-graph' and n_clicks > 0:
        if demodropdown_value == 'SARIMA':
            model = SARIMAX(df.Close, order=(int(sporder_value), int(siorder_value), int(sqorder_value)),
                            seasonal_order=(
                            int(bsporder_value), int(bsiorder_value), int(bsqorder_value), int(morder_value)))
            results = model.fit()
            forecast_values = results.get_forecast(steps=int(forecast_steps_value))
            pred_ci = forecast_values.conf_int()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 1],
                                     fill=None,
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 0],
                                     fill='tonexty',
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines', line_color='blue', name='forecast'))

            fig.add_trace(go.Scatter(x=forecast_values.predicted_mean.index, y=forecast_values.predicted_mean,
                                     fill=None,
                                     mode='lines',
                                     line_color='orange', name='forecast'
                                     ))
            return dcc.Graph(id='sheeesh', figure=fig, className='eight columns offset-by-two column')


#####################""""""


@app.callback(Output('PACF_ACF_plot', 'children'),
              Input('tabs-example-graph', 'value'),
              Input('submit-button', 'n_clicks'))
def decomp_para(tab, n_clicks):
    if tab == 'tab-4-example-graph' and n_clicks > 0:
        df_diff = df['Close'].diff().dropna()
        df_diffs = df['Close'].diff(5).dropna()
        return dbc.Card(
            dbc.CardBody([
                dbc.Row([dbc.Col(html.Div([dcc.Graph(id='timeseries',
                                                     config={'displayModeBar': False},
                                                     animate=True,
                                                     figure=px.line(df['Close'])),
                                           html.Div(html.P(adf_check(df['Close']))),
                                           dcc.Graph(id='timeseries',
                                                     config={'displayModeBar': False},
                                                     animate=True,
                                                     figure=px.line(df_diff
                                                                    )
                                                     ),
                                           html.Div(html.P(adf_check(df_diff)))
                                           ])
                                 )

                         ]), dbc.Row([dbc.Col(html.Div([dcc.Graph(id='timeseries',
                                                                  config={'displayModeBar': False},
                                                                  animate=True,
                                                                  figure=create_corr_plot(df_diff, plot_pacf=True)),

                                                        dcc.Graph(id='timeseries',
                                                                  config={'displayModeBar': False},
                                                                  animate=True,
                                                                  figure=create_corr_plot(df_diff, plot_pacf=False)
                                                                  )
                                                        ])
                                              )
                                      ]),dbc.Row([html.H4("seasonal componants:"), dbc.Col(html.Div([dcc.Graph(id='timeseries',
                                                                  config={'displayModeBar': False},
                                                                  animate=True,
                                                                  figure=create_corr_plot(df_diffs, plot_pacf=True)),

                                                        dcc.Graph(id='timeseries',
                                                                  config={'displayModeBar': False},
                                                                  animate=True,
                                                                  figure=create_corr_plot(df_diffs, plot_pacf=False)
                                                                  )
                                                        ])
                                              )
                                      ]) ,
            ]))


#####################


@app.callback(
    Output('auto-ARIMA_train', 'children'),
    Input('demodropdown', 'value'),
    Input('tabs-example-graph', 'value'),
    Input('TRAINY', 'n_clicks'),
    Input('myslider', 'value'),
)
def update_output(demodropdown_value, tab, n_clicks, myslider_value):
    if tab == 'tab-5-example-graph' and n_clicks > 0:
        if demodropdown_value == 'auto-ARIMA':
            train, test = train_test_split(df.Close, test_size=(100-int(myslider_value))/100, shuffle=False)

            model = auto_arima(df.Close, start_p=1, start_q=1,
                               test='adf',
                               max_p=5, max_q=5,
                               m=1,
                               d=1,
                               seasonal=False,
                               start_P=0,
                               D=None,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
            prediction, confint = model.predict(n_periods=len(test), return_conf_int=True)
            cf = pd.DataFrame(confint)
            pre = pd.DataFrame(prediction)
            future_dates = pd.date_range(train.index[-1], periods=len(test), freq='B')
            cf = cf.set_index(future_dates)
            pre = pre.set_index(future_dates)
            cf['forecast'] = pre
            cf.columns = ['lower', 'upper', 'forecast']

            testdf = pd.DataFrame(test)
            prediction_series = pd.Series(prediction, index=test.index)
            pddd = pd.DataFrame(prediction_series)
            inner_joined_total = pd.merge(pddd, testdf, on='Date')
            inner_joined_total.columns = ["forecast", "test"]

            RMSE1 = mean_absolute_percentage_error(inner_joined_total['test'], inner_joined_total['forecast'])




            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cf.index, y=cf['upper'],
                                     fill=None,
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(x=cf.index, y=cf['lower'],
                                     fill='tonexty',
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines', line_color='blue', name='forecast'))

            fig.add_trace(go.Scatter(x=cf.index, y=cf['forecast'],
                                     fill=None,
                                     mode='lines',
                                     line_color='orange', name='forecast'
                                     ))
            return dbc.Row([html.Div(f'the AIC VALUE IS: {model.aic()} , the MAPE VALUE IS: {RMSE1}' )
               ,dcc.Graph(id='sheeesh', figure=fig, className='eight columns offset-by-two column')])


#####################################""""
@app.callback(
    Output('ARIMA_train', 'children'),
    Input('demodropdown', 'value'),
    Input('tabs-example-graph', 'value'),
    Input('TRAINY', 'n_clicks'),
    Input('myslider', 'value'),
    Input('porder', 'value'),
    Input('iorder', 'value'),
    Input('qorder', 'value'),

)
def update_output(demodropdown_value, tab, n_clicks, myslider_value, porder_value, iorder_value, qorder_value):
    if tab == 'tab-5-example-graph' and n_clicks > 0:
        if demodropdown_value == 'ARIMA':

            train, test = train_test_split(df.Close, test_size=(100 - int(myslider_value)) / 100, shuffle=False)
            traindf = pd.DataFrame(train)
            testdf = pd.DataFrame(test)

            modelor = SARIMAX(df['Close'], order=(int(porder_value), int(iorder_value), int(qorder_value)))
            resultsor = modelor.fit()

            model = SARIMAX(traindf, order=(int(porder_value), int(iorder_value), int(qorder_value)))
            results = model.fit()
            forecast_values = results.get_forecast(steps=len(test))
            pred_ci = forecast_values.conf_int()

            RMSE1 = mean_absolute_percentage_error(testdf, forecast_values.predicted_mean)


            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 1],
                                     fill=None,
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 0],
                                     fill='tonexty',
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines', line_color='blue', name='price real value'))

            fig.add_trace(go.Scatter(x=forecast_values.predicted_mean.index, y=forecast_values.predicted_mean,
                                     fill=None,
                                     mode='lines',
                                     line_color='orange', name='forecast'
                                     ))

            residuals= resultsor.resid
            residuals = residuals[-(len(residuals) - 1):]


            mean = residuals.mean()
            std = residuals.std()
            residualsSTD = (residuals - mean) / std

            figdoom = px.line(residualsSTD)
            figdoom['layout'].update({
                'showlegend': False,
                'width': 700,
                'height': 500,
            })

            qqplot_data = qqplot(residuals, line='s').gca().lines
            fig1 = go.Figure()

            fig1.add_trace({
                'type': 'scatter',
                'x': qqplot_data[0].get_xdata(),
                'y': qqplot_data[0].get_ydata(),
                'mode': 'markers',
                'marker': {
                    'color': '#19d3f3'
                }
            })

            fig1.add_trace({
                'type': 'scatter',
                'x': qqplot_data[1].get_xdata(),
                'y': qqplot_data[1].get_ydata(),
                'mode': 'lines',
                'line': {
                    'color': '#636efa'
                }

            })

            fig1['layout'].update({
                'title': 'Quantile-Quantile Plot',
                'xaxis': {
                    'title': 'Theoritical Quantities',
                    'zeroline': False
                },
                'yaxis': {
                    'title': 'Sample Quantities'
                },
                'showlegend': False,
                'width': 700,
                'height': 500,
            })
            fig2 = px.histogram(residuals)
            fig2['layout'].update({
                'showlegend': False,
                'width': 700,
                'height': 500,
            })
            figacf= create_corr_plot(residuals, plot_pacf=False)
            figacf['layout'].update({
                'showlegend': False,
                'width': 700,
                'height': 500,
            })

            return dbc.Row([html.H3(f'          the AIC VALUE IS: {resultsor.aic} , the MAPE VALUE IS: {RMSE1}' ),
                            dcc.Graph(id='sheeesh', figure=fig),
                            dbc.Row([dbc.Col(dcc.Graph(id='shedaa', figure=figdoom)),
                            dbc.Col(dcc.Graph(id='shedaa', figure=figacf))]),
                            dbc.Row([dbc.Col( dcc.Graph(id='shedaa', figure=fig1)), dbc.Col(dcc.Graph(id='shedaa', figure=fig2))])])


##########################################"
@app.callback(
    Output('SARIMA_train', 'children'),
    Input('demodropdown', 'value'),
    Input('tabs-example-graph', 'value'),
    Input('TRAINY', 'n_clicks'),
    Input('myslider', 'value'),
    Input('sporder', 'value'),
    Input('siorder', 'value'),
    Input('sqorder', 'value'),

    Input('bsporder', 'value'),
    Input('bsiorder', 'value'),
    Input('bsqorder', 'value'),
    Input('morder', 'value')

)
def update_output(demodropdown_value, tab, n_clicks, myslider_value, sporder_value, siorder_value, sqorder_value,
                  bsporder_value, bsiorder_value, bsqorder_value, morder_value):
    if tab == 'tab-5-example-graph' and n_clicks > 0:
        if demodropdown_value == 'SARIMA':

            train, test = train_test_split(df.Close, test_size=(100 - int(myslider_value)) / 100, shuffle=False)
            traindf = pd.DataFrame(train)
            testdf = pd.DataFrame(test)

            modelor = SARIMAX(df.Close, order=(int(sporder_value), int(siorder_value), int(sqorder_value)),
                            seasonal_order=( int(bsporder_value), int(bsiorder_value), int(bsqorder_value), int(morder_value)))
            resultsor = modelor.fit()

            model = SARIMAX(traindf, order=(int(sporder_value), int(siorder_value), int(sqorder_value)),
                            seasonal_order=(int(bsporder_value), int(bsiorder_value), int(bsqorder_value), int(morder_value)))
            results = model.fit()
            forecast_values = results.get_forecast(steps=len(test))
            pred_ci = forecast_values.conf_int()

            RMSE1 = mean_absolute_percentage_error(testdf, forecast_values.predicted_mean)


            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 1],
                                     fill=None,
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 0],
                                     fill='tonexty',
                                     mode='lines',
                                     line_color='lightblue', name='upper hand'
                                     ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines', line_color='blue', name='price real value'))

            fig.add_trace(go.Scatter(x=forecast_values.predicted_mean.index, y=forecast_values.predicted_mean,
                                     fill=None,
                                     mode='lines',
                                     line_color='orange', name='forecast'
                                     ))
            residuals = resultsor.resid
            residuals = residuals[-(len(residuals) - 1):]
            d = 0
            for i in residuals:
                if i == residuals.min():
                    residuals[d] = residuals.mean()
                else:
                    d = d + 1

            mean = residuals.mean()
            std = residuals.std()
            residualsSTD = (residuals - mean) / std

            figdoom = px.line(residualsSTD)
            figdoom['layout'].update({
                'showlegend': False,
                'width': 700,
                'height': 500,
            })

            qqplot_data = qqplot(residuals, line='s').gca().lines
            fig1 = go.Figure()

            fig1.add_trace({
                'type': 'scatter',
                'x': qqplot_data[0].get_xdata(),
                'y': qqplot_data[0].get_ydata(),
                'mode': 'markers',
                'marker': {
                    'color': '#19d3f3'
                }
            })

            fig1.add_trace({
                'type': 'scatter',
                'x': qqplot_data[1].get_xdata(),
                'y': qqplot_data[1].get_ydata(),
                'mode': 'lines',
                'line': {
                    'color': '#636efa'
                }

            })

            fig1['layout'].update({
                'title': 'Quantile-Quantile Plot',
                'xaxis': {
                    'title': 'Theoritical Quantities',
                    'zeroline': False
                },
                'yaxis': {
                    'title': 'Sample Quantities'
                },
                'showlegend': False,
                'width': 700,
                'height': 500,
            })
            fig2 = px.histogram(residuals)
            fig2['layout'].update({
                'showlegend': False,
                'width': 700,
                'height': 500,
            })
            figacf = create_corr_plot(residuals, plot_pacf=False)
            figacf['layout'].update({
                'showlegend': False,
                'width': 700,
                'height': 500,
            })

            return dbc.Row([html.H3(f'          the AIC VALUE IS: {resultsor.aic} , the MAPE VALUE IS: {RMSE1}'),
                            dcc.Graph(id='sheeesh', figure=fig),
                            dbc.Row([dbc.Col(dcc.Graph(id='shedaa', figure=figdoom)),
                                     dbc.Col(dcc.Graph(id='shedaa', figure=figacf))]),
                            dbc.Row([dbc.Col(dcc.Graph(id='shedaa', figure=fig1)),
                                     dbc.Col(dcc.Graph(id='shedaa', figure=fig2))])])
#####################
if __name__ == '__main__':
    app.run_server(debug=True)
