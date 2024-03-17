import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
from util_func import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# Load your CSV data into a DataFrame
path_file_csv = ''
df = read_rew_data(path_file_csv)
df_FR = get_result_forecast(path_file_csv)
df_FR['prediction_AR'] = autoreg_benchmark(df['Power_kW'],df_FR['test'])
error_df = error_df(df_FR)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

# App layout
app.layout = html.Div(children=[
    html.H1(children='Dynamic Data Visualization'),
    # Tabs
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1', children=[
            # Checklist for selecting variables to plot
            html.Div([
                html.H4('Select Variables:'),
                dcc.Checklist(
                    id='variable-selector',
                    options=[{'label': i, 'value': i} for i in df.columns if i != 'Date'],
                    value=[df.columns[0]],  # Default selected value
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            # Graph that updates based on the selected variables
            html.Div([
                html.H4('IST Raw Data'),
                dcc.Graph(id='yearly-data'),
            ])
        ]),
        dcc.Tab(label='Forecast results', value='tab-2', children=[
            # You can add different content here for the second tab
            html.Div([
                html.H4('Forcast results'),
                # Example content: static graph, text, or any other Dash component
                dcc.Graph(figure=px.line(df_FR))  # Example static graph
            ]),
            dash_table.DataTable(
                id='dynamic-table',
                columns=[{'name': i, 'id': i} for i in error_df.columns],  # Initial columns
                data=error_df.to_dict('records'),  # Convert DataFrame to a list of dictionaries
            )
        ]),
    ]),
])


# Callback to update graph based on selected variables
@app.callback(
    Output('yearly-data', 'figure'),
    [Input('variable-selector', 'value')]
)
def update_graph(selected_variables):
    fig = px.line(df, y=selected_variables)
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server()
