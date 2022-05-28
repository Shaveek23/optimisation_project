import plotly.express as px
import pandas as pd


def visualize_cost(costs):
    
    df = pd.DataFrame(costs)
    fig = px.line(df, x=df.index, y=df[0])

    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x")

    fig.update_xaxes(title='iteration')
    fig.update_yaxes(title='cost')
    fig.update_layout(title='Cost function values for training iterations')

    fig.show()
