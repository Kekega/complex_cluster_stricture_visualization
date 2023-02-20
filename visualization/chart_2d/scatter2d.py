import plotly.graph_objs as go
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import random

def scatter_2d(X, y, dist_threshold=2):
    # Concatenate X and y arrays
    arr_concat = np.concatenate((X, y.reshape(y.shape[0],1)), axis=1)
    # Create a Pandas dataframe using the above array
    df = pd.DataFrame(arr_concat, columns=['x', 'y', 'label'])
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)

    fig = go.Figure()
    for label in df['label'].unique():
        c1, c2, c3 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        col = f"rgb({c1},{c2},{c3})"
        tdf = df.loc[df['label'] == label, ['x', 'y']]

        scatter = go.Scatter(
            x=tdf['x'], y=tdf['y'],
            mode='markers',
            marker=dict(
                size=3,
                color=col,
                opacity=0.8),
            name=str(label)
        )

        fig.add_trace(scatter)

    fig.show()
