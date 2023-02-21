import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import random

def mesh_2d(X, y, dist_threshold=3):
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
        c1, c2, c3 = random.sample(range(0, 255), 3)
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

        # Compute the centroid and the median distance
        centroid = tdf.mean()
        dists = np.linalg.norm(tdf - centroid, axis=1)
        median_dist = np.median(dists)

        # Filter out the points that are too far away
        tdf = tdf[dists < dist_threshold * median_dist]

        points = np.array(tdf)
        hull = ConvexHull(points)

        x_hull = np.append(points[hull.vertices,0],
                           points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1],
                           points[hull.vertices,1][0])
        mesh = go.Scatter(
            x=x_hull, y=y_hull,
            fill='toself',
            fillcolor='rgba(0, 0, 0, 0)',
            line=dict(color=col),
            name=str(label)
        )

        fig.add_trace(scatter)
        fig.add_trace(mesh)

    fig.show()
