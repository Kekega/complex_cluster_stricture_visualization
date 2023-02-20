import plotly.graph_objs as go
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import random

def mesh_2d_thresh_chart(X, y, dist_threshold=2):
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

        # Compute the centroid and the median distance
        centroid = tdf.mean()
        dists = np.linalg.norm(tdf - centroid, axis=1)
        median_dist = np.median(dists)

        # Filter out the points that are too far away
        tdf = tdf[dists < dist_threshold * median_dist]

        points = np.array(tdf)
        hull = ConvexHull(points)

        scatter = go.Scatter(
            x=tdf['x'], y=tdf['y'],
            mode='markers',
            marker=dict(
                size=3,
                color=label,
                opacity=0.8),
            name=str(label)
        )

        i, j = hull.simplices.transpose()
        mesh = go.Scatter(
            x=points[:, 0], y=points[:, 1],
            line=dict(width=2, color=col),
            fill='toself',
            fillcolor='rgba(0,0,0,0.1)',
            name=str(label)
        )

        fig.add_trace(scatter)
        fig.add_trace(mesh)

    fig.show()

# import plotly.graph_objs as go
# from scipy.spatial import Delaunay
# import numpy as np
# import pandas as pd
# import random
#
# def mesh_2d_thresh_chart(X, y, dist_threshold=3):
#     # Concatenate X and y arrays
#     arr_concat = np.concatenate((X, y.reshape(y.shape[0],1)), axis=1)
#     # Create a Pandas dataframe using the above array
#     df = pd.DataFrame(arr_concat, columns=['x', 'y', 'label'])
#     # Convert label data type from float to integer
#     df['label'] = df['label'].astype(int)
#     # Finally, sort the dataframe by label
#     df.sort_values(by='label', axis=0, ascending=True, inplace=True)
#
#     fig = go.Figure()
#     for label in df['label'].unique():
#         c1, c2, c3 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
#         col = f"rgb({c1},{c2},{c3})"
#
#         tdf = df.loc[df['label'] == label, ['x', 'y']]
#
#         # Compute the centroid and the median distance
#         centroid = tdf.mean()
#         dists = np.linalg.norm(tdf - centroid, axis=1)
#         median_dist = np.median(dists)
#
#         # Filter out the points that are too far away
#         tdf = tdf[dists < dist_threshold * median_dist]
#
#         points = np.array(tdf)
#         tri = Delaunay(points)
#
#         scatter = go.Scatter(
#             x=tdf['x'], y=tdf['y'],
#             mode='markers',
#             marker=dict(
#                 size=3,
#                 color=col,
#                 opacity=0.8),
#             name=str(label)
#         )
#
#         x, y = points.T
#         mesh = go.Scatter(
#             x=x[tri.simplices.flatten()], y=y[tri.simplices.flatten()],
#             mode='lines',
#             line=dict(width=2, color=col),
#             fill='toself',
#             fillcolor='rgba(0,0,0,0.1)',
#             name=str(label)
#         )
#
#         fig.add_trace(scatter)
#         fig.add_trace(mesh)
#
#     fig.show()

