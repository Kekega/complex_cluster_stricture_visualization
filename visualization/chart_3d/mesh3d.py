from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, Delaunay
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import random

def mesh_3d_no_thresh_chart(X, y):
    # Create a Pandas dataframe using the X and y arrays
    df = pd.DataFrame(X, columns=['x', 'y', 'z'])
    df['label'] = y

    # Create a 3D graph
    fig = go.Figure()

    for label in df['label'].unique():
        c1, c2, c3 = random.sample(range(0, 255), 3)
        col = f"rgb({c1},{c2},{c3})"

        tdf = df.loc[df['label'] == label, ['x', 'y', 'z']]

        # Compute the convex hull
        points = np.array(tdf)
        hull = ConvexHull(points)

        # Create the scatter plot of the points
        scatter = go.Scatter3d(
            x=tdf['x'], y=tdf['y'], z=tdf['z'],
            mode='markers',
            marker=dict(
                size=3,
                color=col,
                opacity=0.8),
            name=str(label)
        )

        # Compute the interpolation
        interpolator = LinearNDInterpolator(points, tdf.index)
        xi = np.linspace(points[:, 0].min(), points[:, 0].max(), 20)
        yi = np.linspace(points[:, 1].min(), points[:, 1].max(), 20)
        zi = np.linspace(points[:, 2].min(), points[:, 2].max(), 20)
        xi, yi, zi = np.meshgrid(xi, yi, zi, indexing='ij')
        points_interp = np.array([xi.ravel(), yi.ravel(), zi.ravel()]).T
        values_interp = interpolator(points_interp)
        mesh_points = points_interp[values_interp >= 0]
        hull_interp = ConvexHull(mesh_points)

        # Create the mesh plot of the convex hull of the interpolation
        i_interp, j_interp, k_interp = hull_interp.simplices.transpose()
        mesh_interp = go.Mesh3d(
            x=mesh_points[:, 0], y=mesh_points[:, 1], z=mesh_points[:, 2],
            i=i_interp, j=j_interp, k=k_interp,
            opacity=0.2,
            color=col,
            name=f'{label} interp'
        )

        # Create the mesh plot of the convex hull of the original points
        i, j, k = hull.simplices.transpose()
        mesh = go.Mesh3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            i=i, j=j, k=k,
            opacity=0.2,
            color=label,
            name=str(label)
        )

        fig.add_trace(scatter)
        fig.add_trace(mesh_interp)
        # fig.add_trace(mesh)

    fig.show()


def mesh_3d_thresh_chart(X, y, dist_threshold=3):
    # Concatenate X and y arrays
    arr_concat = np.concatenate((X, y.reshape(y.shape[0],1)), axis=1)
    # Create a Pandas dataframe using the above array
    df = pd.DataFrame(arr_concat, columns=['x', 'y', 'z', 'label'])
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)

    fig = go.Figure()
    for label in df['label'].unique():
        c1, c2, c3 = random.sample(range(0, 255), 3)
        col = f"rgb({c1},{c2},{c3})"

        tdf = df.loc[df['label'] == label, ['x', 'y', 'z']]

        # Compute the centroid and the median distance
        centroid = tdf.mean()
        dists = np.linalg.norm(tdf - centroid, axis=1)
        median_dist = np.median(dists)

        # Filter out the points that are too far away
        tdf = tdf[dists < dist_threshold * median_dist]

        points = np.array(tdf)
        hull = ConvexHull(points)

        scatter = go.Scatter3d(
            x=tdf['x'], y=tdf['y'], z=tdf['z'],
            mode='markers',
            marker=dict(
                size=3,
                color=col,
                opacity=0.8),
            name=str(label)
        )

        i, j, k = hull.simplices.transpose()
        mesh = go.Mesh3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            i=i, j=j, k=k,
            opacity=0.2,
            color=col,
            name=str(label)
        )

        fig.add_trace(scatter)
        fig.add_trace(mesh)

    fig.show()
