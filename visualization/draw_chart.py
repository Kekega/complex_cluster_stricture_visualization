from visualization.chart_3d.scatter3d import scatter_3d_chart
from visualization.chart_3d.mesh3d import mesh_3d_thresh_chart
from visualization.chart_2d.mesh2d import mesh_2d_chart
from visualization.chart_2d.scatter2d import scatter_2d

def draw_chart(X, y, dim: int, mesh: bool):
    if dim == 3:
        if mesh:
            mesh_3d_thresh_chart(X, y)
        else:
            scatter_3d_chart(X, y)
    else:
        if mesh:
            mesh_2d_chart(X, y)
        else:
            scatter_2d(X, y)