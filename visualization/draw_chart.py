from visualization.chart_3d.scatter3d import scatter_3d
from visualization.chart_3d.mesh3d import mesh_3d
from visualization.chart_2d.mesh2d import mesh_2d
from visualization.chart_2d.scatter2d import scatter_2d

def draw_chart(X, y, dim: int, mesh: bool):
    if dim == 3:
        if mesh:
            mesh_3d(X, y)
        else:
            scatter_3d(X, y)
    else:
        if mesh:
            mesh_2d(X, y)
        else:
            scatter_2d(X, y)