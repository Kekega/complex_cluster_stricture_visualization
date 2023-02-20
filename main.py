from sklearn.datasets import load_digits # for MNIST data

# visualization utils
from reduce_methods.reduce_umap import reduce_umap
from reduce_methods.reduce_tsne import reduce_tsne
from visualization.draw_chart import draw_chart

# parse parameters
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-d', '--dimension', type=int, choices=[2, 3], required=True,
                    help='Dimension of the data: 2 or 3')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-u', '--umap', action='store_true', help='Use UMAP to process data')
group.add_argument('-t', '--tsne', action='store_true', help='Use t-SNE to process data')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-s', '--scatter', action='store_true', help='Draw only scatter on the plot')
group.add_argument('-m', '--mesh', action='store_true', help='Draw mesh surface on the plot')

args = parser.parse_args()

# Load arrays containing digit data (64 pixels per image) and their true labels
X, y = load_digits(return_X_y=True)


# Some stats
print('Shape of X (main data): ', X.shape)
print('Shape of y (true labels): ', y.shape)

print('Dimension:', args.dimension)

if args.umap:
    print('UMAP selected')
    X_trans = reduce_umap(X, n_components=args.dimension)
else:
    print('t-SNE selected')
    X_trans = reduce_tsne(X, n_components=args.dimension)

draw_chart(X_trans, y, dim=args.dimension, mesh=args.mesh)
