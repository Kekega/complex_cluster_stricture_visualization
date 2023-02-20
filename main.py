# # import umap
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# fashion_train = pd.read_csv("input/fashion-mnist_train.csv")
#
# data = fashion_train.iloc[:, 1:].values.astype(np.float32)
# target = fashion_train['label'].values
#
# reduce = umap.UMAP(random_state = 223) #just for reproducibility
# embedding = reduce.fit_transform(data)
#
# df = pd.DataFrame(embedding, columns=('x', 'y'))
# df["class"] = target
#
# labels = { 0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
#           5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8 : 'Bag', 9 : 'Ankle boot'}
#
# df["class"].replace(labels, inplace=True)
#
# sns.set_style("whitegrid", {'axes.grid' : False})
# #adjusting plot dots with plot_kws
# ax = sns.pairplot(x_vars = ["x"], y_vars = ["y"],data = df,
#              hue = "class",size=11, plot_kws={"s": 4});
# ax.fig.suptitle('Fashion MNIST clustered with UMAP') ;

# Data manipulation

# Visualization
# import trimesh

# Skleran
from sklearn.datasets import load_digits # for MNIST data

# UMAP dimensionality reduction

# Load digits data
digits = load_digits()

# Load arrays containing digit data (64 pixels per image) and their true labels
X, y = load_digits(return_X_y=True)

# Some stats
print('Shape of digit images: ', digits.images.shape)
print('Shape of X (main data): ', X.shape)
print('Shape of y (true labels): ', y.shape)

# Display images of the first 10 digits
# fig, axs = plt.subplots(2, 5, tight_layout=True, figsize=(12,6), facecolor='white')
# n=0
# plt.gray()
# for i in range(0,2):
#     for j in range(0,5):
#         axs[i,j].matshow(digits.images[n])
#         axs[i,j].set(title=y[n])
#         n=n+1
# plt.show()

c_test = ["rgb(255,0,0)", "rgb(0,255,0)",  "rgb(0,0,255)", "rgb(255,255,0)", "rgb(255,0,255)",
          "rgb(0,255,255)", "rgb(100,100,100)", "rgb(255,150,25)", "rgb(110,55,255)", "rgb(77,112,255)"]
objs = []


def getRGBfromI(RGBint):
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return red, green, blue


from visualization.mesh3d import mesh_3d_thresh_chart
from reduce_methods.reduce_umap import reduce_umap
from reduce_methods.reduce_tsne import reduce_tsne
from visualization.scatter2d import

# X_trans = reduce_umap(X)
X_trans = reduce_tsne(X)
mesh_3d_thresh_chart(X_trans, y)
