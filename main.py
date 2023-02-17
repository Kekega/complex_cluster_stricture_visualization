# import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fashion_train = pd.read_csv("input/fashion-mnist_train.csv")

data = fashion_train.iloc[:, 1:].values.astype(np.float32)
target = fashion_train['label'].values

reduce = umap.UMAP(random_state = 223) #just for reproducibility
embedding = reduce.fit_transform(data)

df = pd.DataFrame(embedding, columns=('x', 'y'))
df["class"] = target

labels = { 0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
          5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8 : 'Bag', 9 : 'Ankle boot'}

df["class"].replace(labels, inplace=True)

sns.set_style("whitegrid", {'axes.grid' : False})
#adjusting plot dots with plot_kws
ax = sns.pairplot(x_vars = ["x"], y_vars = ["y"],data = df,
             hue = "class",size=11, plot_kws={"s": 4});
ax.fig.suptitle('Fashion MNIST clustered with UMAP') ;