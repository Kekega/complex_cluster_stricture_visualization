from sklearn.manifold import TSNE # for t-SNE dimensionality reduction

def reduce_tsne(X, n_components=2):
    embed = TSNE(
        n_components=n_components,  # default=2, Dimension of the embedded space.
        perplexity=10,
        # default=30.0, The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
        early_exaggeration=12,
        # default=12.0, Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them.
        learning_rate=200,
        # default=200.0, The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high, the data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the learning rate is too low, most points may look compressed in a dense cloud with few outliers.
        n_iter=5000,  # default=1000, Maximum number of iterations for the optimization. Should be at least 250.
        n_iter_without_progress=300,
        # default=300, Maximum number of iterations without progress before we abort the optimization, used after 250 initial iterations with early exaggeration.
        min_grad_norm=0.0000001,
        # default=1e-7, If the gradient norm is below this threshold, the optimization will be stopped.
        metric='euclidean',
        # default=’euclidean’, The metric to use when calculating distance between instances in a feature array.
        init='random',
        # {‘random’, ‘pca’} or ndarray of shape (n_samples, n_components), default=’random’. Initialization of embedding
        verbose=0,  # default=0, Verbosity level.
        random_state=42,
        # RandomState instance or None, default=None. Determines the random number generator. Pass an int for reproducible results across multiple function calls.
        method='barnes_hut',
        # default=’barnes_hut’. By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time. method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time. The exact algorithm should be used when nearest-neighbor errors need to be better than 3%.
        angle=0.5,
        # default=0.5, Only used if method=’barnes_hut’ This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        n_jobs=-1,
        # default=None, The number of parallel jobs to run for neighbors search. -1 means using all processors.
    )

    X_embedded = embed.fit_transform(X)

    return X_embedded