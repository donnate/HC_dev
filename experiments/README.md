The purpose of this experiments folder is to test our method and benchmark it against other hierarchical clustering benchmarks.

We divide the experiments along the four main axes of investigation:

__1. Ability of the method to recover modular substructures:__ this is assessed on a variety of synthetic experiments,
providing different ways of creating hierarchies (Gaussian Mixtures, nested subcliques, Stochastic Block models with varying densities. In this setting, we have groun
truth labels at our disposal, and are able to assess the accuracy of the recovered hierarchy; The accuracy 
of the model is averaged over 100 trials (that is, random graph instances).

__2. Ability of the method to be robust against noise and perturbations__: this is evaluated through a real dataset.

__3. Comparison of the results with other hierarchical clustering techniques__: we enrich the analysis of our model
by comparing it to: (a) average linkage (which seems to be people's favorite in applied studies), (b) ward linkage,
(c) Spectral clustering with a varying number of clusters (d) robust hierarchical clustering [1] and (e) HC with spreading metrics [2].

__4. Understanding the method__: the goal here is to get a full picture of the strengths and weaknesses of the method, and the influence of its parameter $alpha$.: what graph representation is the most suitable? What is its impact on the results? How does the value of alpha influence the results?

In order to assess the quality of our method, we  rely on different standard clustering and classification metrics:

+ __F-score__: measures the trade-off between precision and recall and relies only on the label of each instance. Since the clusters are balanced by design, we here use the 'macro' F1 score, which is the average one-vs-rest F1 score over the different classes 
+ __RSM matrices__: (Representational Similarity Matrices): the goal here is to measure the correlation between the distances in the tree induced by two different HC. We represent each HC by a square matrix giving similarites between nodes according to the HC: each entry (i,j) is the distance between node i and j induced by th HC, integrated over the whole tree. It is thus 0 if the two nodes are in branches that diverge directly from the root onwards, and 1 - k/tree_depth with k the level of the clsest common ancestor of i and j, and tree_depth the overall tree depth. For non-tree-like hierarchical clustering, this similarity is the Riemann integral of the similarities over the regularization path. This allows us to compare easiy the tree-like and non-tree like HC by taking the (rank-based) Spearman correlation between induced.
+ __Silhouette scores__ : The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. This measures the "discriminative" power of the clustering: are all clusters well separated?
+ __Calinski Harabaz score__ ratio between within-cluster dispersion and the between-cluster dispersion.
+ __Folkes Mallows score__ : similarity between two clusterings
+ __homogeneity, completeness__: A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class. A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.Both scores have positive values between 0.0 and 1.0, larger values being desirable.
+ __The Matthew coefficient__: The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction. The statistic is also known as the phi coefficient. [source: Wikipedia]


[1] Balcan, Maria-Florina, Yingyu Liang, and Pramod Gupta. "Robust hierarchical clustering." The Journal of Machine Learning Research 15.1 (2014): 3831-3871.

[2] Roy, Aurko, and Sebastian Pokutta. "Hierarchical clustering via spreading metrics." Advances in Neural Information Processing Systems. 2016.
