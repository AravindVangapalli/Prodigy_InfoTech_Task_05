import numpy as np
np.random.seed(42)
num_samples = 1000
num_features = 4
X = np.random.rand(num_samples, num_features)
y = (X[:, 0] + X[:, 1] > 1).astype(int) 
split_ratio = 0.8
split_index = int(split_ratio * num_samples)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
def gini_impurity(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini = 1 - np.sum(probabilities ** 2)
    return gini
class TreeNode:
    def __init__(self, data_indices, gini):
        self.data_indices = data_indices
        self.gini = gini
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.prediction = None
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)  
    def _grow_tree(self, X, y, depth=0):
        data_indices = np.arange(len(y))
        gini = gini_impurity(y)
        node = TreeNode(data_indices, gini)
        if depth < self.max_depth:
            best_gini = gini
            best_criteria = None
            best_sets = None

            for feature_index in range(self.n_features):
                thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))
                class_counts = np.bincount(classes)
                for i in range(1, len(class_counts)):
                    left_indices = np.where(X[:, feature_index] <= thresholds[i])[0]
                    right_indices = np.where(X[:, feature_index] > thresholds[i])[0]
                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue
                    left_gini = gini_impurity(y[left_indices])
                    right_gini = gini_impurity(y[right_indices])
                    weighted_gini = (len(left_indices) / len(y)) * left_gini + (len(right_indices) / len(y)) * right_gini            
                    if weighted_gini < best_gini:
                        best_gini = weighted_gini
                        best_criteria = (feature_index, thresholds[i])
                        best_sets = (left_indices, right_indices)
                        continue
            if best_gini < gini:
                left = self._grow_tree(X[best_sets[0]], y[best_sets[0]], depth + 1)
                right = self._grow_tree(X[best_sets[1]], y[best_sets[1]], depth + 1)
                node.feature_index = best_criteria[0]
                node.threshold = best_criteria[1]
                node.left = left
                node.right = right
                return node
        class_counts = np.bincount(y[node.data_indices])
        node.prediction = np.argmax(class_counts)
        return node
    def predict(self, X):
        return [self._predict_tree(self.tree, x) for x in X]
    def _predict_tree(self, node, x):
        if node.prediction is not None:
            return node.prediction
        if x[node.feature_index] <= node.threshold:
            return self._predict_tree(node.left, x)
        else:
            return self._predict_tree(node.right, x)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
print("There is 0.49 of accuracy  whether a customer will purchase a product or service based on demographic and behavioral data")
