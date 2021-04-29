import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class RBFModule(object):
    def __init__(self, hidden_shape, centers=None, sigma=1.0):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers_all = centers
        self.centers_sele = None
        self.lr = LinearRegression()
        self.poly = PolynomialFeatures(1)
    
    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)
    
    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers_sele):
                G[data_point_arg, center_arg] = self._kernel_function(
                        center, data_point)
        return G
    
    def _select_centers(self, X):
        #random_args = np.random.choice(len(X), self.hidden_shape)
        #centers = X[random_args]
        kmeans = KMeans(n_clusters=self.hidden_shape, init='random', random_state=0).fit(X)
        centers = kmeans.cluster_centers_
        return centers
    
    def fit(self, X, Y):
        if self.centers_all is None:
            self.centers_all = X
        self.centers_sele = self._select_centers(self.centers_all) 
        G = self._calculate_interpolation_matrix(X)
        G = self.poly.fit_transform(G)
        self.lr.fit(G, Y)
        # self.weights = np.dot(np.linalg.pinv(G), Y)
        
    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        G = self.poly.fit_transform(G)
        predictions = self.lr.predict(G)
        # predictions = np.dot(G, self.weights)
        return predictions
