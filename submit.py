import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

################################
# Non Editable Region Starting #
################################
def my_fit(X_train, y_train):
################################
#  Non Editable Region Ending  #
################################
    
    features = my_map(X_train)
    # Fit logistic regression
    model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=100)
    model.fit(features, y_train)

    # Fit linearSVC
    # model = LinearSVC(penalty='l1', max_iter=1000, dual=False)
    # model.fit(features, y_train)

    # Return model weights and intercept
    return model.coef_.flatten(), model.intercept_[0]


################################
# Non Editable Region Starting #
################################
def my_map(X):
################################
#  Non Editable Region Ending  #
################################
    # Convert X from {0, 1} to {-1, 1}
    d = 1 - 2 * X  # shape (n, 8)

    # Compute xi: cumulative product from i to 7
    xi = np.cumprod(d[:, ::-1], axis=1)[:, ::-1]  # shape (n, 8)

    # Create monomials from xi (36 features)
    n = d.shape[0]
    xi_monomials = []
    for i in range(8):
        for j in range(i, 8):
            xi_monomials.append((xi[:, i] * xi[:, j]).reshape(n, 1))
    xi_monomials = np.hstack(xi_monomials)

    # Create monomials from di (36 features)
    di_monomials = []
    for i in range(8):
        for j in range(i, 8):
            di_monomials.append((d[:, i] * d[:, j]).reshape(n, 1))
    di_monomials = np.hstack(di_monomials)

    # Concatenate all features: di (8), xi (8), xi_monomials (36), di_monomials (36) = 88
    features = np.hstack([d, xi, xi_monomials, di_monomials])

    # Add bias term (0.5) to make 89 features
    features = np.hstack([features,  0.5 * np.ones((features.shape[0], 1))])

    return features



################################
# Non Editable Region Starting #
################################
def my_decode(w):
################################
#  Non Editable Region Ending  #
################################
    w = np.array(w)
    alpha = np.zeros(64)
    beta = np.zeros(64)

    # Step 1: Reconstruct alpha and beta
    alpha[0] = w[0]
    beta[63] = w[64]
    for i in range(63, 0, -1):
        alpha[i] = w[i] - beta[i - 1]
        beta[i - 1] = w[i] - alpha[i]

    # Step 2: Compute differences
    d = alpha + beta  # p - q
    c = alpha - beta  # r - s

    # Step 3: Choose q, s such that p, q, r, s are all positive
    q = 1.0 + np.maximum(0, -d)  # qᵢ = 1.0 + [-dᵢ]+
    s = 1.0 + np.maximum(0, -c)  # sᵢ = 1.0 + [-cᵢ]+
    p = d + q
    r = c + s

    # Step 4: Return all non-negative delays
    return p.tolist(), q.tolist(), r.tolist(), s.tolist()