import sys
import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
    fig = plt.figure(figsize=(12,8))

    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'x', lw=2, ms=10)
    plt.plot(X[neg, 0], X[neg, 1], 'o', ms=10)

    plt.xlabel('Normalized Exam 1 score')
    plt.ylabel('Normalized Exam 2 score')

    plt.legend(['Admitted', 'Not admitted'])


def mapFeature(X1, X2, degree=6):
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0], dtype=np.float64)]
    else:
        out = [np.ones(1, dtype=np.float64)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1, dtype=np.float64)
    else:
        return np.array(out, dtype=np.float64)


def plotDecisionBoundary(plotData, theta, X, y, degree=6):
    # make sure theta is a numpy array
    theta = np.array(theta)

    # Plot Data (remember first column in X is the intercept)
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]), np.max(X[:, 1])])

        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        #plt.xlim([1, 100])
        #plt.ylim([1, 100])
    else:
        # Here is the grid range
        u = np.linspace(-2, 2, 50)
        v = np.linspace(-2, 2, 50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(ui, vj, degree), theta)

        z = z.T  # important to transpose z before calling contour

        plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')

    plt.tight_layout()

def vis_coef(estimator, feature_names, topn = 10):
    """
    Visualize the top-n most influential coefficients
    for linear models.
    """
    fig = plt.figure(figsize=(10,15))
    feature_names = np.array(feature_names)

    coefs  = estimator.coef_[0]
    sorted_coefs = np.argsort(coefs)
    positive_coefs = sorted_coefs[-topn:]
    negative_coefs = sorted_coefs[:topn]

    top_coefs = np.hstack([negative_coefs, positive_coefs])
    colors = ['r' if c < 0 else 'b' for c in coefs[top_coefs]]
    y_pos = np.arange(2 * topn)
    plt.barh(y_pos, coefs[top_coefs], color = colors, align = 'center')
    plt.yticks(y_pos, feature_names[top_coefs])
    plt.title('top {} positive/negative words'.format(topn))

    plt.tight_layout()
    %