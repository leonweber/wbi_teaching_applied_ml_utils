import numpy as np
import matplotlib.pyplot as plt

__version__ = '0.0.3'


def plotData(x_train, y_train, x_test, y_test):
    plt.figure(figsize=(10,5))
    plt.scatter(x_train, y_train, label='train data')
    plt.scatter(x_test, y_test, label='test data')
    plt.ylabel('Price in 1000$')
    plt.xlabel('Size in 100 sq-feet')
    plt.legend(loc=4)

def plotLine(x_train, y_train, x_test, y_test, w):
    plotData(x_train, y_train, x_test, y_test)

    # Regression Line
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = w[0] + w[1] * x_vals
    plt.plot(x_vals, y_vals, '-')


def plotPolyLine(x_train, y_train, x_test, y_test, w):
    plotData(x_train, y_train, x_test, y_test)

    # Regression Line
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = w[0] + w[1] * x_vals
    plt.plot(x_vals, y_vals, '-')

    # Regression Polynom
    X_poly = mapPolynomialFeatures(x[:, 0], np.ones(len(x[:, 0])), 3)
    w_poly = normalEqn(X_poly, y);
    x1 = np.float32(np.linspace(500, 4500, 1000))
    x2 = np.float32(np.linspace(1, 1, 1000))
    polys = mapPolynomialFeatures(x1, x2, 3)

    y_vals = predictPrice(polys, w_poly)
    plt.plot(x1, y_vals, '.')

def plotLossFunction(X, y, w0_vals, w1_vals, L_vals, w):
    # surface plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(w0_vals, w1_vals, L_vals, cmap='viridis')
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.title('Surface')

    # contour plot
    ax = plt.subplot(122)
    plt.contour(w0_vals, w1_vals, L_vals, linewidths=2, levels=20)
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.plot(w[0], w[1], 'ro', ms=10, lw=2)
    plt.title('Contour, showing minimum')


def plot_one(X_train, y_train, X_test, y_test, degree_predictions, x_interval, degree):
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate(degree):
        plt.plot(x_interval, degree_predictions[i],
                 alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.legend(loc=4)
    plt.ylim(0,1000)
    plt.xlim(0,60)

def plot_validation_curve(mse_train, mse_test, degrees):
    plt.figure(figsize=(14,5))
    plt.title('Validation Curve')
    plt.xlabel('poly')
    plt.ylabel('RMSE')
    plt.ylim(20, 100)
    plt.xticks(np.arange(min(degrees), max(degrees)+1, 1.0))

    plt.plot(degrees, np.sqrt(mse_train), label='Training MSE', color='darkorange', lw=2)
    plt.plot(degrees, np.sqrt(mse_test), label='Test MSE', color='navy', lw=2)
    plt.legend(loc='best')
    plt.show()


def plot_polynomial_rmse(polys, Ls_poly_train):
    plt.figure(figsize=(8,5))  
    plt.plot(polys, Ls_poly_train, '-', label="RMSE")
    plt.xlabel('polynomial degree')
    plt.ylabel('RMSE')
    plt.title('RMSE for varying polynomial degrees')
    _ = plt.legend(loc='best')

class Exercise3Utils:
    @staticmethod
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

    @staticmethod   
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

    @staticmethod
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

    @staticmethod
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