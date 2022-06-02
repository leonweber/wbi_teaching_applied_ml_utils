import numpy as np
import matplotlib.pyplot as plt
import exercise3_utils

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
