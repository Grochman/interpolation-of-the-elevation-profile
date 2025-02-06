import pandas as pd
import matplotlib.pyplot as plt
import math


def faktoryzacjaLU(A,b):
    n = len(A)
    U = [[0] * n for _ in range(n)]
    L = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j == i:
                L[i][j] = 1
    for i in range(n):
        for j in range(n):
            if j >= i:
                suma = 0
                for k in range(i):
                    suma += L[i][k]*U[k][j]
                U[i][j] = A[i][j] - suma
            if j >= i+1:
                suma = 0
                for k in range(i):
                    suma += L[j][k]*U[k][i]
                L[j][i] = (A[j][i] - suma) / U[i][i]

    y = [0]*n
    y[0] = b[0]/L[0][0]
    for i in range(1,n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j]*y[j]
        y[i] /= L[i][i]

    x = [0]*n
    x[n-1] = y[n-1]/U[n-1][n-1]
    for i in range(n-2, -1, -1):
        x[i] = y[i]
        for j in range(n-1, i, -1):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    return x


def lagrange(x, x_data, y_data):
    n = len(x_data)
    result = 0

    for i in range(n):
        phi = 1
        for j in range(n):
            if i != j:
                phi *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += phi*y_data[i]

    return result

def plotLagrange(x, y, points):
    original_size = len(x)
    reduction = math.floor(original_size/(points-1)-1)
    new_size = reduction*(points-1)
    if new_size < original_size:
        new_size += 1
    x = x[:new_size]
    y = y[:new_size]
    x_probed = x[::reduction]
    y_probed = y[::reduction]
    x_interpolated = []
    y_interpolated = []
    for i in range(0, new_size, 1):
        x_interpolated.append(x[i])
        y_interpolated.append(lagrange(x[i], x_probed, y_probed))

    plt.title(f"Lagrange interpolation with {points} points")
    plt.xlabel('distance')
    plt.ylabel('height')
    plt.plot(x_probed, y_probed, 'o')
    plt.plot(x, y)
    plt.plot(x_interpolated, y_interpolated)
    plt.legend(['points for interpolation', 'original', 'interpolated'])
    plt.show()

def plotsLagrange(path):
    data = pd.read_csv(path)
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    x = x.tolist()
    y = y.tolist()
    plotLagrange(x, y, 5)
    plotLagrange(x, y, 8)
    plotLagrange(x, y, 11)
    plotLagrange(x, y, 13)


def makeMatrix2(points, h):
    matrix_size = (points - 1) * 4
    matrix = []

    for i in range(matrix_size):
        matrix.append([])
        counter = 0
        for j in range(matrix_size):
            if i%4 == 0:
                if i == j:
                    matrix[-1].append(1)
                else:
                    matrix[-1].append(0)
            elif i%4 == 1:
                if i + 3 > j >= i-1:
                    matrix[-1].append(h ** counter)
                    counter += 1
                else:
                    matrix[-1].append(0)
            elif i%4 == 2:
                if i == 2:
                    if j == 2:
                        matrix[-1].append(2)
                    else:
                        matrix[-1].append(0)
                else:
                    if i - 4 == j:
                        matrix[-1].append(2)
                    elif i - 3 == j:
                        matrix[-1].append(6 * h)
                    elif i == j:
                        matrix[-1].append(-2)
                    else:
                        matrix[-1].append(0)
            elif i%4==3:
                if i == matrix_size-1:
                    if j == matrix_size-2:
                        matrix[-1].append(2)
                    elif j == matrix_size-1:
                        matrix[-1].append(6*h)
                    else:
                        matrix[-1].append(0)
                else:
                    if i - 2 == j:
                        matrix[-1].append(1)
                    elif i - 1 == j:
                        matrix[-1].append(2 * h)
                    elif i == j:
                        matrix[-1].append(3 * h * h)
                    elif i + 2 == j:
                        matrix[-1].append(-1)
                    else:
                        matrix[-1].append(0)

    return matrix

def spline(x, y, points, x_probed, y_probed):
    h = x_probed[1]-x_probed[0]
    matrix = makeMatrix2(points, h)

    y_solution = []
    for i in range(len(y_probed)-1):
        y_solution.append(y_probed[i])
        y_solution.append(y_probed[i+1])
        y_solution.append(0)
        y_solution.append(0)

    result_vector = faktoryzacjaLU(matrix, y_solution)

    x_interpolated = []
    y_interpolated = []
    index = 0
    for i in range(0, len(x), 1):
        x_interpolated.append(x[i])
        if x_interpolated[i] > x_probed[index+1]:
            index += 1
        xn = x[i]-x_probed[index]
        y_interpolated.append(result_vector[index*4] + result_vector[index*4+1]*xn + result_vector[index*4+2]*xn**2 + result_vector[index*4+3]*xn**3)

    return x_interpolated, y_interpolated

def plotSpline(x, y, points):
    original_size = len(x)
    reduction = math.floor(original_size / (points - 1)-1)
    new_size = reduction * (points - 1)
    if new_size < original_size:
        new_size += 1
    x = x[:new_size]
    y = y[:new_size]
    x_probed = x[::reduction]
    y_probed = y[::reduction]
    x_interpolated, y_interpolated = spline(x,y,points, x_probed, y_probed)

    plt.title(f"Spline 3rd degree interpolation with {points} points")
    plt.xlabel('distance')
    plt.ylabel('height')
    plt.plot(x_probed, y_probed, 'o')
    plt.plot(x, y)
    plt.plot(x_interpolated, y_interpolated)
    plt.legend(['points for interpolation', 'original', 'interpolated'])
    plt.show()

def plotsSpline(path):
    data = pd.read_csv(path)
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    x = x.tolist()
    y = y.tolist()
    plotSpline(x, y, 5)
    plotSpline(x, y, 8)
    plotSpline(x, y, 11)
    plotSpline(x, y, 13)


def plotCzebyszew(x, y, type):
    points = 17
    original_size = len(x)
    reduction = math.floor(original_size / (points - 1) - 1)
    new_size = reduction * (points - 1)
    if new_size < original_size:
        new_size += 1
    x = x[:new_size]
    y = y[:new_size]
    x_probed = []
    y_probed = []
    if type == 'ends':
        x_probed = x[::reduction]
        y_probed = y[::reduction]
        x_probed[1] = x[10]
        x_probed[2] = x[20]
        y_probed[1] = y[10]
        y_probed[2] = y[20]

        x_probed[-2] = x[-10]
        x_probed[-3] = x[-20]
        y_probed[-2] = y[-10]
        y_probed[-3] = y[-20]
    elif type == 'czebyszew1':
        for i in range(points):
            x_probed.append(-math.cos((2*i+1)*3.14/(2*points)))
            y_probed.append(y[math.floor((1+x_probed[-1])/2*len(x))])
            x_probed[-1] = x[math.floor((1+x_probed[-1])/2*len(x))]
    elif type == 'czebyszew2':
        for i in range(points):
            x_probed.append(-math.cos(i*3.14/(points-1)))
            y_probed.append(y[math.floor((1 + x_probed[-1]) / 2 * len(x))])
            x_probed[-1] = x[math.floor((1 + x_probed[-1]) / 2 * len(x))]
    else:
        x_probed = x[::reduction]
        y_probed = y[::reduction]

    x_interpolated = []
    y_interpolated = []
    for i in range(0, new_size, 1):
        x_interpolated.append(x[i])
        y_interpolated.append(lagrange(x[i], x_probed, y_probed))

    plt.title(f"{type} interpolation with {points} points")
    plt.xlabel('distance')
    plt.ylabel('height')
    plt.plot(x_probed, y_probed, 'o')
    plt.plot(x, y)
    plt.plot(x_interpolated, y_interpolated)
    plt.legend(['points for interpolation', 'original', 'interpolated'])
    plt.show()

def czebyszew(path):
    data = pd.read_csv(path)
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    x = x.tolist()
    y = y.tolist()
    plotCzebyszew(x, y, 'normal')
    plotCzebyszew(x, y, 'ends')
    plotCzebyszew(x, y, 'czebyszew1')
    plotCzebyszew(x, y, 'czebyszew2')


def main():
    plotsLagrange('2018_paths/100.csv')
    plotsLagrange('2018_paths/Obiadek.csv')
    czebyszew('2018_paths/100.csv')
    plotsSpline('2018_paths/100.csv')
    plotsSpline('2018_paths/Obiadek.csv')

if __name__ == '__main__':
    main()
