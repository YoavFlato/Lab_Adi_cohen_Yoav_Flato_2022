import os.path

import results
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import openpyxl
from tqdm import tqdm


def halfplate_experiment(relative_error=True):
    xdata = [float(x) for x in results.lab_halfplate_angles]
    # ydata = [float(y) for y in results.lab_halfplate_results]
    ydata = results.lab_halfplate_results
    vars = results.lab_halfplate_vars
    max_val = max(ydata)
    func = lambda x: max_val * np.power(np.sin(np.deg2rad(2 * x)), 2)
    if relative_error:
        y = np.array([func(val) for val in xdata])
        plt.title("error of halfplate results")
        plt.xlabel("angle(Degress)")
        plt.ylabel("error(Current)")
        relative_err = []
        for i, val in enumerate(y):
            if val == 0:
                relative_err.append(0)
            else:
                relative_err.append(ydata[i] - val)
                # relative_err.append(100 * np.abs(ydata[i] - val) / val)
        plt.scatter(xdata[:-1], relative_err[:-1], marker="o")
        plt.show()
        print(np.mean(sorted(relative_err)[:-1]))
    else:
        experiment_results(xdata, ydata, func, xlabel="Angle(degrees)", ylabel="Current(Amper)",
                           title="Electric current of Halfplate wave experiment\n as function of angle",
                           vars=np.array(vars))


def two_polarizer_experiment(relative_error=True):
    xdata = results.lab_two_polarizer_angles
    ydata = np.array(results.lab_two_polarizer_results)
    max_val = max(ydata)
    func = lambda x: max_val * np.power(np.cos(np.deg2rad(x)), 2)
    if relative_error:
        y = np.array([func(val) for val in xdata])
        plt.title("relative error of results")
        plt.xlabel("angle(Degress)")
        plt.ylabel("relative error(Amper)")
        plt.plot(xdata, 100 * np.abs(ydata - y) / y, marker="o")
        plt.show()
    else:
        experiment_results(xdata, ydata, func, xlabel="Angle(degrees)", ylabel="Current(Amper)",
                           title="Electric current of Two polarizers experiment as function of angle")


def three_polarizer_experiment():
    xdata = results.lab_three_polarizer_angles
    ydata = results.lab_three_polarizer_results
    max_val = max(ydata)
    func = lambda x: max_val * np.power(np.sin(2 * np.deg2rad(x)), 2)
    experiment_results(xdata, ydata, func, xlabel="Angle(degrees)", ylabel="Current(Amper)",
                       title="Electric current of Three polarizers experiment as function of angle")


def lab_brooster():
    xdata = results.lab_brooster_angles
    ydata = results.lab_brooster_results
    # max_val = max(ydata)
    plt.scatter(xdata, ydata)
    plt.xlabel("Angle(degrees)")
    plt.ylabel("Current(Amper)")
    plt.title("Electric current of brooster experiment\nas function of angle")
    plt.show()


def chiral_experiment():
    df = pd.DataFrame(results.results_chiral, columns=results.columns_chiral)
    for amount in df["sugar"].unique():
        df1 = df[df["sugar"] == amount]
        x = df1["length"]
        y = df1["power"]
        plt.plot(x, y, marker="o", label=f"{amount}")
        # x = [[i] for i in x]
        # reg = LinearRegression().fit(x, y)
        # score = reg.score(x, y)
        # plt.title(f"Current as function of length {float(amount)/600:.3f}(gr/ml) sugar\nlinearity score : {score}\ny={reg.coef_[0]}x+{reg.intercept_}")
        plt.legend()
        plt.show()


def experiment_results(xdata, ydata, func, xlabel="", ylabel="", title="", vars=None):
    # plt.plot(xdata, ydata, 'b-', label='data')
    # popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
    # plt.plot(xdata, func(xdata, *popt), 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    x = np.linspace(min(xdata), max(xdata), 1000)
    y = [func(val) for val in x]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x, y, label='expected')

    if vars is not None:
        plt.errorbar(xdata, ydata, yerr=vars / 2, ecolor='red', fmt='.k', color="black", label='data')
    else:
        diff = [np.abs(ydata[i] - func(val)) for i, val in enumerate(xdata)]
        plt.plot(xdata, ydata, marker="o", label='data')
        plt.plot(xdata, diff, label="error (|expected-data|)")
    plt.legend()
    plt.show()


def analyze_halfplate(
        path=r"C:\Users\t9028387\Documents\Yoav\Academy\2nd Year\Semester B\Physics Lab 2\Experiments\HalfPlate"):
    res = []
    vars = []
    for i in tqdm(range(250, 341, 3)):
        file_path = os.path.join(path, f"{i}.xlsx")
        df = pd.read_excel(file_path)
        vals = df[0][5:]
        res.append(np.mean(vals))
        vars.append(np.max(vals) - np.min(vals))
    print(res)
    print(vars)


if __name__ == "__main__":
    # analyze_halfplate()
    # halfplate_experiment(relative_error=True)
    lab_brooster()
    # two_polarizer_experiment()
    # three_polarizer_experiment()
    # chiral_experiment()
