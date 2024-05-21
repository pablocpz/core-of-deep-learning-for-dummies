import marimo

__generated_with = "0.1.63"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    import numpy as np
    return np,


@app.cell
def __(np):
    X = np.linspace(-10,10).reshape(-1,1)

    import matplotlib.pyplot as plt

    def linear(x):

        return np.random.rand(1,1) + np.dot(x,
                                           np.random.rand(x.shape[1], 1)) 

    def nonlinear(x):

        return np.maximum(0,x)

    l1, l2, l3 = linear(X), linear(X), linear(X)
    return X, l1, l2, l3, linear, nonlinear, plt


@app.cell
def __(l1, l2, l3, nonlinear):
    h1, h2, h3 = nonlinear(l1), nonlinear(l2), nonlinear(l3)
    return h1, h2, h3


@app.cell
def __(mo):
    w1, w2, w3, b = mo.ui.slider(-10.0, 10.0, 1.0,
                                label=r"$\omega_1$"), mo.ui.slider(-10.0, 10.0, 1.0, label=r"$\omega_2$"
                                                                  ), mo.ui.slider(-10.0, 10.0, 1.0,
                                                                                 label=r"$\omega_3$"), mo.ui.slider(-10.0, 10.0, 1.0,
                                                                                                                 label="b")
    return b, w1, w2, w3


@app.cell
def __(mo):
    mo.md("## Neural Network with 3 hidden layers **visualization**")
    return


@app.cell
def __(mo):
    mo.md(r"""
    $$ \large{y = \omega_0 + \omega_1[a(\theta_0 \cdot x + \theta_1 \cdot x)] + \omega_2[a(\theta_2 \cdot x + \theta_3 \cdot x)] + \omega_3[a(\theta_4 \cdot x + \theta_5 \cdot x)]} $$
    """)
    return


@app.cell
def __(mo):
    mo.md(r"($\theta_i$ will be fixed parameters for visualization purposes)")
    return


@app.cell
def __(mo):
    mo.image(
        src="https://i.postimg.cc/65nhMrRj/nn.png",
        alt="NN Depicted",
        width=300,
        height=250,
        rounded=True,
    )
    return


@app.cell
def __(b, w1, w2, w3):
    w1.center(),w2.center(),w3.center(),b.center()
    return


@app.cell
def __(X, b, h2, h3, l1, plt, w1, w2, w3):
    plt.plot(X, w1.value*l1 +w2.value*h2 + w3.value*h3 + b.value)
    return


@app.cell
def __(mo):
    mo.md("We certainly see how combining linear operations of $x$ and applying non-linear functions to them, and finnally applying another linear operation on the top result in complex non-linear patterns")
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
