import marimo

__generated_with = "0.1.63"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(np):
    def shallow_1_1_3(x, activation_fn, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31):

      #3 linear equations
      pre_1 = theta_10 + np.dot(x, theta_11)
      pre_2 = theta_20 + np.dot(x, theta_21)
      pre_3 = theta_30 + np.dot(x, theta_31)

      # ReLU for each one
      act_1 = activation_fn(pre_1)
      act_2 = activation_fn(pre_2)
      act_3 = activation_fn(pre_3)

      # Weight each non-linear out = w1*h1 + w2*h2 + w3*h3
      w_act_1 = np.dot(act_1, phi_1)
      w_act_2 = np.dot(act_2, phi_2)
      w_act_3 = np.dot(act_3, phi_3)

      # Sum all up the weighted activations
      y = phi_0 + w_act_1 + w_act_2 + w_act_3

      # Return everything we have calculated
      return y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3
    return shallow_1_1_3,


@app.cell
def __(plt):
    def plot_layers(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, plot_all=False, x_data=None, y_data=None):

      # Plot intermediate plots if flag set
      if plot_all:
        fig, ax = plt.subplots(3,3)
        fig.set_size_inches(8.5, 8.5)
        fig.tight_layout(pad=3.0)
        ax[0,0].plot(x,pre_1,'r-'); ax[0,0].set_ylabel('Preactivation')
        ax[0,1].plot(x,pre_2,'b-'); ax[0,1].set_ylabel('Preactivation')
        ax[0,2].plot(x,pre_3,'g-'); ax[0,2].set_ylabel('Preactivation')
        ax[1,0].plot(x,act_1,'r-'); ax[1,0].set_ylabel('Activation')
        ax[1,1].plot(x,act_2,'b-'); ax[1,1].set_ylabel('Activation')
        ax[1,2].plot(x,act_3,'g-'); ax[1,2].set_ylabel('Activation')
        ax[2,0].plot(x,w_act_1,'r-'); ax[2,0].set_ylabel('Weighted Act')
        ax[2,1].plot(x,w_act_2,'b-'); ax[2,1].set_ylabel('Weighted Act')
        ax[2,2].plot(x,w_act_3,'g-'); ax[2,2].set_ylabel('Weighted Act')

        for plot_y in range(3):
          for plot_x in range(3):
            ax[plot_y,plot_x].set_xlim([0,1]);ax[plot_x,plot_y].set_ylim([-1,1])
            ax[plot_y,plot_x].set_aspect(0.5)
          ax[2,plot_y].set_xlabel('Input, $x$');
        return plt.gca()
    return plot_layers,


@app.cell
def __(mo, plt):
    def plot_neural(x, y, x_data=None, y_data=None):


          fig, ax = plt.subplots()
          ax.plot(x,y)
          ax.set_xlabel('Input, $x$'); ax.set_ylabel('Output, $y$')
          ax.set_xlim([0,1]);ax.set_ylim([-1,1])
          ax.set_aspect(0.5)
          if x_data is not None:
            ax.plot(x_data, y_data, 'mo')
            for i in mo.status.progress_bar(range(len(x_data)), title="Plotting..."):
              ax.plot(x_data[i], y_data[i],)
          return plt.gca()
    return plot_neural,


@app.cell
def __(mo):
    with open("architecture.png", "rb") as file:
        mo.image(src=file)
    return file,


@app.cell
def __(mo):
    mo.md(r"""" 
    **Example Neural Network**

    $$\large f[x, (\theta_i, \phi_i)] = \phi_0 + \phi_1 \cdot relu[\theta_{10} + \theta_{11} x] + \phi_2 \cdot relu[\theta_{20} + \theta_{21} x] + \phi_3 \cdot relu[\theta_{30} + \theta_{31} x]$$
    """)
    return


@app.cell
def __(mo):
    theta_10 = mo.ui.slider(-2,2, step=0.1, value=0.3
                          ,label=r"$\large \theta_{10}$")
    theta_11 = mo.ui.slider(-2,2, step=0.1, value=-1.0
                           ,label=r"$\large \theta_{11}$")
    theta_20 = mo.ui.slider(-2,2, step=0.1, value=-1.0
                           ,label=r"$\large \theta_{20}$")
    theta_21 = mo.ui.slider(-2,2, step=0.1, value=2.0
                           ,label=r"$\large \theta_{21}$")
    theta_30 = mo.ui.slider(-2,2, step=0.1, value=-0.5
                           ,label=r"$\large \theta_{30}$")
    theta_31 = mo.ui.slider(-2,2, step=0.1, value=0.65
                           ,label=r"$\large \theta_{31}$")
    return theta_10, theta_11, theta_20, theta_21, theta_30, theta_31


@app.cell
def __(mo):
    phi_0 = mo.ui.slider(-2,2, step=0.1, value=0.3
                          ,label=r"$\large \phi_{0}$")
    phi_1 = mo.ui.slider(-2,2, step=0.1, value=2.0
                          ,label=r"$\large \phi_{1}$")
    phi_2 = mo.ui.slider(-2,2, step=0.1, value=-1.0
                          ,label=r"$\large \phi_{2}$")
    phi_3 = mo.ui.slider(-2,7, step=0.1, value=7.0
                          ,label=r"$\large \phi{3}$")
    return phi_0, phi_1, phi_2, phi_3


@app.cell
def __(
    mo,
    phi_0,
    phi_1,
    phi_2,
    phi_3,
    theta_10,
    theta_11,
    theta_20,
    theta_21,
    theta_30,
    theta_31,
):

    theta_params = mo.vstack([theta_10, theta_11, theta_20, theta_21, theta_30, theta_31])
    phi_params = mo.vstack([phi_0, phi_1, phi_2, phi_3])
    mo.hstack([theta_params, phi_params, mo.md(f"## Tweak these **parameters** of the neural network!")],
             justify="start")
    return phi_params, theta_params


@app.cell
def __():
    import numpy as np
    return np,


@app.cell
def __(np):
    def ReLU(preactivation):
      activation = np.maximum(0, preactivation)

      return activation
    return ReLU,


@app.cell
def __():
    import matplotlib.pyplot as plt
    return plt,


@app.cell
def __(
    ReLU,
    mo,
    np,
    phi_0,
    phi_1,
    phi_2,
    phi_3,
    plot_layers,
    plot_neural,
    shallow_1_1_3,
    theta_10,
    theta_11,
    theta_20,
    theta_21,
    theta_30,
    theta_31,
):
    # Define a range of input values
    x = np.arange(0,1,0.01)

    # We run the neural network for each of these input values
    y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3 = \
        shallow_1_1_3(x, ReLU, phi_0.value, phi_1.value, phi_2.value, phi_3.value, theta_10.value, theta_11.value, theta_20.value, theta_21.value, theta_30.value, theta_31.value)
    # And then plot it
    mo.hstack([plot_layers(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, plot_all=True), plot_neural(x,y), mo.md("**Output**")], justify="start")

    return (
        act_1,
        act_2,
        act_3,
        pre_1,
        pre_2,
        pre_3,
        w_act_1,
        w_act_2,
        w_act_3,
        x,
        y,
    )


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
