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
def __():
    import matplotlib.pyplot as plt
    return plt,


@app.cell
def __(mo):
    fx = mo.ui.slider(-5,5, step=0.5, value= 1, label="X-component")
    fy = mo.ui.slider(-5,5, step=0.5, value=2, label="y-component")
    mo.md(
        f"""
        **Visualizing vectors.**
        {fx}
        {fy}
            """
    )
    return fx, fy


@app.cell
def __(fx, fy, mo, plt):
    mo.md(f"$${fx.value}, {fy.value}$$")
    plt.quiver(0,0,fx.value, fy.value,angles="xy", scale_units="xy", scale=1, color="red")
    # plt.quiver(0,0,v2[0],v2[1],angles="xy", scale_units="xy", scale=1, color="blue")
    # plt.quiver(0,0,vr[0],vr[1],angles="xy", scale_units="xy", scale=1, color="orange")
    plt.grid()
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.gca()
    return


@app.cell
def __(fx, fy, mo, np):
    mo.md(rf"""

        $$\huge \vec R = ({fx.value}) \cdot \vec i + ({fy.value}) \cdot \vec j$$

        
        
        | Fuerza | Componente x | Componente y | Polar |
    |---|---|---|---|
    |$\vec R$|${fx.value}$|${fy.value}$|$\vec R= {np.round(np.sqrt((fx.value)**2 + (fy.value)**2), 2)}\angle {np.round(np.arctan2(fy.value, fx.value), 2)}º$|

        """)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
