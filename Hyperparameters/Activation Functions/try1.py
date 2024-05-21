import marimo

__generated_with = "0.1.63"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return mo,


@app.cell
def _():
    #marimo convert your_notebook.ipynb > your_notebook.py
    return


app._unparsable_cell(
    r"""
    !marimo convert script.ipynb > your_notebook.py
    """,
    name="_"
)


@app.cell
def _(mo):
    text_input = mo.ui.text()
    mo.md("Enter intpu")
    return text_input,


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

