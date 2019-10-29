import IPython

# Thanks to https://github.com/ipython/ipython/issues/9732
from IPython import get_ipython
ipython = get_ipython()

# Determine if this is running in Jupyter notebook or not
if ipython:
    running_in_notebook = ipython.has_trait('kernel')

    if running_in_notebook:
        ipython.magic('reload_ext autoreload')
        ipython.magic('autoreload 2')
        ipython.magic('matplotlib inline')
else:
    # cannot even get ipython object...
    running_in_notebook = False


def fit_notebook_to_window():
    """Fit notebook width to width of browser window.
    Thanks to https://stackoverflow.com/questions/21971449/how-do-i-increase-the-cell-width-of-the-jupyter-ipython-notebook-in-my-browser
    """
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))


# Didn't work -> https://stackoverflow.com/questions/32906669/how-to-use-ipython-magic-within-a-script-to-auto-reload-modules
