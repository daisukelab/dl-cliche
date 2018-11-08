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

# Didn't work -> https://stackoverflow.com/questions/32906669/how-to-use-ipython-magic-within-a-script-to-auto-reload-modules
