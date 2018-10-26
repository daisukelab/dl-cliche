# Thanks to https://stackoverflow.com/questions/32906669/how-to-use-ipython-magic-within-a-script-to-auto-reload-modules
from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('reload_ext autoreload')
    ipython.magic('autoreload 2')
    ipython.magic('matplotlib inline')
