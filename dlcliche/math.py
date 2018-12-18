import math

def roundup(x, n=10):
    """Round up x to multiple of n."""
    return int(math.ceil(x / n)) * n
