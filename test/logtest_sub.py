"""test_log sub module.
"""
from dlcliche.utils import *

def logtest_sub(utcls, name):
    log = get_logger(name)
    log.info('other module logging info')
    log.debug('other module logging debug')
    log.error('other module logging error')


