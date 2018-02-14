# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:19:44 2017

@author: deepak
"""
#http://mg.pov.lt/profilehooks/ is an alternative package for the same


import time                                                


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.2f sec' % \
              (method.__name__, te-ts))
        return result

    return timed
'''
def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts)
        return result

    return timed
'''