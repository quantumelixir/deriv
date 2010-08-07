""" Math functions for handling objects of the UTPS class and its derived classes
"""

import numpy
from numpy import zeros
from utilities import *

def exp(x):
    # u' = ua'
    a = x.coeffs
    u = zeros(len(a), dtype=a.dtype)
    u[0] = numpy.exp(a[0])
    ueqva(u, a, u)
    return x.__class__(u, x.coeffs.dtype)

def log(x):
    # a' = au'
    a = x.coeffs
    u = zeros(len(a), dtype=a.dtype)
    u[0] = numpy.log(a[0])
    aeqvu(u, a, a)
    return x.__class__(u, x.coeffs.dtype)

def log10(x):
    # a' = log(10) * au'
    a = x.coeffs
    u = zeros(len(a), dtype=a.dtype)
    u[0] = numpy.log(a[0])
    aeqvu(u, a, a)
    u /= numpy.log(10)
    return x.__class__(u, x.coeffs.dtype)

def sin(x):
    a = x.coeffs
    [s, c] = _sincos(a)
    return x.__class__(s, x.coeffs.dtype)

def cos(x):
    a = x.coeffs
    [s, c] = _sincos(a)
    return x.__class__(c, x.coeffs.dtype)

def tan(x):
    # u' = va'
    # v' = 2uu'
    a = x.coeffs
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.tan(a[0]), (1 + numpy.tan(a[0])**2)
    multi('ueqva', u, 1.0, a, v, 'ueqva', v, 2.0, u, u)
    return x.__class__(u, x.coeffs.dtype)

def asin(x):
    # a' = vu'
    # v' = -au'
    a = x.coeffs
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.arcsin(a[0]), numpy.sqrt(1 - a[0]**2)
    multi('aeqvu', u, 1.0, a, v, 'ueqva', v, -1.0, u, a)
    return x.__class__(u, x.coeffs.dtype)

def acos(x):
    # a' = vu'
    # v' = -au'
    a = x.coeffs
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.arccos(a[0]), -numpy.sqrt(1 - a[0]**2)
    multi('aeqvu', u, 1.0, a, v, 'ueqva', v, -1.0, u, a)
    return x.__class__(u, x.coeffs.dtype)

def atan(x):
    # a' = vu'
    # v' = 2aa'
    a = x.coeffs
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.arctan(a[0]), (1 + a[0]**2)
    multi('aeqvu', u, 1.0, a, v, 'ueqva', v, 2.0, a, a)
    return x.__class__(u, x.coeffs.dtype)

def sinh(x):
    a = x.coeffs
    [s, c] = _sincosh(a)
    return x.__class__(s, x.coeffs.dtype)

def cosh(x):
    a = x.coeffs
    [s, c] = _sincosh(a)
    return x.__class__(c, x.coeffs.dtype)

def tanh(x):
    # u' = va'
    # v' = -2uu'
    a = x.coeffs
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.tanh(a[0]), (1 - numpy.tanh(a[0])**2)
    multi('ueqva', u, 1.0, a, v, 'ueqva', v, -2.0, u, u)
    return x.__class__(u, x.coeffs.dtype)

def asinh(x):
    # a' = vu'
    # v' = au'
    a = x.coeffs
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.arcsinh(a[0]), numpy.sqrt(1 + a[0]**2)
    multi('aeqvu', u, 1.0, a, v, 'ueqva', v, 1.0, u, a)
    return x.__class__(u, x.coeffs.dtype)

def acosh(x):
    # a' = vu'
    # v' = au'
    a = x.coeffs
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.arccosh(a[0]), numpy.sqrt(a[0]**2 - 1)
    multi('aeqvu', u, 1.0, a, v, 'ueqva', v, 1.0, u, a)
    return x.__class__(u, x.coeffs.dtype)

def atanh(x):
    # a' = vu'
    # v' = -2aa'
    a = x.coeffs
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.arctanh(a[0]), (1 - a[0]**2)
    multi('aeqvu', u, 1.0, a, v, 'ueqva', v, -2.0, a, a)
    return x.__class__(u, x.coeffs.dtype)

def _sincos(a):
    # u' = va'
    # v' = -ua'
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.sin(a[0]), numpy.cos(a[0])
    multi('ueqva', u, 1.0, a, v, 'ueqva', v, -1.0, a, u)
    return [u, v]

def _sincosh(a):
    # u' = va'
    # v' = ua'
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u, v = zeros(len(a), dtype=a.dtype), zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.sinh(a[0]), numpy.cosh(a[0])
    multi('ueqva', u, 1.0, a, v, 'ueqva', v, 1.0, a, u)
    return [u, v]
