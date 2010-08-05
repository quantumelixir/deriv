""" Utilities for solving simple linear differential equations
    Example: u' = va' where v and a are known i.e. Taylor coefficients
    and we are to find the Taylor coefficients of u.
"""

import numpy

def convolve(a, b):
    """ Solve c = a * b """
    c = numpy.zeros(len(a), dtype=a.dtype)
    for k in range(len(a)):
        c[k] = sum(a[i]*b[k-i] for i in range(k + 1))
    return c

def deconvolve(a, b):
    """ Solve a = c * b i.e. c = a / b """
    c = numpy.zeros(len(a), dtype=a.dtype)
    for k in range(len(a)):
        c[k] = (a[k] - sum(c[i]*b[k-i] for i in range(k))) / b[0];
    return c

def ueqva(u, a, v):
    """ Solve u' = va' """
    for k in range(1, len(u)):
        u[k] = sum(i*a[i]*v[k-i] for i in range(1, k + 1)) / k;

def aeqvu(u, a, v):
    """ Solve a' = vu' """
    for k in range(1, len(u)):
        u[k] = (k*a[k] - sum(i*u[i]*v[k-i] for i in range(1, k)))/ k / v[0];

def multi(t1, u1, alpha1, a1, v1, t2, u2, alpha2, a2, v2):
    """ Solve a system of two linear differential equations """
    if t1 == 'ueqva' and t2 == 'ueqva':
        for k in range(1, len(u1)):
            u1[k] = alpha1 * sum(i*a1[i]*v1[k-i] for i in range(1, k + 1)) / k;
            u2[k] = alpha2 * sum(i*a2[i]*v2[k-i] for i in range(1, k + 1)) / k;
    elif t1 == 'ueqva' and t2 == 'aqevu':
        for k in range(1, len(u1)):
            u1[k] = alpha1 * sum(i*a1[i]*v1[k-i] for i in range(1, k + 1)) / k;
            u2[k] = 1. * (k*a2[k] - sum(i*u2[i]*v2[k-i] for i in range(1, k))) \
                    / k / v2[0];
    elif t1 == 'aeqvu' and t2 == 'ueqva':
        for k in range(1, len(u1)):
            u1[k] = 1. * (k*a1[k] - sum(i*u1[i]*v1[k-i] for i in range(1, k))) \
                    / k / v1[0];
            u2[k] = alpha2 * sum(i*a2[i]*v2[k-i] for i in range(1, k + 1)) / k;
    elif t1 == 'aeqvu' and t2 == 'aqevu':
        for k in range(1, len(u1)):
            u1[k] = 1. * (k*a1[k] - sum(i*u1[i]*v1[k-i] for i in range(1, k))) \
                    / k / v1[0];
            u2[k] = 1. * (k*a2[k] - sum(i*u2[i]*v2[k-i] for i in range(1, k))) \
                    / k / v2[0];
