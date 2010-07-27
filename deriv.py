""" Deriv -- python module to compute derivatives numerically

    TODO: Mirror the functionality in numpy modules
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
            u2[k] = 1. * (k*a2[k] - sum(i*u2[i]*v2[k-i] for i in range(1, k))) / k / v2[0];
    elif t1 == 'aeqvu' and t2 == 'ueqva':
        for k in range(1, len(u1)):
            u1[k] = 1. * (k*a1[k] - sum(i*u1[i]*v1[k-i] for i in range(1, k))) / k / v1[0];
            u2[k] = alpha2 * sum(i*a2[i]*v2[k-i] for i in range(1, k + 1)) / k;
    elif t1 == 'aeqvu' and t2 == 'aqevu':
        for k in range(1, len(u1)):
            u1[k] = 1. * (k*a1[k] - sum(i*u1[i]*v1[k-i] for i in range(1, k))) / k / v1[0];
            u2[k] = 1. * (k*a2[k] - sum(i*u2[i]*v2[k-i] for i in range(1, k))) / k / v2[0];

def sincos(a):
    # u' = va'
    # v' = -ua'
    u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.sin(a[0]), numpy.cos(a[0])
    multi('ueqva', u, 1.0, a, v, 'ueqva', v, -1.0, a, u)
    return [u, v]

def sincosh(a):
    # u' = va'
    # v' = ua'
    u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
    u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
    u[0], v[0] = numpy.sinh(a[0]), numpy.cosh(a[0])
    multi('ueqva', u, 1.0, a, v, 'ueqva', v, 1.0, a, u)
    return [u, v]

class Deriv:
    """ Stores the Taylor series coefficients of an expression """

    def __init__(self, coeffs, dtype='float64'):
        """ Create an instance using the coefficients """
        self.coeffs = numpy.array(coeffs, dtype)

    @classmethod
    def Variable (cls, x0, ord, dtype='float64'):
        """ Represent a variable """
        if  ord <= 0:
            raise ValueError, ("Order >= 1")
        else:
            coeffs = numpy.zeros(ord + 1, dtype)
            coeffs[0] = x0
            coeffs[1] = 1
        return cls(coeffs, dtype)

    @classmethod
    def Constant (cls, x0, ord, dtype='float64'):
        """ Represent a constant """
        if  ord <= 0:
            raise ValueError, ("Order >= 1")
        else:
            coeffs = numpy.zeros(ord + 1, dtype)
            coeffs[0] = x0
        return cls(coeffs, dtype)

    def __len__ (self):
        return len(self.coeffs)

    def __getitem__ (self, index):
        assert(0 <= index < len(self))
        return self.coeffs[index]

    def __neg__ (self):
        return Deriv(-self.coeffs, self.coeffs.dtype)

    def __add__ (self, other):
        if isinstance(other, Deriv):
            assert(len(self) == len(other))
            return  Deriv(self.coeffs + other.coeffs, self.coeffs.dtype)
        ret = Deriv(numpy.copy(self.coeffs), self.coeffs.dtype)
        ret.coeffs[0] += other
        return ret

    def __sub__ (self, other):
        if isinstance(other, Deriv):
            assert(len(self) == len(other))
            return  Deriv(self.coeffs - other.coeffs, self.coeffs.dtype)
        ret = Deriv(numpy.copy(self.coeffs), self.coeffs.dtype)
        ret.coeffs[0] -= other
        return ret

    def __mul__ (self, other):
        if isinstance(other, Deriv):
            assert(len(self) == len(other))
            return  Deriv(convolve(self.coeffs, other.coeffs), self.coeffs.dtype)
        return Deriv(self.coeffs * other, self.coeffs.dtype)

    def __div__ (self, other):
        if isinstance(other, Deriv):
            assert(len(self) == len(other))
            return  Deriv(deconvolve(self.coeffs, other.coeffs), self.coeffs.dtype)
        return Deriv(self.coeffs / other, self.coeffs.dtype)

    def __radd__ (self, other):
        return self + other

    def __rsub__ (self, other):
        return -self + other

    def __rmul__ (self, other):
        return self * other

    def __rdiv__ (self, other):
        return Deriv.Constant(other, len(self) - 1, self.coeffs.dtype) / self

    def __str__ (self):
        """ Scale the Taylor coefficients to display derivatives """
        d = numpy.zeros(len(self.coeffs))
        d[0] = self.coeffs[0]
        fac = 1
        for i in range(1, len(self.coeffs)):
            d[i] = fac * self.coeffs[i]
            fac *= i + 1
        return str(d)

    def exp(self):
        # u' = ua'
        a = self.coeffs
        u = numpy.zeros(len(a), dtype=a.dtype)
        u[0] = numpy.exp(a[0])
        ueqva(u, a, u)
        return Deriv(u, self.coeffs.dtype)

    def log(self):
        # a' = au'
        a = self.coeffs
        u = numpy.zeros(len(a), dtype=a.dtype)
        u[0] = numpy.log(a[0])
        aeqvu(u, a, a)
        return Deriv(u, self.coeffs.dtype)

    def sin(self):
        a = self.coeffs
        [s, c] = sincos(a)
        return Deriv(s, self.coeffs.dtype)

    def cos(self):
        a = self.coeffs
        [s, c] = sincos(a)
        return Deriv(c, self.coeffs.dtype)

    def tan(self):
        # u' = va'
        # v' = 2uu'
        a = self.coeffs
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u[0], v[0] = numpy.tan(a[0]), (1 + numpy.tan(a[0])**2)
        multi('ueqva', u, 1.0, a, v, 'ueqva', v, 2.0, u, u)
        return Deriv(u, self.coeffs.dtype)

    def asin(self):
        # a' = vu'
        # v' = -au'
        a = self.coeffs
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u[0], v[0] = numpy.arcsin(a[0]), numpy.sqrt(1 - a[0]**2)
        multi('aeqvu', u, 1.0, a, v, 'ueqva', v, -1.0, u, a)
        return Deriv(u, self.coeffs.dtype)

    def acos(self):
        # a' = vu'
        # v' = -au'
        a = self.coeffs
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u[0], v[0] = numpy.arccos(a[0]), -numpy.sqrt(1 - a[0]**2)
        multi('aeqvu', u, 1.0, a, v, 'ueqva', v, -1.0, u, a)
        return Deriv(u, self.coeffs.dtype)

    def atan(self):
        # a' = vu'
        # v' = 2aa'
        a = self.coeffs
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u[0], v[0] = numpy.arctan(a[0]), (1 + a[0]**2)
        multi('aeqvu', u, 1.0, a, v, 'ueqva', v, 2.0, a, a)
        return Deriv(u, self.coeffs.dtype)

    def sinh(self):
        a = self.coeffs
        [s, c] = sincosh(a)
        return Deriv(s, self.coeffs.dtype)

    def cosh(self):
        a = self.coeffs
        [s, c] = sincosh(a)
        return Deriv(c, self.coeffs.dtype)

    def tanh(self):
        # u' = va'
        # v' = -2uu'
        a = self.coeffs
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u[0], v[0] = numpy.tanh(a[0]), (1 - numpy.tanh(a[0])**2)
        multi('ueqva', u, 1.0, a, v, 'ueqva', v, -2.0, u, u)
        return Deriv(u, self.coeffs.dtype)

    def asinh(self):
        # a' = vu'
        # v' = au'
        a = self.coeffs
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u[0], v[0] = numpy.arcsinh(a[0]), numpy.sqrt(1 + a[0]**2)
        multi('aeqvu', u, 1.0, a, v, 'ueqva', v, 1.0, u, a)
        return Deriv(u, self.coeffs.dtype)

    def acosh(self):
        # a' = vu'
        # v' = au'
        a = self.coeffs
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u[0], v[0] = numpy.arccosh(a[0]), numpy.sqrt(a[0]**2 - 1)
        multi('aeqvu', u, 1.0, a, v, 'ueqva', v, 1.0, u, a)
        return Deriv(u, self.coeffs.dtype)

    def atanh(self):
        # a' = vu'
        # v' = -2aa'
        a = self.coeffs
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u, v = numpy.zeros(len(a), dtype=a.dtype), numpy.zeros(len(a), dtype=a.dtype)
        u[0], v[0] = numpy.arctanh(a[0]), (1 - a[0]**2)
        multi('aeqvu', u, 1.0, a, v, 'ueqva', v, -2.0, a, a)
        return Deriv(u, self.coeffs.dtype)

if __name__ == "__main__":

    x0 = complex(1, 1) # 1 + j1
    ord = 3

    print 'Computing using AD:'
    x = Deriv.Variable(x0, ord, 'complex64')
    f = x.asin() * x.acosh() /(x.sin() - x.exp()) + x*x*x.tan()/(3 - x.log() + x.acosh())
    for i in range(ord + 1):
        print f[i] * reduce(lambda x,y: x*y, range(1, i + 1), 1)

    print
    print 'Computing using sympy:'
    from sympy import *
    x = symbols('x')
    f = asin(x) * acosh(x) / (sin(x) - exp(x)) + x*x*tan(x)/(3 - log(x) + acosh(x))
    for i in range(ord + 1):
        print f.diff(x, i).subs(x, x0).n()
