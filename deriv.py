""" Python module to compute numerical derivatives of python code

    Goal: To mimic all the functionality that is provided to numpy array
    objects so that we can differentiate any python code that uses numpy
    arrays.
"""

import numpy

# custom
from mathfuncs import *

class UTPS:
    """ Store the Taylor series coefficients and perform arithmetic on it """

    def __init__(self, coeffs, dtype):
        """ Create an instance using the given coefficients and type """
        self.coeffs = numpy.array(coeffs, dtype)

    def __len__ (self):
        return len(self.coeffs)

    def __getitem__ (self, index):
        assert(0 <= index < len(self))
        return self.coeffs[index]

    def __str__ (self):
        """ Display the Taylor coefficients """
        return str(self.coeffs)

    def __neg__ (self):
        return self.__class__(-self.coeffs, self.coeffs.dtype)

    def __add__ (self, other):
        if isinstance(other, UTPS):
            assert(len(self) == len(other))
            return  self.__class__(self.coeffs + other.coeffs, \
                    self.coeffs.dtype)
        ret = self.__class__(numpy.copy(self.coeffs), self.coeffs.dtype)
        ret.coeffs[0] += other
        return ret

    def __sub__ (self, other):
        if isinstance(other, UTPS):
            assert(len(self) == len(other))
            return  self.__class__(self.coeffs - other.coeffs, \
                    self.coeffs.dtype)
        ret = self.__class__(numpy.copy(self.coeffs), self.coeffs.dtype)
        ret.coeffs[0] -= other
        return ret

    def __mul__ (self, other):
        if isinstance(other, UTPS):
            assert(len(self) == len(other))
            return  self.__class__(convolve(self.coeffs, other.coeffs), \
                    self.coeffs.dtype)
        return self.__class__(self.coeffs * other, self.coeffs.dtype)

    def __div__ (self, other):
        if isinstance(other, UTPS):
            assert(len(self) == len(other))
            return  self.__class__(deconvolve(self.coeffs, other.coeffs), \
                    self.coeffs.dtype)
        return self.__class__(self.coeffs / other, self.coeffs.dtype)

    def __radd__ (self, other):
        return self + other

    def __rsub__ (self, other):
        return -self + other

    def __rmul__ (self, other):
        return self * other

    def __rdiv__ (self, other):
        left = numpy.zeros(len(self), dtype=self.coeffs.dtype)
        left[0] = other
        return self.__class__(left) / self

class Deriv(UTPS):
    """ Wrapper around UTPS for computing derivatives """

    def __init__(self, coeffs, dtype):
        UTPS.__init__(self, coeffs, dtype)

    @classmethod
    def Variable (cls, x0, length, dtype='float64'):
        """ Represent a variable """
        if  length <= 0:
            raise ValueError, ("Length >= 1")
        else:
            coeffs = numpy.zeros(length, dtype)
            coeffs[0] = x0
            coeffs[1] = 1
        return cls(coeffs, dtype)

    def __setitem__ (self, index, value):
        raise NotImplementedError

    def __getitem__ (self, index):
        assert(0 <= index < len(self))
        return self.coeffs[index] * reduce(lambda x,y: x*y, \
                range(1, index + 1), numpy.array(1, dtype=self.coeffs.dtype))

    def __repr__ (self):
        """ Scale the Taylor coefficients to display derivatives """
        d = numpy.zeros(len(self.coeffs), dtype=self.coeffs.dtype)
        d[0] = self.coeffs[0]
        fac = 1
        for i in range(1, len(self.coeffs)):
            d[i] = fac * self.coeffs[i]
            fac *= i + 1
        return str(d)

    def __str__ (self):
        return 'Deriv(%s, Type=%s)' % (repr(self), self.coeffs.dtype)

if __name__ == "__main__":

    x0 = complex(1, 1) # 1 + j1
    ord = 2

    print 'Computing using AD:'
    x = Deriv.Variable(x0, ord + 1, 'complex64')
    f = asin(x) * acosh(x)/(sin(x) - exp(x)) + x*x*tan(x)/(3 - log(x)+acosh(x))
    print f

    print
    print 'Computing using sympy:'
    from sympy import *
    x = symbols('x')
    f = asin(x) * acosh(x)/(sin(x) - exp(x)) + x*x*tan(x)/(3 - log(x)+acosh(x))
    for i in range(ord + 1):
        print f.diff(x, i).subs(x, x0).n()
