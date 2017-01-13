# -*- coding: utf-8 -*-
"""Definition and implementation of linear measurement objects.

This module specifies the interface for objects representing linear measurement
operators and implements several examples.

All measurements derive from a base class `Measurement`. We have the following
implementations:

    * `InnerProductMeasurement`: The inner product with a fixed operator.
    * `DirectActionMeasurement`: Applying the operator on a fixed matrix (viewed
        as a linear action).
    * `IndirectActionMeasurement`: Applying the operator on a fixed matrix and
        then applying the result on a fixed vector (viewed as a linear action).
    * `SubsampleMeasurement`: Retrieving a fixed set of indices from an operator.
    * `IdentityMeasurement`: Returns the operator itself.
    * `CombinedMeasurements`: Returns the result of any number of `Measurement`
        objects.

"""

from . import operators
import numpy as np
import cvxpy
import itertools


class Measurement(object):
    """Abstract class for measurements.

    This class has no implementation but provides the base for all linear measurement objects.


    Attributes
    ----------
    shape : tuple
        The shape of operators the map accepts as input.
    nmeas : int
        The number of measurements in the output of the map.
    apply : function
        Apply the measurement map to a `DyadsOperator` and/or an `ArrayOperator`.
    cvxapply : function
        Apply the measurement map to CVXPY operators.
    asOperator : function
        Return the measurement map as a list of operators.

        Each entry of the output of `apply` is the inner product between these operators and the input to `apply`.
    initfrommeas : function
        Applies the adjoint of the measurement map.

        This functionality is used by default for the initialization of `altminsolve`.

    """
    pass


class InnerProductMeasurement(Measurement):
    """The inner product with a fixed operator."""
    def __init__(self, fixed_oper):
        self.fixed_oper = fixed_oper
        self.shape = fixed_oper.shape
        self.nmeas = 1

    def apply(self, oper):
        return operators.innerprod(self.fixed_oper, oper)

    def cvxapply(self, oper):
        return operators.cvxinnerprod(self.fixed_oper, oper)

    def asOperator(self):
        return self.fixed_oper

    def initfromoper(self, oper):
        return self.asOperator()*self.apply(oper)

    def initfrommeas(self, meas):
        return self.asOperator()*meas


class DirectActionMeasurement(Measurement):
    r"""Applies the operator on a fixed matrix.

    .. math::
        \mu\bigg( \sum_i \mathbf{X}_i \otimes \mathbf{Y}_i \bigg) = \sum_i \mathbf{X}_i \mathbf{M} \mathbf{Y}_i^T

    """
    def __init__(self, fixed_mat, oper_shape):
        self.fixed_mat = fixed_mat
        self.shape = oper_shape
        self.nmeas = oper_shape[0]*oper_shape[2]
        self._measmat = None

    def apply(self, oper):
        return oper.apply(self.fixed_mat).flatten(order='F')

    def cvxapply(self, oper):
        return cvxpy.vec(oper.cvxapply(np.mat(self.fixed_mat)))

    def matapply(self, mat, compute_measmat=False):
        """Apply to oper A \otimes B given as vec(ab^T), all vecs F-order"""
        if self._measmat is not None:
            if isinstance(mat, cvxpy.expressions.expression.Expression):
                out = self._measmat * cvxpy.vec(mat)
            else:
                out = self._measmat @ mat.flatten(order='F')
        else:
            # FIXME: will fail if cvxpy expression passed
            out = np.zeros((self.nmeas, 1))
            if compute_measmat:
                self._measmat = np.zeros((self.nmeas, np.prod(self.shape)))
            M, N, P, Q = self.shape
            for i in range(M):
                for k in range(P):
                    measvec = np.zeros((np.prod(self.shape),))
                    for j in range(N):
                        for l in range(Q):
                            measvec[M*N*(P*l + k) + M*j + i] = self.fixed_mat[j, l]
                    out[i + M*k] = np.dot(measvec, mat.flatten(order='F'))
                    if compute_measmat:
                        self._measmat[i+M*k, :] = measvec
        if compute_measmat:
            return (out, self._measmat)
        else:
            return out

    def asOperator(self):
        out = []
        sz = self.shape
        for i in range(sz[0]):
            for j in range(sz[2]):
                tempoper = operators.ArrayOperator(np.zeros(sz))
                for I in range(sz[1]):
                    for J in range(sz[3]):
                        tempoper[i, I, j, J] = self.fixed_mat[I, J]
                out.append(tempoper)
        return out

    def _innerinit(self, applymat):
        out = operators.ArrayOperator(np.zeros(self.shape))
        sz = self.shape
        for i in range(sz[0]):
            for I in range(sz[1]):
                for j in range(sz[2]):
                    for J in range(sz[3]):
                        out[i, I, j, J] = self.fixed_mat[I, J] * applymat[i, j]
        return out

    def initfromoper(self, oper):
        return self._innerinit(oper.apply(self.fixed_mat))

    def initfrommeas(self, meas):
        return self._innerinit(meas.reshape(self.shape[0], self.shape[2], order='F'))


class IndirectActionMeasurement(Measurement):
    """Applies the operator on a fixed matrix and the result on a fixed vector."""
    def __init__(self, fixed_mat, fixed_vec, oper_shape):
        self.fixed_mat = fixed_mat
        self.fixed_vec = fixed_vec
        self.shape = oper_shape
        self.nmeas = fixed_vec.size

    def apply(self, oper):
        return oper.apply(self.fixed_mat).dot(self.fixed_vec)

    def asOperator(self):
        raise NotImplementedError

    def initfromoper(self, oper):
        raise NotImplementedError

    def initfrommeas(self, meas):
        raise NotImplementedError


class SubsampleMeasurement(Measurement):
    """Retrieves a fixed set of indices from the operator."""
    pass


class IdentityMeasurement(Measurement):
    """Returns the operator itself."""
    def __init__(self, oper_shape):
        self.shape = oper_shape
        self.nmeas = np.prod(oper_shape)

    def apply(self, oper):
        if isinstance(oper, operators.DyadsOperator):
            return sum([np.outer(oper.lfactors[r].flatten(order='F'),
                                 oper.rfactors[r].flatten(order='F'))
                        for r in range(oper.nfactors)]).flatten(order='F')
        else:
            return oper.asArrayOperator().flatten(order='F')

    def cvxapply(self, oper):
        return cvxpy.vec(sum([cvxpy.vec(oper.lfactors[r]) *
                              cvxpy.vec(oper.rfactors[r]).T
                              for r in range(oper.nfactors)]))

    def matapply(self, mat):
        if isinstance(mat, cvxpy.expressions.expression.Expression):
            return cvxpy.vec(mat)
        else:
            # being explicit that we want a column vec
            # NB: (cvxpy.vec().value is a column vec)
            return mat.reshape((np.prod(mat.shape), 1), order='F')

    def asOperator(self):
        raise NotImplementedError

    def initfromoper(self, oper):
        # does not care about order vis a vis apply
        return oper.asArrayOperator()

    def initfrommeas(self, meas, from_dyads=True):
        # from_dyads used to make order match from apply
        if from_dyads:
            sz = self.shape
            out = operators.ArrayOperator(np.zeros(sz))
            for i, j, k, l in itertools.product(*[range(sz[n])
                                                  for n in range(4)]):
                # for sz = (m,n,p,q) the i,j,k,l entry is located at:
                # i+(j*m)+(k*mn)+(l*mnp)
                out[i, j, k, l] = meas[i+(j*sz[0])+(k*sz[0]*sz[1])+(l*sz[0]*sz[1]*sz[2])]
            return out
        else:
            return operators.ArrayOperator(meas.reshape(self.shape, order='F'))


class CombinedMeasurements(Measurement, list):
    """Returns the results from any number of measurements.

    Creation takes a list of other `Measurement` objects.

    """
    def __init__(self, *args):
        list.__init__(self, *args)

    def apply(self, oper):
        return np.concatenate((m.apply(oper) for m in self))

    def cvxapply(self, oper):
        return cvxpy.vec([m.cvxapply(oper) for m in self])

    @property
    def shape(self):
        self[0].shape

    @property
    def nmeas(self):
        return sum((m.nmeas for m in self))

    def initfromoper(self, oper):
        return sum((m.initfromoper(oper) for m in self))

    def initfrommeas(self, meas):
        currix = 0
        out = operators.ArrayOperator(np.zeros(self.shape))
        for m in self:
            out += m.initfrommeas(meas[currix:(currix + m.nmeas)])
            currix += currix + m.nmeas
        return out
