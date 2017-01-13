# -*- coding: utf-8 -*-
"""Definition and implementation of operator objects.

This module specifies the interface for operator objects and implements the two
common operators we use throughout the rest of the package.

Notes
-----
    Currently there is no abstract operator object from which to inherit. this
    is partially due to the fact that the ArrayOperator currently subclasses
    numpy.ndarray. In lieu of a proper abstract base class we briefly describe
    the attributes and methods expected of an operator.

    Attributes::

        shape : tuple of ints
            The shape of the operator as if it were an array.

    Methods::

        __add__
            Sums two operators.
        __mul__
            Multiplies the operator by a scalar.
        apply
            Applies the operator as a bilinear map to a matrix.
        asArrayOperator
            Converts the operator to an ArrayOperator object (numpy.ndarray).
        asmatrix
            Converts the operator to a matrix.
        cvxapply
            Applies the operator as a bilinear map to a cvxpy matrix.

"""

import numpy as np
import cvxpy
import itertools


class ArrayOperator(np.ndarray):
    """An operator as a numpy ndarray.

    This class inherits from numpy.ndarray and given an array will create a view
    of that array with the `ArrayOperator` type.

    Note
    ----
    No error checking is done to ensure that the array is numeric or of order 4.

    Parameters
    ----------
    array : array-like
        The array representing the operator.

    """

    def __new__(cls, array):
        return np.array(array).view(cls)

    def apply(self, mat):
        """Applies the operator as a bilinear map to the matrix `mat`.

        Note
        ----
        Not implemented.

        """
        raise NotImplementedError

    def asArrayOperator(self):
        """Converts the operator to an ArrayOperator object.

        Note
        ----
        Returns self

        """
        return self

    def asmatrix(self):
        """Converts the operator to a matrix."""
        shape = self.shape
        return np.array(self.reshape((shape[0]*shape[1], shape[2]*shape[3]), order='F'))


class DyadsOperator(object):
    r"""An operator as a sum of dyads.

    This class stores an operator in dyad form as

    .. math:: \mathcal{A} = \sum_{i=1}^r \mathbf{X}_i \otimes \mathbf{Y}_i,

    where we call the :math:`\mathbf{X}_i` and :math:`\mathbf{Y}_i` the left and
    right factors, respectively. This "factorized" form is natural in many
    settings, and storing the representation in the factored form requires less
    space.

    Note
    ----
    No error checking is done to ensure that the array is numeric or of order 4.

    Parameters
    ----------
    lfactors : list
        The (ordered) list of left factors.
    rfactors : list
        The (ordered) list of right factors.

    Attributes
    ----------
    lfactors : list
        The (ordered) list of left factors.
    rfactors : list
        The (ordered) list of right factors.
    nfactors : int
        The number of factors/dyads in the representation.

    """

    def __init__(self, lfactors, rfactors):
        # TODO: error checking
        assert len(lfactors) == len(rfactors)
        self.lfactors = lfactors
        self.rfactors = rfactors
        self.nfactors = len(lfactors)

    def apply(self, mat):
        """Applies the operator as a bilinear map to the matrix `mat`."""
        # TODO: can make more efficient using kron prods
        return np.sum([self.lfactors[i].dot(mat).dot(self.rfactors[i].T)
                       for i in range(self.nfactors)], axis=0)

    def cvxapply(self, mat):
        """Applies the operator as a bilinear map to the cvxpy matrix `mat`."""
        return sum([self.lfactors[i]*mat*self.rfactors[i].T for i in range(self.nfactors)])

    def asArrayOperator(self):
        """Converts the operator to an ArrayOperator object."""
        # FIXME: is self.asmatrix().reshape(self.shape, order='F') more efficient?
        temp = np.zeros(self.shape)
        for i, j, k, l in itertools.product(*[list(range(self.shape[n])) for n in range(4)]):
            temp[i, j, k, l] = sum([self.lfactors[r][i, j] *
                                    self.rfactors[r][k, l]
                                    for r in range(self.nfactors)])
        return ArrayOperator(temp)

    def asmatrix(self):
        """Converts the operator to a matrix.

        Note
        ----
        Does not work on cvxpy expressions.

        """
        shape = self.shape
        lfs = [lf.reshape((shape[0]*shape[1], 1), order='F') for lf in self.lfactors]
        rfs = [rf.reshape((shape[2]*shape[3], 1), order='F') for rf in self.rfactors]
        return np.hstack(lfs).dot(np.hstack(rfs).T)

    def __add__(self, other):
        """Sums two operators.

        Note
        ----
        Not implemented.

        """
        raise NotImplementedError # TODO: will require some error checking

    def __mul__(self, other):
        """Multiplies the operator by a scalar."""
        if np.isscalar(other):
            return DyadsOperator([other*lf for lf in self.lfactors], self.rfactors)
        else:
            # FIXME: should return a type or value error
            raise NotImplementedError

    @property
    def shape(self):
        """Tuple[int]: The shape of the operator as if it were an array."""
        if isinstance(self.lfactors[0], (cvxpy.Variable, cvxpy.Parameter)):
            lshape = self.lfactors[0].size
        else:
            lshape = self.lfactors[0].shape
        if isinstance(self.rfactors[0], (cvxpy.Variable, cvxpy.Parameter)):
            rshape = self.rfactors[0].size
        else:
            rshape = self.rfactors[0].shape
        return lshape + rshape


def innerprod(oper1, oper2):
    """Computes the inner product between two operators."""
    if isinstance(oper1, DyadsOperator) and isinstance(oper2, DyadsOperator):
        return sum([_matinnerprod(oper1.lfactors[r1], oper2.lfactors[r2]) *
                    _matinnerprod(oper1.rfactors[r1], oper2.rfactors[r2])
                    for r1 in range(oper1.nfactors)
                    for r2 in range(oper2.nfactors)])
    else:
        return np.sum(oper1.asArrayOperator() * oper2.asArrayOperator())


def _matinnerprod(mat1, mat2):
    """Computes inner product between two matrices."""
    if isinstance(mat1, (cvxpy.Variable, cvxpy.Parameter)) or isinstance(mat2, (cvxpy.Variable, cvxpy.Parameter)):
        return cvxpy.sum_entries(cvxpy.mul_elemwise(mat1, mat2))
    else:
        return np.sum(np.multiply(mat1, mat2))


def cvxinnerprod(oper1, oper2):
    """Computes the inner product between two CVXPY operators in dyads form.

    Parameters
    ----------
    oper1, oper2 : DyadsOperator
        DyadsOperators whose factors are CVXPY expressions.

    """
    return sum([cvxpy.sum_entries(cvxpy.mul_elemwise(oper1.lfactors[r1], oper2.lfactors[r2])) *
                cvxpy.sum_entries(cvxpy.mul_elemwise(oper1.rfactors[r1], oper2.rfactors[r2]))
                for r1 in range(oper1.nfactors)
                for r2 in range(oper2.nfactors)])


def kpsvd(oper):
    """Computes the Kronecker product SVD of the operator `oper`.

    The Kronecker product singular value decomposition (KP-SVD) is one
    generalization of the SVD to order-2 tensors of matrices (e.g., what we call
    operators.

    The decomposition results from computing the SVD of the matricized operator.

    Returns
    -------
        u : array
            Unitary matrix whose columns are the left singular vectors.
        s : array
            The singular values.
        v : array
            Unitary matrix whose rows are the right singular vectors.

    """
    mat = oper.asmatrix()
    return np.linalg.svd(mat)


def RandomArrayOperator(shape, dist=np.random.normal, args=[], kwargs={}):
    """Creates a random `ArrayOperator` object.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the resulting operator.
    dist : Optional[function]
        The function that will generate the random numbers.

        The default function is `normal` from the `numpy.random` submodule.

        The function must be able to return an array of random numbers whose
        shape is given by tuple of integers passed through the keyword argument
        `size`.
    args : Optional[list]
        A list of arguments to pass to the `dist` function.
    kwargs : Optional[dict]
        A dictionary of keyword arguments to pass to the `dist` function.

    Returns
    -------
    ArrayOperator
        The random operator.

    """

    return ArrayOperator(dist(*args, size=shape, **kwargs))


def RandomDyadsOperator(shape, nfactors=1,
                        ldist=np.random.normal, largs=[], lkwargs={},
                        rdist=np.random.normal, rargs=[], rkwargs={}):
    """Creates a random `DyadsOperator` object.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the resulting operator.
    nfactors : Optional[int]
        The number of random dyads to generate.

        The default value is 1.
    ldist : Optional[function]
        The function that will generate random numbers for the left factors.

        The default function is `normal` from the `numpy.random` submodule.

        The function must be able to return an array of random numbers whose
        shape is given by tuple of integers passed through the keyword argument
        `size`.
    largs : Optional[list]
        A list of arguments to pass to the `ldist` function.
    lkwargs : Optional[dict]
        A dictionary of keyword arguments to pass to the `ldist` function.
    rdist : Optional[function]
        The function that will generate random numbers for the right factors.

        This parameter is analogous to `ldist` and has the same requirements.
    rargs : Optional[list]
        A list of arguments to pass to the `rdist` function.
    rkwargs : Optional[dict]
        A dictionary of keyword arguments to pass to the `rdist` function.

    Returns
    -------
    DyadsOperator
        The random operator.

    """
    lfactors = [ldist(*largs, size=shape[0:2], **lkwargs) for r in range(nfactors)]
    rfactors = [rdist(*rargs, size=shape[2:4], **rkwargs) for r in range(nfactors)]
    return DyadsOperator(lfactors, rfactors)
