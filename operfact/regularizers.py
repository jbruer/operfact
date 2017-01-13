# -*- coding: utf-8 -*-
"""Objects and methods to create regularizers for operators.

This module contains the classes that define regularizers on operators and
provides helper methods to assist in creating regularizers.

"""

import cvxpy
import numpy as np


#
# Aliases for cvxpy norms
#
def norm_l1(mat):
    r"""The :math:`\ell_1` vector norm."""
    return cvxpy.norm(mat, 1)


def norm_l2(mat):
    r"""The :math:`\ell_2` vector norm."""
    return cvxpy.norm(cvxpy.vec(mat), 2)


def norm_linf(mat):
    r"""The :math:`\ell_\infty` vector norm."""
    return cvxpy.norm(mat, 'inf')


def norm_fro(mat):
    """The Frobenius matrix norm."""
    return cvxpy.norm(mat, 'fro')


def norm_s1(mat):
    """The Schatten 1-norm for matrices.

    The sum of the singular values of the matrix (often called the trace norm or
    the nuclear norm).

    """
    return cvxpy.norm(mat, 'nuc')


def norm_s2(mat):
    """The Schatten 2-norm for matrices.

    Synonymous with the Frobenius norm.

    """
    return norm_fro(mat)


def norm_sinf(mat):
    r"""The Schatten :math:`\infty`-norm for matrices.

    The largest singular value of the matrix (often called the operator norm).

    """
    return cvxpy.norm(mat, 2)


def norm_l1l2(mat):
    r"""The :math:`\ell_1 \otimes \ell_2` nuclear norm for matrices.

    Sum of the :math:`\ell_2` norms of the rows of the matrix.

    Synonymous with the matrix norm :math:`\Vert \cdot \Vert_{2,1}`.

    """
    return cvxpy.mixed_norm(mat, 2, 1)


def norm_l1linf(mat):
    r"""The :math:`\ell_1 \otimes \ell_\infty` nuclear norm for matrices.

    Sum of the :math:`\ell_\infty` norms of the rows of the matrix.

    Synonymous with the matrix norm :math:`\Vert \cdot \Vert_{\infty,1}`.

    """
    return cvxpy.mixed_norm(mat, np.Inf, 1)


def norm_l2l1(mat):
    r"""The :math:`\ell_2 \otimes \ell_1` nuclear norm for matrices.

    Sum of the :math:`\ell_2` norms of the columns of the matrix.

    Synonymous with the matrix norm :math:`\Vert \cdot \Vert_{2,1}` performed
    on the transpose of the matrix.

    """

    return cvxpy.mixed_norm(mat.T, 2, 1)


def norm_l2l2(mat):
    r"""The :math:`\ell_2 \otimes \ell_2` nuclear norm for matrices.

    Synonymous with the Schatten 1-norm.

    """
    return norm_s1(mat)


def norm_linfl1(mat):
    r"""The :math:`\ell_\infty \otimes \ell_1` nuclear norm for matrices.

    Sum of the :math:`\ell_\infty` norms of the columns of the matrix.

    Synonymous with the matrix norm :math:`\Vert \cdot \Vert_{\infty,1}`
    performed on the transpose of the matrix.

    """
    return cvxpy.mixed_norm(mat.T, np.Inf, 1)


def zero(mat):
    """The zero regularizer."""
    return 0


#
# Regularizer objects
#
class Regularizer(object):
    """Base class for all objects representing regularizers.

    This class implements the basic interface for a regularizer object---
    essentially a container to hold references to functions that implement the
    regularizers for each of the different solvers. All of the other
    regularizer types derive from this class.

    The convention used is that `norm_<name>` is the function that implements
    the regularizer for `solvers.<name>solve`.

    Attributes
    ----------
    norm_altmin : function
        Function that implements the regularizer for `solvers.altminsolve`.
    norm_mat : function
        Function that implements the regularizer for `solvers.matsolve`.
    norm_sdp : function
        Function that implements the regularizer for `solvers.sdpsolve`.

    """

    norm_altmin = None
    norm_mat = None
    norm_sdp = None

    def available_solvers(self):
        """Returns the solvers for which this regularizer has an implementation.

        Returns
        -------
        list
            A list of strings containing the names of the solvers.

            The convention used is that each `<name>` in the list corresponds to
            the solver `solvers.<name>solve`.

            See also the module variable `solvers.SOLVERS`.

        """


        out = []
        if self.norm_altmin is not None:
            out.append('altmin')
        if self.norm_mat is not None:
            out.append('mat')
        if self.norm_sdp is not None:
            out.append('sdp')
        return out


class NucNorm(Regularizer):
    """The base class for a nuclear norm-type regularizer.

    This class derives from `Regularizer` and implements the interface for a
    nuclear norm-type regularizer on operators. These regularizers are created
    by norming the spaces of the left and right factors individually. The norms
    of each space are stored as the attributes `lnorm` and `rnorm`. If the
    nuclear norm has a representation directly computable by CVXPY on matrices,
    that function is stored in the attribute `norm_mat` (see the documentation
    for `regularizers.Regularizer`).

    This class does not implement any other nuclear norm formulations; those are
    left for derived classes.

    Parameters
    ----------
    lnorm : function
        Function that computes the norm on the space of the left factors.
    rnorm : function
        Function that computes the norm on the space of the right factors.

    Attributes
    ----------
    lnorm : function
        Function that computes the norm on the space of the left factors.
    rnorm : function
        Function that computes the norm on the space of the right factors.
    norm_mat : function
        Function that implements the regularizer for `solvers.matsolve`.

        If no such function exists, this attribute is set to None.

    """

    lnorm = None
    rnorm = None

    def __init__(self, lnorm, rnorm):
        self.lnorm = lnorm
        self.rnorm = rnorm

        def norm_cvx(lnorm, rnorm):
            if lnorm is norm_l1:
                if rnorm is norm_l2:
                    return norm_l1l2
                elif rnorm is norm_l1:
                    return norm_l1
                elif rnorm is norm_linf:
                    return norm_l1linf
                else:
                    return None
            elif lnorm is norm_l2:
                if rnorm is norm_l2:
                    return norm_s1
                elif rnorm is norm_l1:
                    return norm_l2l1
                else:
                    return None
            elif lnorm is norm_linf:
                if rnorm is norm_linf:
                    return None
                elif rnorm is norm_l1:
                    return norm_linfl1
                else:
                    return None
            else:
                return None

        self.norm_mat = norm_cvx(self.lnorm, self.rnorm)

    def __call__(self, X):
        """Returns the value of the norm represented by the object.

        If the norm is not directly computable by CVXPY (i.e., `norm_mat` is
        None), then this function returns None.

        """
        if self.norm_mat is None:
            return None
        else:
            return self.norm_mat(X).value

    def __repr__(self):
        def strip_norm(name):
            return name.lstrip('norm_') if name.startswith('norm_') else name

        if hasattr(self.lnorm, '__name__'):
            lname = strip_norm(self.lnorm.__name__)
        else:
            lname = repr(lname)

        if hasattr(self.rnorm, '__name__'):
            rname = strip_norm(self.rnorm.__name__)
        else:
            rname = repr(rname)

        return '{0}: {1}, {2}'.format(self.__class__.__name__, lname, rname)


class NucNorm_Sum(NucNorm):
    r"""Implements nuclear norms using a sum representation.

    The nuclear norm on :math:`\Vert\cdot\Vert_{X} \otimes \Vert\cdot\Vert_{Y}`
    can be computed as

    .. math::

        N_{X,Y}(\mathcal{A}) = \inf\bigg\lbrace \sum_i
            \frac{1}{2}(\Vert \mathbf{X}_i \Vert_X^2 +
                        \Vert \mathbf{Y}_i \Vert_Y^2) :
            \mathcal{A} = \sum_i \mathbf{X}_i \otimes \mathbf{Y}_i \bigg\rbrace.

    The class method `norm_altmin` implements the sum

    .. math::

        \sum_i \frac{1}{2}(\Vert \mathbf{X}_i \Vert_X^2 +
                           \Vert \mathbf{Y}_i \Vert_Y^2),

    for use in the alternating minimization solver `solvers.altminsolve`.

    Parameters
    ----------
    lnorm : function
        Function that computes the norm on the space of the left factors.
    rnorm : function
        Function that computes the norm on the space of the right factors.

    Attributes
    ----------
    lnorm : function
        Function that computes the norm on the space of the left factors.
    rnorm : function
        Function that computes the norm on the space of the right factors.
    norm_mat : function
        Function that implements the regularizer for `solvers.matsolve`.

        If no such function exists, this attribute is set to None.

    """

    def __init__(self, lnorm, rnorm):
        super().__init__(lnorm, rnorm)

    def norm_altmin(self, lfactors, rfactors):
        r"""Implements the regularizer for `solvers.altminsolve`.

        As described in the class documentation, we can compute nuclear norms
        as a minimization problem. This function creates a CVXPY expression that
        represents the objective of that minimization problem. Specifically, we
        compute

        .. math::

            \sum_i \frac{1}{2}(\Vert \mathbf{X}_i \Vert_X^2 +
                               \Vert \mathbf{Y}_i \Vert_Y^2),

        where :math:`\mathbf{X}_i` and :math:`\mathbf{Y}_i` are the left and
        right factors of the operator in dyads representation.

        Parameters
        ----------
        lfactors : list
            The (ordered) list of the left factors in the dyad representation of
            the operator.
        rfactors : list
            The (ordered) list of the right factors in the dyad representation
            of the operator.

        Returns
        -------
        cvxpy.Expression
            The CVXPY representation of the norm objective to minimize.

        """
        obj = 0.0
        assert len(lfactors) == len(rfactors)
        for r in range(len(lfactors)):
            obj += 0.5*(cvxpy.square(self.lnorm(lfactors[r])) +
                        cvxpy.square(self.rnorm(rfactors[r])))
        return obj

    def __call__(self, X):
        """Returns the value of the norm.

        If the norm is not directly computable by CVXPY (i.e., `norm_mat` is
        None), then this function returns None.

        Parameters
        ----------
        X : matrix-like
            The matrix representation of the operator.

        Returns
        -------
        float
            The numeric value of the norm.

        """
        if self.norm_mat is None:
            return None
        else:
            return self.norm_mat(X).value


class NucNorm_Prod(NucNorm):
    r"""Implements nuclear norms using a product representation.

    The nuclear norm on :math:`\Vert\cdot\Vert_{X} \otimes \Vert\cdot\Vert_{Y}`
    can be computed as

    .. math::

        N_{X,Y}(\mathcal{A}) = \inf\bigg\lbrace \sum_i
            \Vert \mathbf{X}_i \Vert_X \Vert \mathbf{Y}_i \Vert_Y :
            \mathcal{A} = \sum_i \mathbf{X}_i \otimes \mathbf{Y}_i \bigg\rbrace.

    The class method `norm_altmin` implements the product

    .. math:: \sum_i \Vert\mathbf{X}_i\Vert_X \Vert\mathbf{Y}_i\Vert_Y,

    for use in the alternating minimization solver `solvers.altminsolve`.

    Parameters
    ----------
    lnorm : function
        Function that computes the norm on the space of the left factors.
    rnorm : function
        Function that computes the norm on the space of the right factors.

    Attributes
    ----------
    lnorm : function
        Function that computes the norm on the space of the left factors.
    rnorm : function
        Function that computes the norm on the space of the right factors.
    norm_mat : function
        Function that implements the regularizer for `solvers.matsolve`.

        If no such function exists, this attribute is set to None.

    """
    def __init__(self, lnorm, rnorm):
        super().__init__(lnorm, rnorm)

    def norm_altmin(self, lfactors, rfactors):
        r"""Implements the regularizer for `solvers.altminsolve`.

        As described in the class documentation, we can compute nuclear norms
        as a minimization problem. This function creates a CVXPY expression that
        represents the objective of that minimization problem. Specifically, we
        compute

        .. math:: \sum_i \Vert\mathbf{X}_i\Vert_X \Vert\mathbf{Y}_i\Vert_Y,

        where :math:`\mathbf{X}_i` and :math:`\mathbf{Y}_i` are the left and
        right factors of the operator in dyads representation.

        Parameters
        ----------
        lfactors : list
            The (ordered) list of the left factors in the dyad representation of
            the operator.
        rfactors : list
            The (ordered) list of the right factors in the dyad representation
            of the operator.

        Returns
        -------
        cvxpy.Expression
            The CVXPY representation of the norm objective to minimize.

        """
        obj = 0.0
        assert len(lfactors) == len(rfactors)
        for r in range(len(lfactors)):
            obj += self.lnorm(lfactors[r])*self.rnorm(rfactors[r])
        return obj


class NucNorm_SDR(NucNorm):
    r"""Implements nuclear norms using a semidefinite relaxation.

    The norm :math:`\Vert\cdot\Vert_X` on :math:`\mathbb{R}^m` is superquadratic
    if

    .. math:: \Vert\mathbf{x}\Vert_X = g_X(|\mathbf{x}|^2),

    for some function :math:`g_X\colon \mathbb{R}^m_+ \to \mathbb{R}` and all
    :math:`\mathbf{x} \in \mathbb{R}^m`, where :math:`|\mathbf{x}|^2` denotes
    the elementwise squaring of the vector :math:`\mathbf{x}`. We call
    :math:`g_X` the _gauge_ associated with the superquadratic norm
    :math:`\Vert\cdot\Vert_X`.

    For superquadratic norms :math:`\Vert\cdot\Vert_X` on :math:`\mathbb{R}^m`
    and :math:`\Vert\cdot\Vert_Y` on :math:`\mathbb{R}^n` (with associated
    gauges :math:`g_X` and :math:`g_Y`), we can compute a relaxation of the
    nuclear norm on :math:`\Vert\cdot\Vert_X \otimes \Vert\cdot\Vert_Y` as

    .. math::

        R_{X_Y}(\mathbf{A}) = \inf\bigg\lbrace \frac{1}{2}[
                                    g_X(\mathrm{diag}(\mathbf{W}_1)) +
                                    g_Y(\mathrm{diag}(\mathbf{W}_2))] :
                                    \begin{bmatrix}
                                    \mathbf{W}_1 & \mathbf{A} \\
                                    \mathbf{A}^T & \mathbf{W}_2
                                    \end{bmatrix}
                                    \succeq \mathbf{0}\bigg\rbrace,

    for all :math:`\mathbf{A} \in \mathbb{R}^m \otimes \mathbb{R}^n`.

    The class method `norm_sdp` implements the objective in semidefinite form as

    .. math::

        \frac{1}{2}[g_X(\mathrm{diag}(\mathbf{W}_1)) +
                    g_Y(\mathrm{diag}(\mathbf{W}_2))],

    for use in the semidefinite solver `solvers.sdpsolve`.

    The class method `norm_altmin` implements the objective in dyads form as

    .. math::

        \frac{1}{2}[g_X(\sum_i |\mathbf{x}_i|^2) +
                    g_Y(\sum_i |\mathbf{y}_i|^2)],

    for use in the alternating minimization solver `solvers.altminsolve`.

    Parameters
    ----------
    lnorm : function
        Function handle to the superquadratic norm on the left factors.
    rnorm : function
        Function handle to the superquadratic norm on the right factors.

    Attributes
    ----------
    lnorm : function
        Function handle to the superquadratic norm on the left factors.
    rnorm : function
        Function handle to the superquadratic norm on the right factors.
    lgauge : function
        Function handle to the gauge associated with the left norm.
    rgauge : function
        Function handle to the gauge associated with the right norm.

    """
    lgauge = None
    rgauge = None

    def __init__(self, lnorm, rnorm):
        self.lnorm = lnorm
        self.rnorm = rnorm
        self.norm_mat = None  # do not allow calling matsolve

        # From the provided norms, return the appropriate gauges
        def norm_to_gauge(norm):
            if norm is norm_l2:
                return cvxpy.sum_entries
            elif norm is norm_linf:
                return cvxpy.max_entries
            else:
                return None

        self.lgauge = norm_to_gauge(lnorm)
        self.rgauge = norm_to_gauge(rnorm)

        if self.lgauge is None:
            raise ValueError('NucNorm_SDR: lnorm {0} not valid'.format(lnorm))
        if self.rgauge is None:
            raise ValueError('NucNorm_SDR: rnorm {0} not valid'.format(rnorm))

    def norm_altmin(self, lfactors, rfactors):
        r"""Implements the regularizer for `solvers.altminsolve`.

        As described in the class documentation, we can compute the semidefinite
        relaxation of the nuclear norm as a minimization problem. This function
        creates a CVXPY expression that represents the objective of that
        minimization problem. Specifically, we compute

        .. math::

            \frac{1}{2}[g_X(\sum_i |\mathbf{x}_i|^2) +
                        g_Y(\sum_i |\mathbf{y}_i|^2)],

        where :math:`\mathbf{x}_i` and :math:`\mathbf{y}_i` are the left and
        right factors of the operator in dyads representation.

        Parameters
        ----------
        lfactors : list
            The (ordered) list of the left factors in the dyad representation of
            the operator.
        rfactors : list
            The (ordered) list of the right factors in the dyad representation
            of the operator.

        Returns
        -------
        cvxpy.Expression
            The CVXPY representation of the norm objective to minimize.

        """
        assert len(lfactors) == len(rfactors)
        ltemp = cvxpy.Constant(np.zeros(lfactors[0].size))
        rtemp = cvxpy.Constant(np.zeros(rfactors[0].size))
        for r in range(len(lfactors)):
            ltemp += cvxpy.square(lfactors[r])
            rtemp += cvxpy.square(rfactors[r])
        return 0.5*(self.lgauge(ltemp) + self.rgauge(rtemp))

    def norm_sdp(self, ldiag, rdiag):
        r"""Implements the regularizer for `solvers.altminsolve`.

        As described in the class documentation, we can compute a relaxation of
        the nuclear norm of superquadratic norms as a semidefinite program. This
        function creates a CVXPY expression that represents the objective of
        that minimization problem. Specifically, we compute

        .. math::

            \frac{1}{2}[g_X(\mathrm{diag}(\mathbf{W}_1)) +
                        g_Y(\mathrm{diag}(\mathbf{W}_2))],

        where :math:`\mathbf{W}_1` and :math:`\mathbf{W}_1` are on the diagonal
        of the semidefinite decision variable in block form as

        .. math:: \begin{bmatrix} \mathbf{W}_1 & \mathbf{A} \\
                                  \mathbf{A}^T & \mathbf{W}_2 \end{bmatrix},

        and :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` is the matrix
        representation of the operator
        :math:`\mathcal{A} \in \mathbb{R}^m \otimes \mathbb{R}^n`.

        Parameters
        ----------
        ldiag : vector-like
            The diagonal of the matrix :math:`\mathbf{W}_1` (above) as a vector.
        rdiag : vector-like
            The diagonal of the matrix :math:`\mathbf{W}_2` (above) as a vector.

        Returns
        -------
        cvxpy.Expression
            The CVXPY representation of the norm objective to minimize.

        """
        return 0.5*(self.lgauge(ldiag) + self.rgauge(rdiag))


class MaxNorm(Regularizer):
    r"""Implements the max-norm.

    The max-norm is defined as

    .. math::

        R_{\infty,\infty}(\mathcal{A}) :=
        \inf\left\lbrace \frac{1}{2} \left[\max_i (\mathbf{W}_1)_{ii} + \max_j (\mathbf{W}_2)_{jj} \right] \mid
        \begin{bmatrix} \mathbf{W}_1 & \mathbf{A} \\ \mathbf{A}^T & \mathbf{W}_2 \end{bmatrix}
        \succeq \mathbf{0} \right\rbrace

    This is equivalent to the semidefinite relaxation of the
    :math:`\ell_\infty \otimes \ell_\infty` nuclear norm. This class serves as
    an alias for

    .. code:: python

        NucNorm_SDR(norm_linf, norm_linf)

    Attributes
    ----------
    inner_regularizer : NucNorm_SDR
        The actual regularizer object that `MaxNorm` aliases.

    """
    # alias for NucNorm_SDR(norm_linf, norm_linf), remove?
    inner_regularizer = NucNorm_SDR(norm_linf, norm_linf)

    def norm_altmin(self, lfactors, rfactors):
        """Returns the implementation of the max-norm for `solvers.altminsolve`.

        See the documentation of `NucNorm_SDR` for implementation details.

        """
        return self.inner_regularizer.norm_altmin(lfactors, rfactors)

    def norm_sdp(self, ldiag, rdiag):
        """Returns the implementation of the max-norm for `solvers.sdpsolve`.

        See the documentation of `NucNorm_SDR` for implementation details.

        """
        return self.inner_regularizer.norm_sdp(ldiag, rdiag)

    def __call__(self, X):
        """Returns the max-norm of the matrix `X`.

        Note
        ----
        This function is currently unimplemented and will return `None`. This is
        the expected return of `__call__` when the computation is not
        implemented. See the documentation of `NucNorm.__call__` for details.

        """
        return None

    def __repr__(self):
        return 'max_norm'


class VectorNorm(Regularizer):
    r"""Implements vector p-norms.

    Given an operator :math:`\mathcal{A} \in \mathbb{R}^m \otimes \mathbb{R}^n`,
    we can form the matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}`. From
    this we can also form the vector :math:`\mathbf{a} \in \mathbb{R}^{mn}` and
    compute its :math:`\ell_p` norm.

    This class serves as a wrapper for applying CVXPY :math:`\ell_p` norms to
    matrices.

    Parameters
    ----------
    p : int or 'inf'
        Specifies the p-norm to compute; requires `p>=1` or `p='inf'`.

    Attributes
    ----------
    p : int of 'inf'
        Specifies the p-norm computed.


    """
    p = None

    def __init__(self, p):
        self.p = p

    def norm_mat(self, X):
        r"""Returns the implementation of the vector norm for matrices.

        This function uses CVXPY commands to transform the input matrix `X` into
        a vector and then compute the :math:`\ell_p` norm.

        Parameters
        ----------
        X : matrix-like
            The matrix whose vector norm we wish to compute.

        Returns
        -------
        cvxpy.Expression
            The CVXPY expression of the :math:`\ell_p` norm.

        """
        return cvxpy.pnorm(cvxpy.vec(X), self.p)

    def __call__(self, X):
        """Returns the p-norm of the matrix `X` converted to a vector."""
        return self.norm_mat(X).value

    def __repr__(self):
        return 'vec: {0}'.format(self.p)


#
# Penalty constant helper functions
#
def penconst_denoise(shape, scale, regularizer):
    """Computes a suggestion for the penalty constant of a denoising problem.

    Assume that we have noisy observations of an operator :math:`\mathcal{A}`:

    .. math:: \mathcal{Y} = \mathcal{A} + \mathcal{W},

    where :math:`\mathcal{W}` is Gaussian noise with scale :math:`\sigma`.

    To denoise this operator with the nuclear norm :math:`N` as a regularizer,
    we solve

    .. math::

        \operatorname{minimize}_\mathcal{X} \Vert \mathcal{X} - \mathcal{Y} \Vert^2_F +
                                    \lambda N(\mathcal{X}).

    The performance of the denoiser depends on the choice of :math:`\lambda >0`,
    and we choose

    .. math:: \lambda = \mathbb{E} N^*(\mathcal{W}),

    where :math:`N^*` is the dual of the nuclear norm :math:`N`.

    Note that this computation depends on the shape of :math:`\mathcal{A}`, the
    noise level :math:`\sigma`, and the choice of the nuclear norm :math:`N`.

    Note
    ----
    Not all choices of the nuclear norm :math:`N` lead to efficient computations
    of the dual norm :math:`N^*`. In cases where this computation is not
    feasible, we return a heuristic guess.

    Parameters
    ----------
    shape : tuple
        The shape of the operator to denoise.
    scale : float
        The standard deviation of the random Gaussian noise (:math:`\sigma`).
    regularizer : Regularizer
        The regularizer used in the denoising problem (:math:`N`)

    Returns
    -------
    float
        The suggested penalty constant (:math:`\lambda`).

    """
    from scipy.stats import chi, gumbel_r, norm

    def max_of_variates(n, ppf):
        return gumbel_r.mean(loc=ppf(1-1.0/(n+1)),
                             scale=(ppf(1-1.0/(np.e*(n+1)))-ppf(1-1.0/(n+1))))

    def max_of_gaussians(n):
        return max_of_variates(n, norm.ppf)

    # Warnings and errors
    def warn_guess():
        print('warning: guess penconst for {0}'.format(regularizer))

    def warn_default():
        print('warning: default penconst for {0}'.format(regularizer))
        return scale*(np.sqrt(lshape) + np.sqrt(rshape))

    def error_value():
        raise ValueError('Bad regularizer {0}'.format(regularizer))

    lshape = shape[0]*shape[1]  # total dimension of left factor
    rshape = shape[2]*shape[3]  # total dimension of right factor
    lrank = min(shape[0:2])
    rrank = min(shape[2:4])

    if isinstance(regularizer, NucNorm):
        lnorm = regularizer.lnorm
        rnorm = regularizer.rnorm
        # Choose appropriate penalty constant
        if lnorm is norm_l1:
            if rnorm is norm_l2:
                out = scale*max_of_variates(lshape,
                                            ppf=lambda q: chi.ppf(q, rshape))
            elif rnorm is norm_l1:
                # \ell_\infty norm of noise
                out = scale*max_of_gaussians(2*np.prod(shape)) # doubling because we want max of folded Gaussians --- maybe use chi instead? slower and probably not much difference
            elif rnorm is norm_linf:
                # max of \ell_1 norms of rows of noise
                out = scale*np.mean([np.max([np.sum(np.abs(np.random.normal(size=(rshape)))) for i in range(lshape)]) for j in range(1000)])
            elif rnorm is norm_s1:
                # max of S_\infty norms of right factors
                out = scale*np.mean([np.max([np.linalg.norm(np.random.normal(size=shape[2:4]), ord=2) for j in range(lshape)]) for i in range(1000)])
            elif rnorm is norm_sinf:
                # max of S_1 norms of right factors
                out = scale*np.mean([np.max([np.linalg.norm(np.random.normal(size=shape[2:4]), ord='nuc') for j in range(lshape)]) for i in range(1000)])
            else:
                out = warn_default()
        elif lnorm is norm_l2:
            if rnorm is norm_l2:
                # S_\infty norm of noise
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))
            elif rnorm is norm_l1:
                # max of \ell_2 norm of columns of noise
                out = scale*max_of_variates(rshape,
                                            ppf=lambda q: chi.ppf(q, lshape))
            elif rnorm is norm_linf:
                # guess for \ell_2, \ell_infty
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(rshape)  # SHAPE[2] correct?
            elif rnorm is norm_sinf:
                # guess for \ell_2, S_\infty
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(rrank)
            else:
                out = warn_default()
        elif lnorm is norm_linf:
            if rnorm is norm_linf:
                # guess for \ell_\infty, \ell_\infty
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(lshape*rshape)
            elif rnorm is norm_l1:
                # max of \ell_1 norm of columns of noise
                out = scale*np.mean([np.max([np.sum(np.abs(np.random.normal(size=(lshape)))) for i in range(rshape)]) for j in range(1000)])
            elif rnorm is norm_l2:
                # guess for \ell_\infty, \ell_2
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(lshape)
            elif rnorm is norm_s1:
                # guess for \ell_\infty, S_1
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(lshape)
            elif rnorm is norm_sinf:
                # guess for \ell_\infty, S_\infty
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(lshape*rrank)
            else:
                out = warn_default()
        elif lnorm is norm_s1:
            if rnorm is norm_l1:
                # max of S_\infty norms of left factors
                out = scale*np.mean([np.max([np.linalg.norm(np.random.normal(size=shape[0:2]), ord=2) for j in range(rshape)]) for i in range(1000)])
            elif rnorm is norm_linf:
                # guess for S_1, \ell_\infty
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(rshape)
            elif rnorm is norm_sinf:
                # guess for S_1, S_\infty
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(rrank)
            else:
                out = warn_default()
        elif lnorm is norm_sinf:
            if rnorm is norm_l1:
                # max of S_1 norms of left factors
                out = scale*np.mean([np.max([np.linalg.norm(np.random.normal(size=shape[0:2]), ord='nuc') for j in range(rshape)]) for i in range(1000)])
            elif rnorm is norm_l2:
                # guess for S_\infty, \ell_2
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(lrank)
            elif rnorm is norm_linf:
                # guess for S_\infty, \ell_\infty
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(lrank*rshape)
            elif rnorm is norm_s1:
                # guess for S_\infty, S_1
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(lrank)
            elif rnorm is norm_sinf:
                # guess for S_\infty, S_\infty
                warn_guess()
                out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(lrank*rrank)
            else:
                out = warn_default()
        else:
            out = warn_default()
    elif isinstance(regularizer, VectorNorm) and regularizer.p == 'inf':
        out = scale*np.mean([np.sum(np.abs(np.random.normal(size=(np.prod(shape))))) for i in range(1000)])
    elif isinstance(regularizer, MaxNorm):
        # guess for max-norm
        warn_guess()
        out = scale*(np.sqrt(lshape) + np.sqrt(rshape))*np.sqrt(lshape*rshape)
    else:
        # otherwise, just use l2 \otimes l2
        out = warn_default()
    return out
