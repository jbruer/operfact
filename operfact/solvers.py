# -*- coding: utf-8 -*-
"""Objects and methods to solve operator linear inverse problems..

This module contains the classes that define the input and output of our solvers
and the solvers themselves.

The implemented solvers are:

    * `altminsolve`: An alternating minimization solver. This nonconvex solver
        is the most general in this package and provides the largest support for
        various nuclear norm regularizers.
    * `matsolve`: A solver that works with matrix representations of operators.
        For some choices of regularizer, we can express the inverse problem as a
        convex problem over matrices.
    * `sdpsolve`: A solver that works with a semidefinite representation of
        operators. For some choices of regularizer, we can write the nuclear
        norm (or at least a relaxation of it) as a convex function of a
        semidefinite matrix.

See the documentation of each solver for more details.

"""

import cvxpy
import numpy as np
import time
from . import operators


class Problem(object):
    """Represents a complete operator inverse problem to pass to the solvers.

    The solvers in this module expect instances of this class as input. Not all
    solvers require all attributes to be populated. See the documentation of
    each solver for its requirements.

    Attributes
    ----------
    trueOperator : ArrayOperator or DyadsOperator
        If known, the operator we wish to recover.

        This is used for reference and debugging. In practical problems, the
        true operator would not be known.
    shape : tuple
        The shape of the operator to recover.
    measurementobj : Measurement
        An object representing the linear measurement operator in the problem.
    measurementvec : vector-like
        The observed (possibly noisy) measurements.
    lmeasurementvec : vector-like
        The measurements for the "left" subproblem (if necessary).
    rmeasurementvec : vector-like
        The measurements for the "right" subproblem (if necessary).
    norm : Regularizer
        An object representing the regularizer in the problem.
    rank : int
        The desired rank of the solution.
    solveropts : dict
        A dictionary of options to pass to the CVXPY solver.
    relconvergetol : float or list
        A stopping criterion for `altminsolve`.

        See the documentation of `altminsolve` for details (default: 1e-3).
    penconst : float
        The penalty (regularization) constant in the problem (default: 1.0).
    maxiters : int or list
        A stopping criterion for `altminsolve`.

        See the documentation of `altminsolve` for details (default: 10).
    solver : str
        The CVXPY solver to use. (default: cvxpy.SCS)
    rfactorsinit : list
        The starting point for the right factors in `altminsolve`.

        See the documentation of `altminsolve` for details.
    cache : dict
        A dictionary used to store intermediate calculations

    """
    def __init__(self):
        self.trueOperator = None
        self.shape = None
        self.measurementobj = None
        self.measurementvec = None
        self.lmeasurementvec = None
        self.rmeasurementvec = None
        self.norm = None
        self.rank = None
        self.solveropts = {}
        self.relconvergetol = 1e-3
        self.penconst = 1.0
        self.maxiters = 10
        self.solver = cvxpy.SCS
        self.rfactorsinit = None
        self.cache = {}

class SolverOutput(object):
    """Stores the output of our solvers.

    The solvers in this module use instances of this class to store their
    output. Not all populate all attributes. See the documentation of
    each solver for its output.

    Attributes
    ----------
    problem : Problem
        The original `Problem` object passed to the solver.
    cvxpy_probs : tuple(cvxpy.Problem)
        The CVXPY problem object(s) created by the solver.
    recovered : ArrayOperator or DyadsOperator
        The recovered operator.
    outer_iters : int
        The number of outer iterations required.

        Only applicable for `altminsolve`.
    setup_time : float
        The time taken before calling `solve` on the CVXPY problems.
    solve_time : float
        The time taken by CVXPY to solve the problem.
    objval : float
        The value of the objective after optimization.
    relchange : float
        The relative change of the operator at the last outer iteration of
        `altminsolve`.
    relconvtol : float
        The (effective) relative convergence tolerance used by `altminsolve`.
    maxiters : int
        The (effective) outer iterations tolerance used by `altminsolve`.
    debug : dict
        A dictionary containing extra debugging information.

    """
    problem = None
    cvxpy_probs = None
    recovered = None
    outer_iters = None
    setup_time = 0.0
    solve_time = 0.0
    objval = None
    relchange = None
    relconvtol = None
    maxiters = None
    debug = {}

    @property
    def total_time(self):
        """float: The total time taken by the solver."""
        return self.setup_time + self.solve_time


def altminsolve(problem, noquad=False, eqconstraint=False, sthresh=0.0):
    r"""An alternating minimization solver for operator linear inverse problems.

    Uses an explicit factorization of the decision variable to solve:

    .. math::

        \operatorname{minimize}_{\mathbf{X}_i,\mathbf{Y}_i}
        \frac{1}{2} \bigg\lVert \mathbf{b} -
        \mu\bigg( \sum_{i=1}^r \mathbf{X}_i \otimes \mathbf{Y}_i \bigg) \bigg\rVert^2_{\ell_2} +
        \lambda \sum_{i=1}^r \lVert \mathbf{X}_i \rVert_X \lVert \mathbf{Y}_i \rVert_Y.


    The above formulation corresponds to using a `NucNorm_Prod` object with X and Y norms as the regularizer.

    Parameters
    ----------
    problem : Problem
        The `Problem` object to solve.
    noquad : Optional[bool]
        Whether we replace the quadratic measurement error with a norm.

        Default value is False.
    eqconstraint : Optional[bool]
        Whether we solve the problem with an equality constraint.

        Default value is False.
    sthresh : Optional[float]
        The hard thresholding limit when solving with an equality constraint.

        Default value is 0 (i.e., no thresholding).

    Returns
    -------
    SolverOutput or list(SolverOutput)
        An object with the results of the optimization problem.

    """
    start = time.time()
    LAMBDA = problem.penconst
    rank = problem.rank
    shape = problem.shape

    relconvtols = np.sort(np.array(problem.relconvergetol, ndmin=1))[::-1]
    maxiters = np.sort(np.array(problem.maxiters, ndmin=1))

    lmeasvec = problem.measurementvec if problem.lmeasurementvec is None \
        else problem.lmeasurementvec
    rmeasvec = problem.measurementvec if problem.rmeasurementvec is None \
        else problem.rmeasurementvec

    lvars = [cvxpy.Variable(shape[0], shape[1]) for r in range(rank)]
    lparams = [cvxpy.Parameter(shape[2], shape[3]) for r in range(rank)]
    lconsts = []
    rvars = [cvxpy.Variable(shape[2], shape[3]) for r in range(rank)]
    rparams = [cvxpy.Parameter(shape[0], shape[1]) for r in range(rank)]
    rconsts = []

    lmeas_temp = problem.measurementobj.cvxapply(operators.DyadsOperator(lvars,lparams))
    rmeas_temp = problem.measurementobj.cvxapply(operators.DyadsOperator(rparams,rvars))

    if eqconstraint:
        lconsts.append(lmeas_temp == np.array(lmeasvec))
        rconsts.append(rmeas_temp == np.array(rmeasvec))
    else:
        if noquad:
            lmeaserr = cvxpy.norm(lmeas_temp - np.array(lmeasvec))
            rmeaserr = cvxpy.norm(rmeas_temp - np.array(rmeasvec))
        else:
            lmeaserr = cvxpy.quad_over_lin(lmeas_temp - np.array(lmeasvec), 2.0)
            rmeaserr = cvxpy.quad_over_lin(rmeas_temp - np.array(rmeasvec), 2.0)

    if eqconstraint:
        lobj = cvxpy.Minimize(problem.norm.norm_altmin(lvars, lparams))
        robj = cvxpy.Minimize(problem.norm.norm_altmin(rparams, rvars))
    else:
        lobj = cvxpy.Minimize(LAMBDA*problem.norm.norm_altmin(lvars, lparams) + lmeaserr)
        robj = cvxpy.Minimize(LAMBDA*problem.norm.norm_altmin(rparams, rvars) + rmeaserr)

    lprob = cvxpy.Problem(lobj, lconsts)
    rprob = cvxpy.Problem(robj, rconsts)

    # If solving the constrained version, set up needed matrices
    if eqconstraint:
        # NB: using the normal equations directly could be unstable
        start_linalg = time.time()
        measmat = problem.measurementobj.asmatrix()
        graminv = np.linalg.inv(measmat @ measmat.T)
        xfeas = measmat.T @ (graminv @ problem.measurementvec)
        end_linalg = time.time()
        if problem.solveropts.get('verbose', False):
            print('linalg time: ', end_linalg - start_linalg)

    if problem.rfactorsinit is None:
        # Will initialize lparams.
        if eqconstraint:
            temp = operators.ArrayOperator(xfeas.reshape(shape, order='F'))
            [U, S, V] = operators.kpsvd(temp)
            for r in range(rank):
                lparams[r].value = V[r, ].reshape(shape[2:4], order='F')
        else:
            # FIXME: if initfrommeas not implemented, resort to random init
            U, S, V = operators.kpsvd(problem.measurementobj.initfrommeas(lmeasvec))
            # only filling up to SOLVE_RANK, won't neeed more for initialization
            # If SVD = u s v.H, numpy returns U = u, S = s, V = v.H!
            # So we want to pull the rows of V!
            for r in range(rank):
                lparams[r].value = V[r,].reshape((shape[2], shape[3]), order='F')
    else:
        for r in range(rank):
            lparams[r].value = problem.rfactorsinit[r]

    end_setup = time.time()
    iters = 0
    objval = -np.Inf
    outlist = []
    time_offset = 0
    tol_ix = 0
    iters_ix = 0
    Sout = np.zeros((1, rank))  # for debugging eqconstraint
    while True:
        iters += 1
        if problem.solveropts.get('verbose', False):
            print('Outer iteration: %i' % iters)

        lprob.solve(solver=problem.solver, **problem.solveropts)

        if eqconstraint:
            temp = operators.DyadsOperator([np.matrix(lvars[r].value) for r in range(rank)],
                                           [np.matrix(lparams[r].value) for r in range(rank)])
            temp = temp.asArrayOperator().reshape(xfeas.shape, order='F')
            temp -= measmat.T @ (graminv @ (measmat @ (temp - xfeas)))
            temp = operators.ArrayOperator(temp.reshape(shape, order='F'))
            [U, S, V] = operators.kpsvd(temp)
            Sout = np.vstack((Sout, S))
            Ssum = np.sum(S**2)
            for r in range(rank):
                S[r] = 0.0 if S[r]**2/Ssum < sthresh else S[r]
                rparams[r].value = np.sqrt(S[r])*U[:, r].reshape(shape[0:2], order='F')
            if problem.solveropts.get('verbose', False):
                print('feasgap: ', np.linalg.norm((measmat@temp.flatten(order='F')) - rmeasvec.flatten(order='F')))
                print('Vinnerprod: ', np.dot(V[:,0].flatten(order='F'),
                                             problem.trueOperator.rfactors[0].flatten(order='F')/np.linalg.norm(problem.trueOperator.rfactors[0].flatten())))
                print('spectrum: ', S)
        else:
            for r in range(rank):
                rparams[r].value = lvars[r].value

        rprob.solve(solver=problem.solver, **problem.solveropts)

        if eqconstraint:
            temp = operators.DyadsOperator([np.matrix(rparams[r].value) for r in range(rank)],
                                           [np.matrix(rvars[r].value) for r in range(rank)])
            temp = temp.asArrayOperator().reshape(xfeas.shape, order='F')
            temp -= measmat.T @ (graminv @ (measmat @ (temp - xfeas)))
            temp = operators.ArrayOperator(temp.reshape(shape, order='F'))
            [U, S, V] = operators.kpsvd(temp)
            Sout = np.vstack((Sout, S))
            Ssum = np.sum(S**2)
            for r in range(rank):
                S[r] = 0.0 if S[r]**2/Ssum < sthresh else S[r]
                lparams[r].value = np.sqrt(S[r])*V[r, :].reshape(shape[2:4], order='F')
            if problem.solveropts.get('verbose', False):
                print('feasgap: ', np.linalg.norm((measmat@temp.flatten(order='F')) - lmeasvec.flatten(order='F')))
                print('Uinnerprod: ', np.dot(U[:,0].flatten(order='F'),
                                             problem.trueOperator.lfactors[0].flatten(order='F')/np.linalg.norm(problem.trueOperator.lfactors[0].flatten())))
                print('spectrum: ', S)
        else:
            for r in range(rank):
                lparams[r].value = rvars[r].value

        objval_new = rprob.objective.value
        relchange = np.abs(objval - objval_new) / np.max((objval_new, 1))
        objval = objval_new

        if problem.solveropts.get('verbose', False):
            print('relchange: %f' % relchange)

        def create_out(relconvtol, maxiters):
            out = SolverOutput()
            out.problem = problem
            out.cvxpy_probs = (lprob, rprob)
            # When solving the constrained problem, recompute rparams as well
            if eqconstraint:
                for r in range(rank):
                    rparams[r].value = np.sqrt(S[r])*U[:, r].reshape(shape[0:2], order='F')
            lfactors = [np.matrix(rparams[r].value) for r in range(rank)]
            rfactors = [np.matrix(lparams[r].value) for r in range(rank)]
            out.recovered = operators.DyadsOperator(lfactors, rfactors)
            out.outer_iters = iters
            out.setup_time = end_setup - start
            out.solve_time = (end_solve - end_setup) - time_offset
            out.objval = objval
            out.relchange = relchange
            out.relconvtol = relconvtol
            out.maxiters = maxiters
            out.debug['S'] = Sout
            return out

        # put in [relconvtol] X [maxiter] (need to output all)
        # 1) Check if we hit an iter limit
        #   - fill in all relconvtols from tol_ix to end
        # 2) Check if we hit a relconvtol
        #   - fill in from iters_ix to end (can result in duplicate)
        # 3) Check if we're done
        iters_ix_curr = iters_ix
        tol_ix_curr = tol_ix
        end_solve = time.time()

        if iters == maxiters[iters_ix_curr]:
            for tol in relconvtols[tol_ix:]:
                outlist.append(create_out(tol, maxiters[iters_ix_curr]))
            iters_ix += 1

        while tol_ix < relconvtols.size and relchange <= relconvtols[tol_ix]:
            for eff_iters in maxiters[iters_ix_curr:]:
                # If also hit a maxiters, don't ouput duplicate
                if iters == eff_iters:
                    continue
                outlist.append(create_out(relconvtols[tol_ix], eff_iters))
            tol_ix += 1

        time_offset += time.time() - end_solve

        if ((tol_ix == relconvtols.size) or
                (np.max(maxiters) == iters)):
            break

    assert len(outlist) == maxiters.size*relconvtols.size
    return outlist if len(outlist) > 1 else outlist[0]


def matsolve(problem, compute_dyads=False, noquad=False, eqconstraint=False):
    """A solver for operator linear inverse problems in matrix representation.

    Solves the convex operator recovery problem provided that the `Regularizer` object has a CVXPY implementation.

    Parameters
    ----------
    problem : Problem
        The `Problem` object to solve.
    compute_dyads : Optional[bool]
        Whether to return the operator in dyadic representation.

        When True, we compute the SVD of the matrix decision variable and create
        a `DyadsOperator` using the scaled left/right singular vectors as the
        left/right factors. Otherwise we reshape the matrix decision variable
        to have the shape of the operator and return an `ArrayOperator`.

        Default value is False.
    noquad : Optional[bool]
        Whether we replace the quadratic measurement error with a norm.

        Default value is False.
    eqconstraint : Optional[bool]
        Whether we solve the problem with an equality constraint.

        Default value is False.

    Returns
    -------
    SolverOutput
        An object with the results of the optimization problem.

    """
    matshape = (int(np.prod(problem.shape[0:2])),
                int(np.prod(problem.shape[2:4])))
    RANK = min(matshape)
    start = time.time()
    X = cvxpy.Variable(*matshape)

    reg = problem.norm.norm_mat(X)
    meastemp = problem.measurementobj.matapply(X)
    if eqconstraint:
        # solve: min ||X|| s.t. A(X) = A(X_0)
        prob = cvxpy.Problem(cvxpy.Minimize(reg),
                             [meastemp == problem.measurementvec, ])
    else:
        # solve: min .5*||A(X) - A(X_0)||_F^2 + LAMBDA*||X||
        LAMBDA = problem.penconst
        if noquad:
            measerr = cvxpy.norm(meastemp - problem.measurementvec)
        else:
            measerr = cvxpy.quad_over_lin(meastemp - problem.measurementvec, 2.0)
        prob = cvxpy.Problem(cvxpy.Minimize(measerr + LAMBDA*reg))

    end_setup = time.time()
    prob.solve(solver=problem.solver, **problem.solveropts)
    end_solve = time.time()
    out = SolverOutput()
    out.problem = problem
    out.cvxpy_probs = (prob,)
    if compute_dyads:
        U, S, V = np.linalg.svd(X.value, full_matrices=0)
        lfactors = [np.sqrt(S[i]) * U[:, i].reshape(problem.shape[0:2], order='F')
                    for i in range(RANK)]
        # just using V.T instead of V.H
        rfactors = [np.sqrt(S[i]) * V.T[:, i].reshape(problem.shape[2:4], order='F')
                    for i in range(RANK)]
        out.recovered = operators.DyadsOperator(lfactors, rfactors)
    else:
        temp = np.array(X.value).reshape(problem.shape, order='F')
        out.recovered = operators.ArrayOperator(temp)
    out.setup_time = end_setup - start
    out.solve_time = end_solve - end_setup
    out.objval = prob.objective.value
    return out


def sdpsolve(problem, compute_dyads=False, noquad=False, eqconstraint=False):
    """A semidefinite convex solver for operator linear inverse problems.

    Solve the nuclear norm recovery problem with a semidefinite inequality replacing the decomposition constraint.
    See `NucNorm_SDR` for more details.

    Parameters
    ----------
    problem : Problem
        The `Problem` object to solve.
    compute_dyads : Optional[bool]
        Whether to return the operator in dyadic representation.

        When True, we compute the SVD of the matrix decision variable and create
        a `DyadsOperator` using the scaled left/right singular vectors as the
        left/right factors. Otherwise we reshape the matrix decision variable
        to have the shape of the operator and return an `ArrayOperator`.

        Default value is False.
    noquad : Optional[bool]
        Whether we replace the quadratic measurement error with a norm.

        Default value is False.
    eqconstraint : Optional[bool]
        Whether we solve the problem with an equality constraint.

        Default value is False.

    Returns
    -------
    SolverOutput
        An object with the results of the optimization problem.

    """
    matshape = (int(np.prod(problem.shape[0:2])),
                int(np.prod(problem.shape[2:4])))
    RANK = min(matshape)
    start = time.time()
    X = cvxpy.Semidef(matshape[0] + matshape[1])
    A = X[0:matshape[0], matshape[0]:]
    diag = cvxpy.diag(X)
    ldiag = diag[0:matshape[0]]
    rdiag = diag[matshape[0]:]

    reg = problem.norm.norm_sdp(ldiag, rdiag)
    meastemp = problem.measurementobj.matapply(A)
    if eqconstraint:
        prob = cvxpy.Problem(cvxpy.Minimize(reg),
                             [meastemp == problem.measurementvec, ])
    else:
        LAMBDA = problem.penconst
        if noquad:
            measerr = cvxpy.norm(meastemp - problem.measurementvec)
        else:
            measerr = cvxpy.quad_over_lin(meastemp - problem.measurementvec, 2.0)
        prob = cvxpy.Problem(cvxpy.Minimize(measerr + LAMBDA*reg))

    end_setup = time.time()
    prob.solve(solver=problem.solver, **problem.solveropts)
    end_solve = time.time()
    out = SolverOutput()
    out.problem = problem
    out.cvxpy_probs = (prob,)
    if compute_dyads:
        # TODO: compute_dyads should use the semidefinite representation
        U, S, V = np.linalg.svd(A.value, full_matrices=0)
        lfactors = [np.sqrt(S[i]) * U[:, i].reshape(problem.shape[0:2], order='F')
                    for i in range(RANK)]
        # just using V.T instead of V.H
        rfactors = [np.sqrt(S[i]) * V.T[:, i].reshape(problem.shape[2:4], order='F')
                    for i in range(RANK)]
        out.recovered = operators.DyadsOperator(lfactors, rfactors)
    else:
        temp = np.array(A.value).reshape(problem.shape, order='F')
        out.recovered = operators.ArrayOperator(temp)
    out.setup_time = end_setup - start
    out.solve_time = end_solve - end_setup
    out.objval = prob.objective.value
    return out

SOLVERS = {'altmin': altminsolve, 'mat': matsolve, 'sdp': sdpsolve}
"""dict: Matches solver functions (values) with string labels (keys)."""
