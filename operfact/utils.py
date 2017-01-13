# -*- coding: utf-8 -*-
"""Utility functions.

This module contains various utility functions that help with:

* Generating random matrices
* Running numerical experiments

"""

import csv
import gzip
import numpy as np
import multiprocessing as mp
import time


def rand_orthmat(shape):
    r"""Generates a random orthogonal matrix.

    Given the shape tuple :math:`(m, n)`, the function will return a matrix
    :math:`Q \in \mathbb{R}^{m\times n}` such that :math:`QQ^T = I_m` if
    :math:`m<n` and :math:`Q^TQ = I_n` otherwise.

    The method ensures that the resulting random matrices are indeed drawn from
    the Haar measure on orthogonal matrices.

    """
    flip = True if (shape[0] < shape[1]) else False
    if flip:
        shape = (shape[1], shape[0])
    Q, R = np.linalg.qr(np.random.normal(size=shape), mode='reduced')
    d = np.sign(np.diag(R))
    Q = Q*d
    return Q.T if flip else Q


def rand_lowrankmat(shape, rank):
    r"""Generates a random low rank matrix.

    Given a shape tuple :math:`(m, n)` and a rank :math:`r`, this function
    returns a random matrix :math:`\mathbf{A} \in \mathbb{R}^{m\times n}` with
    rank :math:`r` such that :math:`\mathbf{A} = \mathbf{U}\mathbf{V}^T`, where
    :math:`\mathbf{U} \in \mathbb{R}^{m\times r}` and
    :math:`\mathbf{V} \in \mathbb{R}^{n\times r}` are random orthogonal
    matrices.

    """
    U = rand_orthmat((shape[0], rank))
    V = rand_orthmat((shape[1], rank))
    return U.dot(V.T)


def rand_sparsemat(shape, nzfrac):
    r"""Generates a random sparse matrix.

    Given a shape tuple :math:`(m, n)` and a fraction :math:`\rho \in [0,1]`,
    the function returns a matrix :math:`\mathbf{A} \in \mathbb{R}^{m\times n}`
    such that the number of nonzeros in :math:`\mathbf{A}` is
    :math:`\lfloor mn\rho \rfloor`, and those nonzero entries take the values
    :math:`\{ -1, +1 \}` with equal probability.

    Note
    ----
    Even though the generated matrix is sparse, we return a dense matrix. The
    use case of this function is to generate test data for operator
    factorization problems. In our testing, we assume that storage of dense
    operators is not an issue. Furthermore, we gain little in terms of
    performance by specifying our test data as sparse matrices. The solvers use
    dense variables and parameters.

    """
    nnz = int(np.floor(np.prod(shape) * nzfrac))
    ixs = np.random.choice(np.prod(shape), nnz, replace=False)
    out = np.zeros((np.prod(shape),))
    out[ixs] = ((-1.0*np.ones(nnz)) ** np.random.randint(0, 2, nnz))
    return out.reshape(shape, order='F')


def rand_gaussianmat(shape, normalize):
    """Generates a random matrix with independent Gaussian entries.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the resulting random matrix.
    normalize : bool
        Controls the variance of the entries of the matrix.

        If True, the entries have variance :math:`1/mn` where :math:`(m, n)` is
        the shape of the matrix. This normalization results in random matrices
        with expected squared Frobenius norm of 1.

        If False, the entries have variance 1.

    Returns
    -------
    numpy.ndarray
        The random matrix.

    """
    return np.random.normal(size=shape,
                            scale=1/np.sqrt(np.prod(shape)) if normalize else 1.0)


def rand_sparsegaussianmat(shape, nzfrac, normalize=False):
    """Generates a random sparse matrix with independent Gaussian entries.

    This function is similar to `rand_sparsemat` except that the nonzero entries
    of the matrix are drawn independently from the Gaussian distribution.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the resulting random matrix.
    nzfrac : float
        A number between 0 and 1 giving the fraction of nonzero entries desired.
    normalize : bool
        Controls the variance of the entries of the matrix.

        If True, the entries have variance :math:`1/s` where s is the number of
        nonzero entires in the matrix. This normalization results in random
        matrices with expected squared Frobenius norm of 1.

        If False, the entries have variance 1.

    Returns
    -------
    numpy.ndarray
        A _dense_ array containing our random matrix.

        See the note in the `rand_sparsemat` method for the reasoning behind
        returning dense versions of sparse objects.

    """
    nnz = int(np.floor(np.prod(shape) * nzfrac))
    ixs = np.random.choice(np.prod(shape), nnz, replace=False)
    out = np.zeros((np.prod(shape),))
    out[ixs] = np.random.normal(size=(nnz,),
                                scale=1/np.sqrt(nnz) if normalize else 1.0)
    return out.reshape(shape)


def rand_signmat(shape, normalize):
    r"""Generates a random sign matrix.

    The resulting matrix has entries of identical magnitude with sign chosen
    uniformly at random. The magnitude of the entries is controlled by the
    `normalize` option.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the resulting random matrix.
    normalize : bool
        Controls the magnitude of the entries of the matrix.

        If True, the entries have magnitude :math:`1/\sqrt{mn}`, where
        :math:`(m, n)` is the shape of the matrix. This ensures that the matrix
        has Frobenius norm 1.

        If False, the entries have magnitude 1.

    Returns
    -------
    numpy.ndarray
        The random matrix.

    """
    return ((-1.0*np.ones(shape)) ** np.random.randint(0, 2, shape)) * (1/np.sqrt(np.prod(shape)) if normalize else 1.0)


def istol(obj, index=None):
    """Checks whether an object is a tuple or a list.

    This function has two different behaviors depending on whether or not the
    optional parameter `index` is provided. In normal operation it returns the
    Boolean result of checking the type of `obj`.

    If an integer `index` is provided, the function returns `obj[index]` if
    `obj` is a tuple or list and `obj` otherwise.

    Parameters
    ----------
    obj
        The object to check.
    index : Optional[int]
        The index of `obj` to return if `obj` is a tuple or list.

    Returns
    -------
    object
        See the description for the possible return values.

    """
    if index is None:
        return isinstance(obj, (tuple, list))
    else:
        return obj[index] if istol(obj) else obj


def experiment(experiment_fcn, params, ntrials=1, nprocs=1, seed=None,
               return_rows=True, process_fcn=None, maxtasksperchild=None):
    """Performs a numerical experiment.

    This driver function facilitates running numerical experiments---possibly
    in parallel---by repeatedly calling a user-provided function with
    user-provided parameters.

    We expect that the results of a numerical experiment will be a list of rows
    (as you would see in a spreadsheet) that record the independent and
    dependent variables for each test case in the experiment.

    Note
    ----
    Running experiments in parallel uses the `forkserver` context of the
    Python `multiprocessing` module.

    This requires Python 3.4+.

    Parameters
    ----------
    experiment_fcn : function
        The function that performs the actual numerical experiment.

        We expect that this function takes one required parameter that can be
        used to specify the test cases of the experiment. We further expect that
        the function takes a keyword argument `seed` that will initialize the
        random number generator (if used) and a keyword argument `redirect` that
        is True if `experiment_fcn` should redirect stdout and stderr to files.
    params : object or list of objects
        The parameters to pass to `experiment_fcn`.

        If this is a list of objects, we handle calling `experiment_fcn` for
        each of the objects in the list.

        In our implementations, the `params` objects usually end up being
        dictionaries that `experiment_fcn` will use to perform multiple test
        cases.
    ntrials : Optional[int]
        The number of trials of the complete experiment to perform.

        Default value is 1.
    nprocs : Optional[int]
        The number of processors to use for conducting the experiment.

        The total number of tasks is given by `ntrials*len(params)`. This
        quantity is the maximum number of processors that we can utilize.

        If `nprocs` is set to 1, then the experiment will not use any
        functionality from the `multiprocessing` package.

        Default value is 1.
    seed : Optional[numeric]
        A base seed for the random number generators used in the experiment.

        The first call to `experiment_fcn` is passed `seed` and for each
        subsequent call we increment `seed` by 1.

        If no value is provided, the base seed is set using the current time.

        Default value is None.
    return_rows : Optional[bool]
        Determines whether the function should return the results of all test.

        Default value is True.
    process_fcn : Optional[function]
        A function applied to each of the returned lists from `experiment_fcn`.

        After a call to `experiment_fcn` its returned value is passed as the
        argument to `process_fcn`. This can be used in conjunction with setting
        `return_rows` False to do any processing (e.g., saving) of experimental
        results immediately without having to store all of the results from the
        individual calls to `experiment_fcn` in memory.
    maxtasksperchild : Optional[int]
        The number of tasks a worker process can handle before respawning.

        Setting this value sets the corresponding option in the multiprocessing
        Pool object. It allows for the resources used by a worker process to be
        freed periodically. If set to None, the worker processes will remain
        open for the duration of the experiment.

        Default value is None.

    Returns
    -------
    list
        A list of the results from each call to `experiment_fcn`.

        If `return_rows` is False this list will be empty.

    """
    if ~return_rows and (process_fcn is None):
        print('Warning: return_rows is false but no process_fcn provided')
    time_start = int(time.time())
    seed = time_start if seed is None else seed
    paramslist = params if istol(params) else (params, )
    rows = []
    curr_job = 0
    total_jobs = len(paramslist)*ntrials

    def process_result(result):
        nonlocal curr_job
        curr_job += 1
        if process_fcn is not None:
            process_fcn(result)
        if return_rows:
            rows.extend(result)
        print('job {0}/{1} at {2}s'.format(curr_job, total_jobs,
                                           time.time() - time_start))

    def error_result(err):
        nonlocal curr_job
        curr_job += 1
        print('job {0}/{1} ERROR {2}: {3}'.format(curr_job, total_jobs, type(err), err))
        raise err

    if nprocs > 1:
        fs = mp.get_context('forkserver')
        pool = fs.Pool(processes=nprocs, maxtasksperchild=maxtasksperchild)

        jobs = [pool.apply_async(experiment_fcn, args=(params,),
                                 kwds={'seed': seed+i, 'redirect': True},
                                 callback=process_result,
                                 error_callback = error_result)
                for i in range(ntrials)
                for params in paramslist]
        pool.close()
        pool.join()
    else:
        for params in paramslist:
            for i in range(ntrials):
                process_result(experiment_fcn(params, seed=seed+i, redirect=False))

    time_total = time.time() - time_start
    print('total time: {0}s'.format(time_total))

    return rows


def experiment_csv(filename, fieldnames, experiment_fcn, params,
                   ntrials=1, nprocs=1, seed=None, return_rows=True,
                   maxtasksperchild=None):
    """Performs a numerical experiment and writes the result to a CSV file.

    This function uses the functionality of `experiment` to perform a numerical
    experiment while writing the results to comma-separated value (CSV) file.

    The documentation of `experiment` explains its action and expected
    parameters. Refer to it for the parameters not described below.

    Parameters
    ----------
    filename : str
        The path where the CSV file should be saved.
    fieldnames : list
        A list giving the names of the columns in the CSV.

        To write a CSV file, it is required that `experiment_fcn` return a list
        of dictionaries whose keys exactly match the list `fieldnames`. This
        matches the model where we consider the result of `experiment_fcn` as a
        list of rows in a spreadsheet.
    see `experiment` for the remaining parameters

    Returns
    -------
    list
        A list of the results from each call to `experiment_fcn`.

        If `return_rows` is False this list will be empty.

    """
    csvfile = gzip.open(filename, 'wt', 9)
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    rows = experiment(experiment_fcn, params, ntrials, nprocs, seed,
                      return_rows, writer.writerows)
    csvfile.close()
    return rows
