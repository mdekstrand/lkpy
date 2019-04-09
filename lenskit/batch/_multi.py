import logging
import pathlib
import collections
import json
import warnings
from copy import copy

from joblib import Parallel, delayed

import pandas as pd

from ..algorithms import Predictor
from .. import topn, util
from ._recommend import recommend
from ._predict import predict

try:
    import fastparquet
except ImportError:
    fastparquet = None

_logger = logging.getLogger(__name__)

_AlgoRec = collections.namedtuple('_AlgoRec', [
    'algorithm',
    'parallel',
    'attributes'
])
_DSRec = collections.namedtuple('_DSRec', [
    'dataset',
    'candidates',
    'attributes'
])


class MultiEval:
    """
    A runner for carrying out multiple evaluations, such as parameter sweeps.

    Args:
        path(str or :py:class:`pathlib.Path`):
            the working directory for this evaluation.
            It will be created if it does not exist.

        predict(bool):
            whether to generate rating predictions.

        recommend(int):
            the number of recommendations to generate per user. Any false-y value (``None``,
            ``False``, ``0``) will disable top-n. The literal value ``True`` will generate
            recommendation lists of unlimited size.

        candidates(function):
            the default candidate set generator for recommendations.  It should take the
            training data and return a candidate generator, itself a function mapping user
            IDs to candidate sets.  If ``None``, each recommender's default candidate set
            is used.

            .. note:: **Changed in 0.7** to default to ``None``.

        n_jobs(None, int, tuple, or dict):
            The number of parallel jobs to use for evaluation.  This takes a few possible
            forms:

            ``None``
                JobLib's defaults are used.  This will be sequential evaluation, unless
                the call to :meth:`run` is wrapped in a :func:`joblib.parallel_backend`
                context specifying a job count.

            *N*
                When ``n_jobs`` is a single integer, it is passed as the job count to
                :cls:`joblib.Parallel` for running the experiment jobs in this eval
                experiment, and is also passed as the `nprocs` argument to
                :func:`batch.predict` and :func:`batch.recommend`.

            (*M*, *N*)
                A tuple of integers provides separate job counts for outer experiment jobs
                and recommendation or prediction within a job.  *M* is used for the
                MultiEval parallel loop, and *N* is passed to :func:`batch.predict` and
                :func:`batch.recomend`.  Either value can also be ``None``.

            ``dict``
                A dictionary with zero or more of the keys ``exp``, ``predict``, and
                ``recommend`` provides full control over individual parallelism levels.

            .. note::
                Parallel experiments work better with on-disk data, either passing
                paths to :meth:`add_dataset` or calling :meth:`persist_data`.

            .. note:: ``nprocs`` is accepted as a deprecated alias for ``n_jobs=(1, nprocs)``.

        combine(bool):
            Whether to combine output; if ``False``, output will be left in separate files, if
            ``True``, it will be in a single set of files (runs, recommendations, and predictions).
    """

    def __init__(self, path, predict=True, recommend=100, *, candidates=None,
                 n_jobs=None, combine=True, **kwargs):
        self.workdir = pathlib.Path(path)
        self.predict = predict
        self.recommend = recommend
        self.candidate_generator = candidates
        self.algorithms = []
        self.datasets = []

        if n_jobs is None and 'nprocs' in kwargs:
            warnings.warn('nprocs is deprecated, use n_jobs', DeprecationWarning)
            n_jobs = (1, kwargs['nprocs'])

        if isinstance(n_jobs, dict):
            self.eval_jobs = n_jobs.get('exp', None)
            self.predict_jobs = n_jobs.get('predict', None)
            self.recommend_jobs = n_jobs.get('recommend', None)
        elif isinstance(n_jobs, tuple):
            outer, inner = n_jobs
            self.eval_jobs = outer
            self.predict_jobs = inner
            self.recommend_jobs = inner
        else:
            self.eval_jobs = n_jobs
            self.predict_jobs = n_jobs
            self.recommend_jobs = n_jobs

        self.combine_output = combine
        self._is_flat = True

    @property
    def run_csv(self):
        return self.workdir / 'runs.csv'

    @property
    def run_file(self):
        return self.workdir / 'runs.parquet'

    @property
    def preds_file(self):
        return self.workdir / 'predictions.parquet'

    @property
    def recs_file(self):
        return self.workdir / 'recommendations.parquet'

    def add_algorithms(self, algos, parallel=False, attrs=[], **kwargs):
        """
        Add one or more algorithms to the run.

        Args:
            algos(algorithm or list): the algorithm(s) to add.
            parallel(bool):
                if ``True``, allow this algorithm to be trained in parallel with others.
            attrs(list of str):
                a list of attributes to extract from the algorithm objects and include in
                the run descriptions.
            kwargs:
                additional attributes to include in the run descriptions.
        """

        if not isinstance(algos, collections.Iterable):
            algos = [algos]

        for algo in algos:
            aa = {'AlgoClass': algo.__class__.__name__, 'AlgoStr': str(algo)}
            aa.update(kwargs)
            for an in attrs:
                aa[an] = getattr(algo, an, None)

            self.algorithms.append(_AlgoRec(algo, parallel, aa))

    def add_datasets(self, data, name=None, candidates=None, **kwargs):
        """
        Add one or more datasets to the run.

        Args:
            data:
                The input data set(s) to run. Can be one of the following:

                * A tuple of (train, test) data.
                * An iterable of (train, test) pairs, in which case the iterable
                  is not consumed until it is needed.
                * A function yielding either of the above, to defer data load
                  until it is needed.

                Data can be either data frames or paths; paths are loaded after
                detection using :py:func:`util.read_df_detect`.

            kwargs:
                additional attributes pertaining to these data sets.
        """

        attrs = {}
        if name is not None:
            attrs['DataSet'] = name
        attrs.update(kwargs)

        # special-case lists to keep multis flat
        if isinstance(data, list):
            for part, e in enumerate(data):
                self.add_datasets(e, name, candidates,
                                  Partition=part+1, **kwargs)
            return

        if not isinstance(data, tuple):
            self._is_flat = False

        self.datasets.append(_DSRec(data, candidates, attrs))

    def persist_data(self):
        """
        Persist the data for an experiment, replacing in-memory data sets with file names.
        Once this has been called, the sweep can be pickled.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)
        ds2 = []
        for i, (ds, cand_f, ds_attrs) in enumerate(self._flat_datasets()):
            train, test = ds
            if isinstance(train, pd.DataFrame):
                fn = self.workdir / 'ds{}-train.feather'.format(i+1)
                _logger.info('serializing to %s', fn)
                train.to_feather(fn)
                train = fn
            if isinstance(test, pd.DataFrame):
                fn = self.workdir / 'ds{}-test.feather'.format(i+1)
                _logger.info('serializing to %s', fn)
                test.to_feather(fn)
                test = fn
            ds2.append(((train, test), cand_f, ds_attrs))
        self.datasets = ds2
        self._is_flat = True

    def _normalize_ds_entry(self, entry):
        # normalize data set to be an iterable of tuples
        ds, cand_f, attrs = entry
        if callable(ds):
            ds = ds()
        if isinstance(ds, tuple):
            yield _DSRec(ds, cand_f, attrs)
        else:
            yield from (_DSRec(dse, cand_f, dict(Partition=part+1, **attrs))
                        for (part, dse) in enumerate(ds))

    def _flat_datasets(self):
        for entry in self.datasets:
            yield from self._normalize_ds_entry(entry)

    def _read_data(self, df):
        if isinstance(df, str) or isinstance(df, pathlib.Path):
            _logger.info('reading from %s', df)
            return util.read_df_detect(df)
        else:
            return df

    def _flat_runs(self):
        for dse in self._flat_datasets():
            for arec in self.algorithms:
                yield (dse, arec)

    def run_count(self):
        "Get the number of runs in this evaluation."
        if self._is_flat:
            nds = len(self.datasets)
        else:
            _logger.warning('attempting to count runs in a non-flattened evaluation')
            nds = len(list(self._flat_datasets()))
        return nds * len(self.algorithms)

    def run(self, runs=None, *, progress=None):
        """
        Run the evaluation.

        Args:
            runs(int or set-like):
                If provided, a specific set of runs to run.  Useful for splitting
                an experiment into individual runs.  This is a set of 1-based run
                IDs, not 0-based indexes.

            progress:
                A :py:func:`tqdm.tqdm`-compatible progress function.
        """

        if runs is not None and self.combine_output:
            raise ValueError('Cannot select runs with combined output')

        if runs is not None and not isinstance(runs, collections.Iterable):
            runs = [runs]

        self.workdir.mkdir(parents=True, exist_ok=True)

        iter = self._flat_runs()
        n_jobs = self.exp_jobs
        if runs is not None and len(runs) == 1:
            n_jobs = 1
        loop = Parallel(n_jobs=n_jobs, batch_size=1)

        with loop:
            if progress is not None:
                if loop._effective_n_jobs() > 1:
                    warnings.warn('progress does not work well with parallel experiments')
                n = self.run_count() if self._is_flat else None
                iter = progress(iter, total=n)

            iter = enumerate(iter)

            oc = self.combine_output
            try:
                collect = False
                if loop._effective_n_jobs() > 1:
                    # will run in parallel
                    collect = self.combine_output
                    self.combine_output = False

                runs = loop(delayed(self._run_job)(i, dsr, ar)
                            for (i, (dsr, ar)) in iter
                            if runs is None or i+1 in runs)

                if collect:
                    _logger.info('collecting output from parallel runs')
                    self.collect_results(len(runs), write_runs=False, delete=True)
                    runs.to_csv(self.run_csv, index=False)
                    runs.to_parquet(self.run_file)

            finally:
                self.combine_output = oc

    def _run_data(self, i, dsrec, arec):
        run_id = i + 1

        ds, cand_f, ds_attrs = dsrec
        if cand_f is None:
            cand_f = self.candidate_generator
        train, test = ds
        train = self._read_data(train)
        test = self._read_data(test)

        ds_name = ds_attrs.get('DataSet', None)
        ds_part = ds_attrs.get('Partition', None)
        cand = cand_f(train)

        _logger.info('starting run %d: %s on %s:%s', run_id, arec.algorithm,
                     ds_name, ds_part)
        run = self._run_algo(run_id, arec, (train, test, ds_attrs, cand))
        _logger.info('finished run %d: %s on %s:%s', run_id, arec.algorithm,
                     ds_name, ds_part)
        return run

    def _run_algo(self, run_id, arec, data):
        train, test, dsp_attrs, cand = data

        run = {'RunId': run_id}
        run.update(dsp_attrs)
        run.update(arec.attributes)

        algo, train_time = self._train_algo(arec.algorithm, train)
        run['TrainTime'] = train_time

        preds, pred_time = self._predict(run_id, algo, test)
        run['PredTime'] = pred_time
        self._write_results('predictions', preds, run_id)

        recs, rec_time = self._recommend(run_id, algo, test, cand)
        run['RecTime'] = rec_time
        self._write_results('recommendations', recs, run_id)

        return run

    def _train_algo(self, algo, train):
        watch = util.Stopwatch()
        _logger.info('training algorithm %s on %d ratings', algo, len(train))
        # clone the algorithm in case some cannot be reused
        clone = util.clone(algo)
        clone.fit(train)
        watch.stop()
        _logger.info('trained algorithm %s in %s', algo, watch)
        return clone, watch.elapsed()

    def _predict(self, rid, algo, test):
        if not self.predict:
            return None, None
        if not isinstance(algo, Predictor):
            return None, None

        watch = util.Stopwatch()
        _logger.info('generating %d predictions for %s', len(test), algo)
        preds = predict(algo, test, nprocs=self.predict_jobs)
        watch.stop()
        _logger.info('generated predictions in %s', watch)
        preds['RunId'] = rid
        preds = preds[['RunId', 'user', 'item', 'rating', 'prediction']]
        return preds, watch.elapsed()

    def _recommend(self, rid, algo, test, candidates):
        if not self.recommend:  # if recommend is any false-y val (0, None, False), turn off recs
            return None, None
        elif self.recommend is True:  # special value True means unlimited
            nrecs = None
        else:  # recommend has rec size
            nrecs = self.recommend

        watch = util.Stopwatch()
        users = test.user.unique()
        _logger.info('generating recommendations for %d users for %s', len(users), algo)
        recs = recommend(algo, users, nrecs, candidates,
                         nprocs=self.recommend_jobs)
        watch.stop()
        _logger.info('generated recommendations in %s', watch)
        recs['RunId'] = rid
        return recs, watch.elapsed()

    def _write_run(self, run, run_data):
        if self.combine_output:
            run_df = pd.DataFrame(run_data)
            # overwrite files to show progress
            run_df.to_csv(self.run_csv, index=False)
            run_df.to_parquet(self.run_file, compression=None)
        else:
            rf = self.workdir / 'run-{}.json'.format(run['RunId'])
            with rf.open('w') as f:
                json.dump(run, f)

    def _write_results(self, name, df, run_id):
        if df is None:
            return

        if self.combine_output:
            out = self.workdir / '{}.parquet'.format(name)
            _logger.info('run %d: writing results to %s', run_id, out)
            append = run_id > 1
            util.write_parquet(out, df, append=append)
        else:
            out = self.workdir / '{}-{}.parquet'.format(name, run_id)
            _logger.info('run %d: writing results to %s', run_id, out)
            df.to_parquet(out)

    def collect_results(self, n=None, *, write_runs=True, delete=False):
        """
        Collect the results from non-combined runs into combined output files.
        """

        oc = self.combine_output
        try:
            self.combine_output = True
            if n is None:
                n = self.run_count()

            if write_runs:
                runs = (self._read_json('run-{}.json', i+1) for i in range(n))
                runs = pd.DataFrame.from_records(runs)
                runs.to_parquet(self.run_file)
                runs.to_csv(self.run_csv, index=False)

            for i in range(n):
                pred_f = self.workdir / 'predictions-{}.parquet'.format(i+1)
                if pred_f.exists():
                    preds = pd.read_parquet(pred_f)
                    self._write_results('predictions', preds, i+1)
                    if delete:
                        pred_f.unlink()

                rec_f = self.workdir / 'recommendations-{}.parquet'.format(i+1)
                if rec_f.exists():
                    recs = pd.read_parqet(rec_f)
                    self._write_results('recommendations', recs, i+1)
                    if delete:
                        rec_f.unlink()

        finally:
            self.combine_output = oc

    def _read_json(self, name, *args):
        fn = self.workdir / name.format(*args)
        if not fn.exists():
            _logger.warning('file %s does not exist', fn)
            return {}

        with fn.open('r') as f:
            return json.load(f)

    def __getstate__(self):
        if not self._is_flat:
            _logger.warning('attempting to pickle non-flattened experiment')
        state = copy(self.__dict__)
        # clone the algorithms to only pickle their parameters
        state['algorithms'] = [a._replace(algorithm=util.clone(a.algorithm))
                               for a in self.algorithms]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
