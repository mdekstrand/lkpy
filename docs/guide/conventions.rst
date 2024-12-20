.. _conventions:

Component Conventions
=====================

The components shipped with LensKit follow certain conventions to make their
configuration and operation consistent and predictable. We encourage you to
follow these conventions in your own code as well.

List Length
~~~~~~~~~~~

Ranking and selection components typically provide two ways to specify the
desired list length: a configuration option (constructor parameter) and a
runtime parameter (input), both named ``n`` and type ``int | None``.  This
allows list length to be baked into a pipeline configuration, and also allows
that length to be specified or overridden at runtime.  If both lengths are
specified, the runtime length takes precedence.

.. _rng:

Random Seeds
~~~~~~~~~~~~

.. _SPEC 7: https://scientific-python.org/specs/spec-0007/

LensKit components follow `SPEC 7`_ for specifying random number seeds.
Components that use randomization (either at runtime, or to set initial
conditions for training) have a constructor parameter `rng` that takes either a
:class:`~numpy.random.Generator` or seed material.  If you want reproducible
stochastic pipelines, configure the random seeds for your components.

This convention is also followed for other LensKit code, such as the `data
splitting support <./splitting>`_.

.. important::

    If you specify random seeds, we strongly recommend specifying seeds instead of
    generators, so that the seed can be included in serialized configurations.

.. versionchanged:: 2025.1

    Now that `SPEC 7`_ has standardized RNG seeding across the scientific Python
    ecosystem, we use that with some lightweight helpers in the
    :mod:`lenskit.util.random` module instead of using SeedBank.

LensKit extends SPEC 7 with a global RNG that components can use as a fallback,
to make it easier to configure system-wide generation for things like tests.
This is configured with :func:`~lenskit.util.random.set_global_rng`.

When implementing a component that uses randomness in its training, we recommend
deferring conversion of the provided RNG into an actual generator until
model-training time, so that serializing an untrained pipeline or its
configuration includes the original seed instead of the resulting generator.
When using the RNG to create initial state for e.g. training a model with
PyTorch, it can be useful to create that state in NumPy and then convert to a
tensor, so that components are consistent in their random number generation
behavior instead of having variation between NumPy and other backends.
Components can use the :func:`~lenskit.util.random_generator` function to
convert seed material or a generator into a NumPy generator, falling back to the
global RNG if one is specified.

Derived Seeds
-------------

Recommendation provides a particular challenge for deterministic random behavior
in the face of multiple recommendation requests, particularly when those
requests are parallelized, resulting in nondeterministic arrival orders.

To handle this, LensKit components that randomize responses at runtime (such as
:class:`~lenskit.basic.RandomSelector` and :class:`~lenskit.basic.SoftmaxRanker`)
support *derivable RNGs*.  They are selected by passing the string ``'user'`` as
the RNG seed, or a tuple of the form ``(seed, 'user')``.  When configured with
such a seed, the component will deterministically derive a seed for each request
based on the request's userID.  This means that, for the same set of items and
starting seed (and LensKit, NumPy, etc. versions),
:class:`~lenskit.basic.RandomSelector` will return the *same* items for a given
user, and different items for other users, regardless of the order in which
those users are processed.

.. seealso:: :func:`lenskit.util.derivable_rng`
