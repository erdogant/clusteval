Saving
##########

Saving and loading can be accomplished with two functions: function :func:`clusteval.clusteval.clusteval.save` and function :func:`clusteval.clusteval.clusteval.load`
Below we illustrate how to save and load models.

Saving a learned model can be done using the function :func:`clusteval.clusteval.clusteval.save`:

.. code:: python

    from clusteval import clusteval

    Save model
    ce.save('learned_model_v1')



Loading
##########

Loading a learned model can be done using the function :func:`clusteval.clusteval.clusteval.load`:

.. code:: python

    from clusteval import clusteval

    # Load model
    model = ce.load(model, 'learned_model_v1')



.. include:: add_bottom.add