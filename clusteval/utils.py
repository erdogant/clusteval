"""Utilities."""
import logging
import numpy as np
from typing import Union  # List, Tuple
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logger = logging.getLogger('')
for handler in logger.handlers[:]: #get rid of existing old handlers
    logger.removeHandler(handler)
console = logging.StreamHandler()
formatter = logging.Formatter('[clusteval] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger(__name__)


# %% Normalize.
def normalize_size(getsizes, minscale: Union[int, float] = 0.5, maxscale: Union[int, float] = 4, scaler: str = 'zscore'):
    """Normalize values between minimum and maximum value.

    Parameters
    ----------
    getsizes : input array
        Array of values that needs to be scaled.
    minscale : Union[int, float], optional
        Minimum value. The default is 0.5.
    maxscale : Union[int, float], optional
        Maximum value. The default is 4.
    scaler : str, optional
        Type of scaler. The default is 'zscore'.
            * 'zscore'
            * 'minmax'

    Returns
    -------
    getsizes : array-like
        scaled values between min-max.

    """
    # Instead of Min-Max scaling, that shrinks any distribution in the [0, 1] interval, scaling the variables to
    # Z-scores is better. Min-Max Scaling is too sensitive to outlier observations and generates unseen problems,

    # Set sizes to 0 if not available
    getsizes[np.isinf(getsizes)]=0
    getsizes[np.isnan(getsizes)]=0

    # out-of-scale datapoints.
    if scaler == 'zscore' and len(np.unique(getsizes)) > 3:
        getsizes = (getsizes.flatten() - np.mean(getsizes)) / np.std(getsizes)
        getsizes = getsizes + (minscale - np.min(getsizes))
    elif scaler == 'minmax':
        try:
            from sklearn.preprocessing import MinMaxScaler
        except:
            raise Exception('sklearn needs to be pip installed first. Try: pip install scikit-learn')
        # scaling
        getsizes = MinMaxScaler(feature_range=(minscale, maxscale)).fit_transform(getsizes).flatten()
    else:
        getsizes = getsizes.ravel()
    # Max digits is 4
    getsizes = np.array(list(map(lambda x: round(x, 4), getsizes)))

    return getsizes


# %%
def _compute_embedding(X, logger):
    logger.info('Compute t-SNE embedding.')
    perplexity = np.minimum(X.shape[0]-1, 30)
    X = TSNE(n_components=2, init='random', perplexity=perplexity).fit_transform(X)
    return X


# %%
def compute_embedding(self, X, embedding, logger):
    if (embedding=='tsne'):
        if hasattr(self, 'results') and (self.results.get('xycoord') is not None):
            logger.info('Retrieving previously computed [%s] embedding.' %(embedding))
            X = self.results['xycoord']
        else:
            X = _compute_embedding(X, logger)
    else:
        logger.info('Coordinates (x, y) are set based on the first two features.')
        X = X[:, :2]

    return X


# %%
def set_font_properties(font_properties):
    return {**{'size_title': 18, 'size_x_axis': 18, 'size_y_axis': 18, 'fontcolor': '#000000', 'axis_color': '#000000'}, **font_properties}


# %%
def init_figure(fig, ax, dpi, figsize, visible):
    """Initialize figure."""
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # Adjust global font settings
        # plt.rcParams['font.size'] = np.maximum(fontsize, 1)  # Set an appropriate font size
        ax = fig.add_subplot()

    if fig is not None:
        fig.set_visible(visible)

    return fig, ax

# %%
def init_logger():
    logger = logging.getLogger('')
    for handler in logger.handlers[:]: #get rid of existing old handlers
        logger.removeHandler(handler)
    console = logging.StreamHandler()
    formatter = logging.Formatter('[clusteval] >%(levelname)s> %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger = logging.getLogger(__name__)
    return logger


# %%
def set_logger(verbose: [str, int] = 'info'):
    """Set the logger for verbosity messages.

    Parameters
    ----------
    verbose : [str, int], default is 'info' or 20
        Set the verbose messages using string or integer values.
        * [0, 60, None, 'silent', 'off', 'no']: No message.
        * [10, 'debug']: Messages from debug level and higher.
        * [20, 'info']: Messages from info level and higher.
        * [30, 'warning']: Messages from warning level and higher.
        * [50, 'critical']: Messages from critical level and higher.

    Returns
    -------
    None.

    > # Set the logger to warning
    > set_logger(verbose='warning')
    > # Test with different messages
    > logger.debug("Hello debug")
    > logger.info("Hello info")
    > logger.warning("Hello warning")
    > logger.critical("Hello critical")

    """
    # Set 0 and None as no messages.
    if (verbose==0) or (verbose is None):
        verbose=60
    # Convert str to levels
    if isinstance(verbose, str):
        levels = {'silent': 60,
                  'off': 60,
                  'no': 60,
                  'debug': 10,
                  'info': 20,
                  'warning': 30,
                  'critical': 50}
        verbose = levels[verbose]

    # Show examples
    logger.setLevel(verbose)

# %%
def get_logger():
    """Return logger status."""
    return logger.getEffectiveLevel()

# %%
def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)
