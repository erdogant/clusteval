import logging

logger = logging.getLogger('')
for handler in logger.handlers[:]: #get rid of existing old handlers
    logger.removeHandler(handler)
console = logging.StreamHandler()
formatter = logging.Formatter('[clusteval] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger(__name__)

# %%
def set_font_properties(font_properties):
    return {**{'size_title': 20, 'size_x_axis': 16, 'size_y_axis': 16}, **font_properties}

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
def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)
