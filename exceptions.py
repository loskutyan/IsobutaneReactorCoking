__all__ = ['NotReadyModelError', 'MissingComponentsWarning']


class MissingComponentsWarning(UserWarning):
    """Custom warning to notify user if some important but unnecessary components not set

    This class inherits from UserWarning.
    """


class NotReadyModelError(ValueError, AttributeError):
    """Exception class to raise if model was not loaded or trained

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """
