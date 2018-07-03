__all__ = ['NotReadyModelError', 'MissingComponentsWarning']


class MissingComponentsWarning(Warning):
    """Warning class to raise if some important but unnecessary components not set

    This class inherits from Warning.
    """


class NotReadyModelError(ValueError, AttributeError):
    """Exception class to raise if model was not loaded or trained

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """
