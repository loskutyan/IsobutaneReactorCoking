__all__ = ['NotReadyModelError']


class NotReadyModelError(ValueError, AttributeError):
    """Exception class to raise if model was not loaded or trained

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """
