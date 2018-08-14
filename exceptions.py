__all__ = ['MissingComponentsWarning', 'MissingTags', 'MissingModel','NotReadyModelError']


class MissingComponentsWarning(UserWarning):
    """Custom warning to notify user if some important but unnecessary components not set

    This class inherits from UserWarning.
    """


class MissingTags(KeyError):
    """Exception class to raise if some tags are missing in data

    This class inherits from KeyError.
    """


class MissingModel(KeyError):
    """Exception class to raise if no model for sensor defined
    and no common model for its plate and reactor found

    This class inherits from KeyError.
    """


class NotReadyModelError(ValueError, AttributeError):
    """Exception class to raise if model was not loaded or trained

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """
