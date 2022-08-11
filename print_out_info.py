import logging


def print_out_info(
    func
):

    """
    Decorator that encapsulates common code pattern in functions. Prompts when a function is starting and ending.
    If function crashes - it raises the same exception, but also log the information about the failure.
    """

    def wrapper(*args, **kwargs):
        try:
            logging.info(f"{func.__name__}: start")
            result = func(*args, **kwargs)
            logging.info(f"{func.__name__}: end")
            return result
        except Exception as e:
            logging.error(f"{func.__name__}: failed")
            raise e
    return wrapper
