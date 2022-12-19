import inspect


def printError(expr: bool):
    if expr:
        current_frame = inspect.currentframe()
        print("file: {}, line: {}".format(
            current_frame.f_back.f_code.co_filename,
            current_frame.f_back.f_lineno))
        raise RuntimeError()
