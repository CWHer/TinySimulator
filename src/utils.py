import inspect
import logging


def printError(expr: bool):
    if expr:
        current_frame = inspect.currentframe()
        print("file: {}, line: {}".format(
            current_frame.f_back.f_code.co_filename,
            current_frame.f_back.f_lineno))
        raise RuntimeError()


def printErrorMsg(expr: bool, msg: str):
    if expr:
        current_frame = inspect.currentframe()
        print("file: {}, line: {}".format(
            current_frame.f_back.f_code.co_filename,
            current_frame.f_back.f_lineno))
        raise RuntimeError(msg)


def printWarnMsg(expr: bool, msg: str):
    if expr:
        current_frame = inspect.currentframe()
        print("file: {}, line: {}".format(
            current_frame.f_back.f_code.co_filename,
            current_frame.f_back.f_lineno))
        logging.warning(msg)
