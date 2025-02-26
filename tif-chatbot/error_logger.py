import sys
import stackprinter

from loguru import logger


def reformat(record):
    format_ = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name: >11}</cyan>:<cyan>{line: <3}</cyan> - <level>{message}</level>\n")

    if record["exception"] is not None:
        record["extra"]["stack"] = stackprinter.format(record["exception"])
        format_ += "{extra[stack]}\n"

    return format_


def configure_error_logger():
    logger.remove()
    logger.add(sys.stderr, level='DEBUG', enqueue=True, format=reformat)
