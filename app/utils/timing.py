import time
import logging
from contextlib import contextmanager

logger = logging.getLogger("uvicorn.error")

@contextmanager
def timer(label:str):
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter()-start) *1000
        logger.info(f"[timing] {label} : {ms:.0f} ms")
