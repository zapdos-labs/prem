import time

def timeit_log(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"Method {func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper
