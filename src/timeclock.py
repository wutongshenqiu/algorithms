import time
import functools
import colorama

colorama.init(autoreset=True)


def clock(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        begin = time.strftime("%Y-%m-%d %H:%M:%S")
        begin_time = time.time()
        print("=" * 100)
        print(
            f"{colorama.Fore.GREEN}function {colorama.Fore.RED}{func.__name__} {colorama.Fore.GREEN}starts at {begin}")
        result = func(*args, **kwargs)
        end = time.strftime("%Y-%m-%d %H:%M:%S")
        end_time = time.time()
        print(f"{colorama.Fore.GREEN}function {colorama.Fore.RED}{func.__name__} {colorama.Fore.GREEN}ends at {end}")
        print(
            f"{colorama.Fore.BLUE}function {colorama.Fore.RED}{func.__name__} {colorama.Fore.BLUE}runs last {end_time - begin_time:.2f} seconds")
        print("=" * 100)
        return result

    return wrapper
