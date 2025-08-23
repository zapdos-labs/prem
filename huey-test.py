import time

# Import your task
from huey_worker import add  

if __name__ == "__main__":
    # Submit the task to the queue
    res = add(5, 3)
    
    print("Task submitted:", res)  # this is an AsyncResult object

    print("Task result:", res(blocking=True))  # prints 8 once the worker has processed it
