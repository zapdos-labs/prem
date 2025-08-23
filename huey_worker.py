from huey import FileHuey
import time

huey = FileHuey('tasks', path='./queue')

@huey.task()
def add(a, b):
    time.sleep(1)
    return a + b
