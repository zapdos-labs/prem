# main.py
from sanic import Sanic
from sanic.response import json
import asyncio
import os

app = Sanic("ConcurrentApp")

@app.get("/long-running")
async def long_running(request):
    print("Task started...")
    await asyncio.sleep(10) 
    print("Task finished...")
    return json({"status": "completed", "message": "Long-running task finished."})

@app.get("/cpu-intensive")
async def cpu_task(request):
    # This blocks the entire worker process
    print("CPU task started...")
    result = 0
    for i in range(10_000_000_0):  # Pure CPU work
        result += i * i
    print("CPU task finished...")
    return json({"result": result})

@app.get("/")
async def root(request):
    return json({"message": "Hello! Hit /long-running to test concurrency."})

if __name__ == "__main__":
    workers = os.cpu_count()
    app.run(host="0.0.0.0", port=8080, workers=workers)
