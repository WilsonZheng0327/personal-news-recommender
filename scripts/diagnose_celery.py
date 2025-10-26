"""
Comprehensive Celery diagnostic script

This will help identify why Celery isn't running tasks.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print("CELERY DIAGNOSTIC SCRIPT")
print("="*70)

# Test 1: Check Redis connection
print("\n1. Testing Redis connection...")
try:
    import redis
    from config.settings import get_settings

    settings = get_settings()
    print(f"   Broker URL: {settings.celery_broker_url}")
    print(f"   Result Backend: {settings.celery_result_backend}")

    # Test broker connection (db 0)
    r_broker = redis.from_url(settings.celery_broker_url)
    r_broker.ping()
    print("   [OK] Broker (Redis DB 0) is reachable")

    # Test backend connection (db 1)
    r_backend = redis.from_url(settings.celery_result_backend)
    r_backend.ping()
    print("   [OK] Result Backend (Redis DB 1) is reachable")

except Exception as e:
    print(f"   [FAIL] Redis connection failed: {e}")
    print("   Make sure Redis is running!")
    print("   Start with: redis-server")
    sys.exit(1)

# Test 2: Check Celery app configuration
print("\n2. Checking Celery app configuration...")
try:
    from backend.celery_app import celery_app

    print(f"   App name: {celery_app.main}")
    print(f"   Broker: {celery_app.conf.broker_url}")
    print(f"   Backend: {celery_app.conf.result_backend}")
    print(f"   Timezone: {celery_app.conf.timezone}")
    print(f"   Task modules: {celery_app.conf.include}")
    print("   [OK] Celery app loaded successfully")

except Exception as e:
    print(f"   [FAIL] Failed to load Celery app: {e}")
    sys.exit(1)

# Test 3: Check registered tasks
print("\n3. Checking registered tasks...")
try:
    registered_tasks = sorted(celery_app.tasks.keys())
    print(f"   Found {len(registered_tasks)} registered tasks:")

    for task in registered_tasks:
        if not task.startswith('celery.'):  # Skip built-in Celery tasks
            print(f"     - {task}")

    # Check if our test task is registered
    if "backend.tasks.processing_tasks.test_task" in registered_tasks:
        print("   [OK] test_task is registered")
    else:
        print("   [FAIL] test_task is NOT registered!")
        print("   This is a problem - check backend/tasks/processing_tasks.py")

except Exception as e:
    print(f"   [FAIL] Failed to get registered tasks: {e}")

# Test 4: Check Beat schedule
print("\n4. Checking Beat schedule...")
try:
    beat_schedule = celery_app.conf.beat_schedule

    print(f"   Found {len(beat_schedule)} scheduled tasks:")
    for name, config in beat_schedule.items():
        task_name = config['task']
        schedule = config['schedule']
        print(f"     - {name}")
        print(f"       Task: {task_name}")
        print(f"       Schedule: {schedule}")
        if 'options' in config:
            print(f"       Queue: {config['options'].get('queue', 'default')}")

    if "test-task" in beat_schedule:
        print("   [OK] test-task is in schedule")
    else:
        print("   [FAIL] test-task is NOT in schedule!")

except Exception as e:
    print(f"   [FAIL] Failed to get beat schedule: {e}")

# Test 5: Check task routing
print("\n5. Checking task routing...")
try:
    task_routes = celery_app.conf.task_routes

    if task_routes:
        print("   Task routing rules:")
        for pattern, config in task_routes.items():
            print(f"     {pattern} -> queue: {config.get('queue', 'default')}")

        # Check what queue test_task will use
        test_task_name = "backend.tasks.processing_tasks.test_task"

        # Check if test_task matches any routing pattern
        matched = False
        for pattern in task_routes.keys():
            if pattern.endswith('*'):
                prefix = pattern[:-1]
                if test_task_name.startswith(prefix):
                    queue = task_routes[pattern].get('queue', 'default')
                    print(f"   test_task matches pattern '{pattern}' -> queue '{queue}'")
                    matched = True
            elif pattern == test_task_name:
                queue = task_routes[pattern].get('queue', 'default')
                print(f"   test_task has explicit route -> queue '{queue}'")
                matched = True

        if not matched:
            print(f"   test_task doesn't match any routing -> uses default 'celery' queue")
    else:
        print("   No task routing configured")

except Exception as e:
    print(f"   [FAIL] Failed to get task routing: {e}")

# Test 6: Manually trigger test task
print("\n6. Testing manual task trigger...")
print("   Attempting to trigger test_task manually...")

try:
    from backend.tasks.processing_tasks import test_task

    result = test_task.delay()
    print(f"   Task submitted!")
    print(f"   Task ID: {result.id}")
    print(f"   Status: {result.status}")

    print("\n   Waiting 5 seconds for task to complete...")
    import time
    time.sleep(5)

    print(f"   Status after wait: {result.status}")

    if result.status == "SUCCESS":
        print(f"   Result: {result.get()}")
        print("   [OK] Task completed successfully!")
        print("\n   If this worked but Beat isn't triggering tasks,")
        print("   the problem is with Celery Beat, not the worker.")
    elif result.status == "PENDING":
        print("   [FAIL] Task is still pending - worker might not be running!")
        print("\n   Make sure worker is started with:")
        print("   celery -A backend.celery_app worker --loglevel=info --pool=solo")
    else:
        print(f"   Task status: {result.status}")

except Exception as e:
    print(f"   [FAIL] Failed to trigger task: {e}")

# Test 7: Check for active workers
print("\n7. Checking for active Celery workers...")
try:
    inspector = celery_app.control.inspect()

    # Get active workers
    active_workers = inspector.active()

    if active_workers:
        print(f"   [OK] Found {len(active_workers)} active worker(s):")
        for worker_name, tasks in active_workers.items():
            print(f"     - {worker_name}: {len(tasks)} active tasks")
    else:
        print("   [FAIL] No active workers found!")
        print("\n   Start a worker with:")
        print("   celery -A backend.celery_app worker --loglevel=info --pool=solo")

    # Get registered tasks on workers
    registered = inspector.registered()
    if registered:
        print("\n   Tasks registered on workers:")
        for worker_name, tasks in registered.items():
            print(f"     Worker: {worker_name}")
            for task in sorted(tasks):
                if not task.startswith('celery.'):
                    print(f"       - {task}")

except Exception as e:
    print(f"   [FAIL] Failed to inspect workers: {e}")
    print("   This usually means no workers are running")

# Summary
print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

print("\nTo fix Celery Beat not running tasks:")
print("\n1. Make sure Redis is running:")
print("   redis-server")
print("\n2. Start Celery worker in one terminal:")
print("   ./venv/Scripts/celery.exe -A backend.celery_app worker --loglevel=info --pool=solo")
print("\n3. Start Celery Beat in another terminal:")
print("   ./venv/Scripts/celery.exe -A backend.celery_app beat --loglevel=info")
print("\n4. Watch the Beat terminal for:")
print("   'Scheduler: Sending due task test-task'")
print("\n5. Watch the Worker terminal for:")
print("   'Task backend.tasks.processing_tasks.test_task[...] received'")
print("="*70)