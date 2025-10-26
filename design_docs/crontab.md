# Complete Guide to Crontab in Celery

## What is Crontab?

**Crontab** is a time-based job scheduler format originally from Unix/Linux systems. Celery uses the same syntax to schedule periodic tasks.

---

## Basic Crontab Syntax

The standard crontab format has **5 fields**:

```
* * * * *
│ │ │ │ │
│ │ │ │ └─── Day of week (0-6, Sunday=0)
│ │ │ └───── Month (1-12)
│ │ └─────── Day of month (1-31)
│ └───────── Hour (0-23)
└─────────── Minute (0-59)
```

**Special characters:**
- `*` = Every (any value)
- `,` = Multiple values (1,5,10)
- `-` = Range (1-5)
- `/` = Step/interval (*/10 = every 10)

---

## Celery's `crontab()` Function

In Celery, you use `crontab()` from `celery.schedules`:

```python
from celery.schedules import crontab

# Syntax:
crontab(
    minute='*',      # 0-59
    hour='*',        # 0-23
    day_of_week='*', # 0-6 (Sunday=0) or mon,tue,wed,thu,fri,sat,sun
    day_of_month='*',# 1-31
    month_of_year='*'# 1-12
)
```

**Important:** Only specify the fields you need. Unspecified fields default to `*` (every).

---

## Common Examples

### Every Minute
```python
crontab()  # All fields default to '*'
# Runs: 10:00, 10:01, 10:02, 10:03, ...
```

### Every 5 Minutes
```python
crontab(minute='*/5')
# Runs: 10:00, 10:05, 10:10, 10:15, 10:20, ...
```

### Every 10 Minutes
```python
crontab(minute='*/10')
# Runs: 10:00, 10:10, 10:20, 10:30, 10:40, 10:50, ...
```

### Every 30 Minutes
```python
crontab(minute='*/30')
# Runs: 10:00, 10:30, 11:00, 11:30, ...
```

### Every Hour (at :00)
```python
crontab(minute=0)
# Runs: 10:00, 11:00, 12:00, 13:00, ...
```

### Every Hour at :15
```python
crontab(minute=15)
# Runs: 10:15, 11:15, 12:15, 13:15, ...
```

### Every 2 Hours
```python
crontab(minute=0, hour='*/2')
# Runs: 00:00, 02:00, 04:00, 06:00, 08:00, 10:00, ...
```

### Every 6 Hours
```python
crontab(minute=0, hour='*/6')
# Runs: 00:00, 06:00, 12:00, 18:00
```

---

## Specific Times

### Every Day at 2:30 AM
```python
crontab(minute=30, hour=2)
# Runs: Daily at 02:30
```

### Every Day at Midnight
```python
crontab(minute=0, hour=0)
# Runs: Daily at 00:00
```

### Every Day at 9 AM and 5 PM
```python
crontab(minute=0, hour='9,17')
# Runs: Daily at 09:00 and 17:00
```

### Multiple Times per Hour
```python
crontab(minute='0,15,30,45')
# Runs: Every hour at :00, :15, :30, :45
```

---

## Days of the Week

### Monday Through Friday (Weekdays)
```python
crontab(minute=0, hour=9, day_of_week='1-5')
# Runs: Mon-Fri at 09:00

# Alternative with names:
crontab(minute=0, hour=9, day_of_week='mon-fri')
```

**Day of week numbers:**
```
0 = Sunday
1 = Monday
2 = Tuesday
3 = Wednesday
4 = Thursday
5 = Friday
6 = Saturday
```

**You can also use names:**
```python
'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'
```

### Every Monday at 9 AM
```python
crontab(minute=0, hour=9, day_of_week=1)
# or
crontab(minute=0, hour=9, day_of_week='mon')
```

### Every Weekend (Saturday & Sunday)
```python
crontab(minute=0, hour=10, day_of_week='0,6')
# or
crontab(minute=0, hour=10, day_of_week='sat,sun')
```

### Every Wednesday and Friday at 3 PM
```python
crontab(minute=0, hour=15, day_of_week='3,5')
# or
crontab(minute=0, hour=15, day_of_week='wed,fri')
```

---

## Days of the Month

### First Day of Every Month
```python
crontab(minute=0, hour=0, day_of_month=1)
# Runs: 1st of each month at 00:00
```

### 15th of Every Month
```python
crontab(minute=0, hour=12, day_of_month=15)
# Runs: 15th of each month at 12:00
```

### Last Day of the Month
```python
crontab(minute=0, hour=23, day_of_month='28-31')
# Runs: 28th, 29th, 30th, 31st at 23:00
# (Celery will skip if day doesn't exist in that month)
```

### First and 15th of Every Month
```python
crontab(minute=0, hour=9, day_of_month='1,15')
# Runs: 1st and 15th at 09:00
```

---

## Months

### Every January
```python
crontab(minute=0, hour=0, day_of_month=1, month_of_year=1)
# Runs: January 1st at 00:00 every year
```

### Every Quarter (Jan, Apr, Jul, Oct)
```python
crontab(minute=0, hour=9, day_of_month=1, month_of_year='1,4,7,10')
# Runs: 1st of Jan, Apr, Jul, Oct at 09:00
```

### Summer Months (Jun, Jul, Aug)
```python
crontab(minute=0, hour=6, month_of_year='6-8')
# Runs: Daily at 06:00 during June, July, August
```

---

## Complex Examples

### Business Hours: Every 15 Minutes, Mon-Fri, 9 AM - 5 PM
```python
crontab(
    minute='*/15',
    hour='9-17',
    day_of_week='mon-fri'
)
# Runs: Every 15 minutes during business hours, weekdays only
```

### Night Processing: Every Hour from 10 PM to 6 AM
```python
crontab(
    minute=0,
    hour='22-23,0-6'  # 22, 23, 0, 1, 2, 3, 4, 5, 6
)
# Runs: 22:00, 23:00, 00:00, 01:00, ..., 06:00
```

### Weekly Report: Every Sunday at Midnight
```python
crontab(
    minute=0,
    hour=0,
    day_of_week='sun'
)
# Runs: Sunday at 00:00
```

### Bi-Weekly: Every Other Monday
```python
# Not directly supported by crontab
# Use a task that checks the week number internally
crontab(minute=0, hour=9, day_of_week='mon')
# Then in the task:
import datetime
week_num = datetime.date.today().isocalendar()[1]
if week_num % 2 == 0:  # Even weeks only
    # Do the work
```

---

## Understanding the `*/N` Syntax

**`*/N` means "every N units"**

### Minute Examples:

```python
# */5 means "every 5 minutes"
crontab(minute='*/5')
# Runs at: 00, 05, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55

# */10 means "every 10 minutes"
crontab(minute='*/10')
# Runs at: 00, 10, 20, 30, 40, 50

# */15 means "every 15 minutes"
crontab(minute='*/15')
# Runs at: 00, 15, 30, 45

# */20 means "every 20 minutes"
crontab(minute='*/20')
# Runs at: 00, 20, 40
```

### Hour Examples:

```python
# */2 means "every 2 hours"
crontab(hour='*/2')
# Runs at: 00:00, 02:00, 04:00, 06:00, ..., 22:00

# */3 means "every 3 hours"
crontab(hour='*/3')
# Runs at: 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00

# */6 means "every 6 hours"
crontab(hour='*/6')
# Runs at: 00:00, 06:00, 12:00, 18:00

# */12 means "every 12 hours"
crontab(hour='*/12')
# Runs at: 00:00, 12:00
```

---

## Your Project's Configuration

### Task 1: Process Pending Articles

```python
"process-pending-articles": {
    "schedule": crontab(minute=f"*/{settings.processing_interval_minutes}"),
}
```

**Breaking it down:**

```python
# If processing_interval_minutes = 10 in settings
minute=f"*/{settings.processing_interval_minutes}"
# Becomes:
minute="*/10"

# This means: Every 10 minutes
# Timeline:
10:00 → Task runs
10:10 → Task runs
10:20 → Task runs
10:30 → Task runs
...
```

**If you change the setting:**

```python
# processing_interval_minutes = 5
crontab(minute="*/5")
# Runs every 5 minutes: 10:00, 10:05, 10:10, ...

# processing_interval_minutes = 15
crontab(minute="*/15")
# Runs every 15 minutes: 10:00, 10:15, 10:30, ...

# processing_interval_minutes = 30
crontab(minute="*/30")
# Runs every 30 minutes: 10:00, 10:30, 11:00, ...
```

---

### Task 2: Save FAISS Index

```python
"save-faiss-index": {
    "schedule": crontab(minute=0),
}
```

**Breaking it down:**

```python
minute=0
# No other fields specified, so hour='*', day_of_week='*', etc.

# This means: Every hour, at minute 0
# Timeline:
10:00 → Save FAISS
11:00 → Save FAISS
12:00 → Save FAISS
13:00 → Save FAISS
...
```

---

## Testing Crontab Schedules

Want to see when your crontab will run? Use this helper:

```python
from celery.schedules import crontab
from datetime import datetime, timedelta

def print_next_runs(schedule, num_runs=10):
    """Print the next N times a crontab schedule will run"""
    now = datetime.now()

    for i in range(num_runs):
        # Check every minute for the next hour
        for minutes_ahead in range(60 * 24):  # Check next 24 hours
            check_time = now + timedelta(minutes=minutes_ahead)

            if schedule.is_due(check_time)[0]:
                print(f"{i+1}. {check_time.strftime('%Y-%m-%d %H:%M:%S')}")
                now = check_time + timedelta(minutes=1)
                break

# Test your schedules
print("Every 10 minutes:")
print_next_runs(crontab(minute='*/10'), num_runs=5)

print("\nEvery hour at :00:")
print_next_runs(crontab(minute=0), num_runs=5)
```

**Output:**
```
Every 10 minutes:
1. 2025-10-25 14:30:00
2. 2025-10-25 14:40:00
3. 2025-10-25 14:50:00
4. 2025-10-25 15:00:00
5. 2025-10-25 15:10:00

Every hour at :00:
1. 2025-10-25 15:00:00
2. 2025-10-25 16:00:00
3. 2025-10-25 17:00:00
4. 2025-10-25 18:00:00
5. 2025-10-25 19:00:00
```

---

## Common Mistakes & Gotchas

### ❌ Mistake 1: Overlapping Values

```python
# This runs every minute (not every 10 minutes!)
crontab(minute='*', hour='*/10')  # WRONG!

# Correct way to run every 10 hours:
crontab(minute=0, hour='*/10')  # RIGHT
```

### ❌ Mistake 2: Confusing Day of Week

```python
# This runs every day (not Monday)
crontab(day_of_week='*')  # Every day

# This runs only Monday
crontab(day_of_week=1)  # or day_of_week='mon'
```

### ❌ Mistake 3: Using Both day_of_month and day_of_week

```python
# This is tricky! It runs on EITHER condition
crontab(day_of_month=1, day_of_week='mon')
# Runs: 1st of month OR every Monday (whichever comes first)

# If you want "First Monday", you need custom logic in the task
```

### ❌ Mistake 4: Timezone Issues

```python
# Celery uses UTC by default
crontab(hour=9)  # This is 9 AM UTC, not your local time!

# If you're in PST (UTC-8):
# 9 AM PST = 5 PM UTC
crontab(hour=17)  # For 9 AM PST
```

**Better solution:** Configure timezone in celery config:

```python
celery_app.conf.update(
    timezone='America/Los_Angeles',  # Your timezone
    enable_utc=False,
)
```

---

## Alternative: Using `schedule` (Interval)

If you don't need specific times, use `schedule` instead:

```python
from celery.schedules import schedule
from datetime import timedelta

celery_app.conf.beat_schedule = {
    'run-every-30-seconds': {
        'task': 'my_task',
        'schedule': timedelta(seconds=30),  # Every 30 seconds
    },
    'run-every-5-minutes': {
        'task': 'my_task',
        'schedule': timedelta(minutes=5),  # Every 5 minutes
    },
    'run-every-hour': {
        'task': 'my_task',
        'schedule': timedelta(hours=1),  # Every hour
    },
}
```

**When to use each:**

| Use `crontab` | Use `timedelta` |
|---------------|-----------------|
| Specific times (9 AM) | Intervals (every 30 min) |
| Days of week (Monday) | Doesn't matter when |
| Human-readable schedule | Simple intervals |

---

## Quick Reference Table

| Schedule | Crontab | Description |
|----------|---------|-------------|
| **Every minute** | `crontab()` | Runs every minute |
| **Every 5 min** | `crontab(minute='*/5')` | :00, :05, :10, ... |
| **Every 10 min** | `crontab(minute='*/10')` | :00, :10, :20, ... |
| **Every 30 min** | `crontab(minute='*/30')` | :00, :30 |
| **Every hour** | `crontab(minute=0)` | On the hour |
| **Every 2 hours** | `crontab(minute=0, hour='*/2')` | 00:00, 02:00, ... |
| **Every day at 2 AM** | `crontab(minute=0, hour=2)` | Daily at 02:00 |
| **Weekdays at 9 AM** | `crontab(minute=0, hour=9, day_of_week='1-5')` | Mon-Fri 09:00 |
| **Every Monday** | `crontab(day_of_week=1)` | Weekly |
| **1st of month** | `crontab(minute=0, hour=0, day_of_month=1)` | Monthly |

---

## Recommendations for Your Project

Based on your news recommender system:

```python
celery_app.conf.beat_schedule = {
    # Process new articles every 10 minutes
    'process-articles': {
        'task': 'process_pending_articles',
        'schedule': crontab(minute='*/10'),  # Every 10 min
    },

    # Save FAISS index every hour (backup)
    'save-faiss': {
        'task': 'save_faiss_index',
        'schedule': crontab(minute=0),  # Every hour at :00
    },

    # Scrape RSS feeds every 30 minutes
    'scrape-rss': {
        'task': 'scrape_all_feeds',
        'schedule': crontab(minute='*/30'),  # Every 30 min
    },

    # Clean up old data every night at 3 AM
    'cleanup': {
        'task': 'cleanup_old_data',
        'schedule': crontab(minute=0, hour=3),  # Daily at 03:00
    },

    # Generate daily report at 8 AM
    'daily-report': {
        'task': 'generate_daily_report',
        'schedule': crontab(minute=0, hour=8),  # Daily at 08:00
    },

    # Rebuild FAISS index from scratch every Sunday at midnight
    'rebuild-index': {
        'task': 'rebuild_faiss_index',
        'schedule': crontab(minute=0, hour=0, day_of_week='sun'),  # Weekly
    },
}
```

---

## Summary

**Crontab in Celery lets you schedule tasks like a cron job:**

- ✅ `crontab()` - Every minute
- ✅ `crontab(minute='*/10')` - Every 10 minutes
- ✅ `crontab(minute=0)` - Every hour
- ✅ `crontab(hour=9, day_of_week='mon-fri')` - Weekdays at 9 AM
- ✅ `crontab(day_of_month=1)` - First of every month

**Key Points:**
- Only specify fields you need (others default to `*`)
- Use `*/N` for intervals ("every N minutes/hours")
- Day of week: 0-6 or 'mon', 'tue', etc.
- Default timezone is UTC (configure if needed)
- Use `timedelta` for simple intervals without specific times