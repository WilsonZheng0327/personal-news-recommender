"""
Test script to manually trigger Celery tasks
Run this AFTER starting the Celery worker
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.tasks import (
    test_task,
    get_processing_stats,
    process_single_article
)

print("="*70)
print("Testing Celery Tasks")
print("="*70)

# Test 1: Simple test task
print("\n1. Testing simple task (test_task)...")
result = test_task.delay()
print(f"   Task ID: {result.id}")
print(f"   Status: {result.status}")
print(f"   Waiting for result...")
print(f"   Result: {result.get(timeout=10)}")

# print("\n3 Testing process single article...")
# result3 = process_single_article.delay(100)
# print(f"   Task ID: {result3.id}")
# print(f"   Status: {result3.status}")
# print(f"   Waiting for result...")
# print(f"   Result: {result3.get(timeout=100)}")

print("\n2 Testing get_processing_stats...")
result2 = get_processing_stats.delay()
print(f"   Task ID: {result2.id}")
print(f"   Status: {result2.status}")
print(f"   Waiting for result...")
print(f"   Result: {result2.get(timeout=100)}")