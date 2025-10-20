.PHONY: help setup start stop restart logs clean test lint format

help:
	@echo "Available commands:"
	@echo "  make setup      - Set up development environment"
	@echo "  make start      - Start all services"
	@echo "  make stop       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - View logs"
	@echo "  make clean      - Clean up containers and volumes"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code"
	@echo "  make shell      - Open Python shell with app context"

setup:
	@echo "Setting up development environment..."
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -r requirements-dev.txt
# 	cp .env.example .env
	@echo "Setup complete! Edit .env with your configuration."

start:
	@echo "Starting services..."
	docker-compose up -d
	@echo "Services started!"
	@echo "PostgreSQL: localhost:5432"
	@echo "Redis: localhost:6379"

stop:
	@echo "Stopping services..."
	docker-compose down

restart:
	@echo "Restarting services..."
	docker-compose restart

logs:
	docker-compose logs -f

clean:
	@echo "Cleaning up..."
	docker-compose down -v
	rm -rf __pycache__ **/__pycache__
	rm -rf .pytest_cache
	@echo "Cleanup complete!"

test:
	. venv/bin/activate && pytest tests/ -v

lint:
	. venv/bin/activate && flake8 backend/ --max-line-length=100
	. venv/bin/activate && mypy backend/ --ignore-missing-imports

format:
	. venv/bin/activate && black backend/ tests/

shell:
	. venv/bin/activate && ipython