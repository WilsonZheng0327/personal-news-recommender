# Docker Compose - Complete Explanation

## What is Docker Compose?

**Simple Answer**: A tool to run multiple Docker containers together as one application.

**Your Case**: You need PostgreSQL + Redis + (later) your FastAPI app. Docker Compose lets you start all of them with ONE command: `docker-compose up`

---

## The Big Picture

### Without Docker Compose (The Hard Way):
```bash
# Start PostgreSQL
docker run -d \
  --name postgres \
  -e POSTGRES_DB=news_recommender \
  -e POSTGRES_USER=news_user \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:15-alpine

# Start Redis
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7-alpine

# Start your app
docker run -d \
  --name app \
  --link postgres \
  --link redis \
  -p 8000:8000 \
  your-app

# Too many commands! Easy to make mistakes!
```

### With Docker Compose (The Easy Way):
```bash
docker-compose up -d
# Done! All services start together
```

---

## Line-by-Line Breakdown

### Top Section: Version & Structure

```yaml
version: '3.8'
```

**What it means**: Which Docker Compose file format to use
- `3.8` is the latest stable version
- Different versions have different features
- You almost always want `3.8`

**Why it matters**: Ensures compatibility with your Docker version

---

```yaml
services:
```

**What it means**: Start of the "services" section
- Each service = one container
- You define: postgres, redis, your app, etc.
- All services are defined under this

---

## Service 1: PostgreSQL Database

```yaml
  postgres:
```

**What it means**: Name of this service
- You can call it anything: `postgres`, `db`, `database`
- Other services reference it by this name
- In your app: `host="postgres"` (not `localhost`!)

---

```yaml
    image: postgres:15-alpine
```

**What it means**: Which Docker image to use
- `postgres` = PostgreSQL database
- `15` = version 15 (latest stable)
- `alpine` = lightweight Linux (smaller image)

**Alternatives**:
```yaml
image: postgres:15        # Normal size (~300 MB)
image: postgres:15-alpine # Alpine (~200 MB) ← Smaller, faster!
image: postgres:14-alpine # Older version
```

**Why alpine?** 
- Smaller download
- Faster startup
- Less disk space
- Same functionality

---

```yaml
    container_name: news_postgres
```

**What it means**: The actual container name in Docker
- Without this: Docker generates random name like `news-recommender_postgres_1`
- With this: Always `news_postgres` (easier to reference)

**Why it matters**:
```bash
# Easy to remember
docker logs news_postgres
docker exec -it news_postgres psql

# vs random name
docker logs news-recommender_postgres_1_a4f3b2
```

---

```yaml
    environment:
      POSTGRES_DB: news_recommender
      POSTGRES_USER: news_user
      POSTGRES_PASSWORD: news_password_dev
```

**What it means**: Environment variables passed to the container
- Like setting variables in `.env` but for the container
- PostgreSQL reads these to configure itself

**What each does**:
```yaml
POSTGRES_DB: news_recommender
# Creates a database named "news_recommender"

POSTGRES_USER: news_user  
# Creates a user named "news_user"

POSTGRES_PASSWORD: news_password_dev
# Sets the password to "news_password_dev"
```

**Result**: 
```sql
-- Database ready to use with these credentials:
postgresql://news_user:news_password_dev@localhost:5432/news_recommender
```

**Security Note**: 
- `news_password_dev` is fine for local development
- In production: Use strong passwords and secrets management
- Never commit real passwords to git!

---

```yaml
    ports:
      - "5432:5432"
```

**What it means**: Port mapping (HOST:CONTAINER)
- `5432` (left) = Port on your computer
- `5432` (right) = Port inside container
- Maps container port to your machine

**Visualization**:
```
Your Computer               Docker Container
localhost:5432    ←→    postgres:5432
     ↑                        ↑
Your Python app         PostgreSQL server
connects here           listening here
```

**Why this matters**:
```python
# Your app can connect to:
DATABASE_URL = "postgresql://news_user:password@localhost:5432/news_recommender"
#                                                    ↑
#                                        This port must match!
```

**Different port example**:
```yaml
ports:
  - "5433:5432"  # Use 5433 on your computer if 5432 is taken
```

Then connect to: `localhost:5433`

---

```yaml
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
```

**What it means**: Persistent storage (data survives container restart)

**First volume**:
```yaml
postgres_data:/var/lib/postgresql/data
```
- `postgres_data` = Named volume (Docker manages it)
- `/var/lib/postgresql/data` = Where Postgres stores data inside container
- **Purpose**: Save your database data permanently

**Without this**:
```bash
docker-compose down
# All your data is DELETED! 💀

docker-compose up
# Empty database, start from scratch
```

**With this**:
```bash
docker-compose down
# Container stops, but data saved in volume

docker-compose up
# Container starts, data still there! ✅
```

**Second volume**:
```yaml
./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
```
- `./scripts/init_db.sql` = File on your computer
- `/docker-entrypoint-initdb.d/init.sql` = Special folder in container
- **Purpose**: Run SQL script on first startup

**How it works**:
1. First time container starts
2. PostgreSQL looks in `/docker-entrypoint-initdb.d/`
3. Runs any `.sql` files found there
4. Your tables are created automatically!

**Example init_db.sql**:
```sql
-- Creates tables on first run
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT
);
```

---

```yaml
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U news_user -d news_recommender"]
      interval: 10s
      timeout: 5s
      retries: 5
```

**What it means**: Docker checks if PostgreSQL is actually ready

**The problem without healthcheck**:
```bash
docker-compose up
# PostgreSQL container starts
# Your app tries to connect immediately
# ERROR: "Connection refused" (Postgres not ready yet!)
```

**How healthcheck works**:
1. Every 10 seconds (`interval`), Docker runs the test command
2. Test: `pg_isready` checks if Postgres is accepting connections
3. If test fails, wait 5 seconds (`timeout`) then retry
4. After 5 failures (`retries`), mark container as unhealthy

**What each line does**:
```yaml
test: ["CMD-SHELL", "pg_isready -U news_user -d news_recommender"]
# Command to check if database is ready
# pg_isready = PostgreSQL utility that checks connection

interval: 10s
# How often to check (every 10 seconds)

timeout: 5s
# How long to wait for check to complete

retries: 5
# How many failures before giving up
```

**Benefit**: Other services can wait for postgres to be healthy before starting

---

```yaml
    networks:
      - news_network
```

**What it means**: Which network this container joins
- Containers on same network can talk to each other
- Like WiFi network for your containers

**Why this matters**:
```yaml
# Both on news_network
postgres:
  networks: [news_network]
  
redis:
  networks: [news_network]

# They can talk to each other!
# postgres can reach redis
# redis can reach postgres
```

**In your app**:
```python
# Connect using service names (not localhost!)
DATABASE_URL = "postgresql://news_user:password@postgres:5432/db"
#                                              ↑
#                                    Service name, not "localhost"!
```

---

## Service 2: Redis Cache

```yaml
  redis:
    image: redis:7-alpine
```

Same concept as postgres:
- `redis` = Redis key-value store
- `7` = version 7
- `alpine` = lightweight

---

```yaml
    container_name: news_redis
    ports:
      - "6379:6379"
```

- Container name: `news_redis`
- Port 6379 = default Redis port
- Map to same port on your machine

---

```yaml
    volumes:
      - redis_data:/data
```

**What it means**: Persist Redis data
- `redis_data` = named volume
- `/data` = where Redis stores data

**Purpose**: Cache and message queue data survives restarts

---

```yaml
    command: redis-server --appendonly yes
```

**What it means**: Override default command container runs
- Default: `redis-server` (in-memory only, data lost on restart)
- Override: `redis-server --appendonly yes` (write to disk)

**Why `--appendonly yes`?**
- Redis normally keeps data in RAM only
- `appendonly` mode writes to disk
- Data survives container restart
- Slightly slower but much safer

**Without this**:
```bash
redis.set("user:123", "data")
docker-compose restart redis
redis.get("user:123")  # Returns None 💀 (data lost!)
```

**With this**:
```bash
redis.set("user:123", "data")
docker-compose restart redis
redis.get("user:123")  # Returns "data" ✅ (data saved!)
```

---

```yaml
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
```

**What it means**: Check if Redis is ready
- `redis-cli ping` = sends PING command
- If Redis responds with PONG, it's healthy
- Same timing as postgres healthcheck

---

## Commented Service: Flower (Celery Monitoring)

```yaml
  # flower:
  #   build: .
  #   container_name: news_flower
  #   command: celery -A backend.workers.celery_app flower --port=5555
  #   ports:
  #     - "5555:5555"
  #   environment:
  #     - CELERY_BROKER_URL=redis://redis:6379/0
  #   depends_on:
  #     - redis
  #   networks:
  #     - news_network
```

**What it means**: Optional service for monitoring Celery tasks
- Currently commented out (not running)
- Uncomment when you add Celery workers

**New concepts here**:

```yaml
build: .
# Build Docker image from Dockerfile in current directory
# (instead of using pre-built image)
```

```yaml
depends_on:
  - redis
# Start redis BEFORE starting flower
# Ensures dependencies start in correct order
```

```yaml
environment:
  - CELERY_BROKER_URL=redis://redis:6379/0
# Use service name "redis" (not "localhost")
# Containers talk to each other by service name
```

**When to uncomment**:
- After you add Celery workers
- Want to monitor background tasks
- Need a UI to see task status

---

## Choosing Where to Store Data

### Option 1: Docker-Managed Volumes (Current Setup)
```yaml
volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
```

**Location**: Docker chooses (usually hidden system folder)
- **Pros**: Docker manages it, works everywhere, automatic cleanup
- **Cons**: Hard to find, can't easily backup by copying folder

### Option 2: Bind Mounts (Specific Folder) - RECOMMENDED FOR DEV

```yaml
volumes:
  postgres_data:
  redis_data:

services:
  postgres:
    # ... other config ...
    volumes:
      - ./data/postgres:/var/lib/postgresql/data  # ← Your folder!
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql

  redis:
    # ... other config ...
    volumes:
      - ./data/redis:/data  # ← Your folder!
```

**Result**: Data stored in your project folder
```
news-recommender/
├── data/
│   ├── postgres/     ← PostgreSQL data HERE
│   │   ├── base/
│   │   ├── global/
│   │   └── pg_wal/
│   └── redis/        ← Redis data HERE
│       └── appendonly.aof
├── backend/
└── docker-compose.yml
```

**Pros**: 
- ✅ Easy to find
- ✅ Easy to backup (just copy `data/` folder)
- ✅ Easy to reset (delete `data/` folder)
- ✅ Can inspect files directly

**Cons**:
- ⚠️ Must add `data/` to `.gitignore` (don't commit database files!)
- ⚠️ Permission issues on Linux (see solutions below)

### Option 3: Absolute Paths (Any Location)

```yaml
services:
  postgres:
    volumes:
      - /Users/yourname/Documents/postgres-data:/var/lib/postgresql/data
      # ↑ Full path to anywhere on your computer

  redis:
    volumes:
      - /Users/yourname/Documents/redis-data:/data
```

**Pros**: Store data anywhere you want
**Cons**: Not portable (path must exist on all machines)

---

## Complete Updated docker-compose.yml

### With Bind Mounts (Your Project Folder)

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: news_postgres
    environment:
      POSTGRES_DB: news_recommender
      POSTGRES_USER: news_user
      POSTGRES_PASSWORD: news_password_dev
    ports:
      - "5432:5432"
    volumes:
      # Use bind mount - data in your project folder
      - ./data/postgres:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U news_user -d news_recommender"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - news_network

  # Redis (Cache + Message Broker)
  redis:
    image: redis:7-alpine
    container_name: news_redis
    ports:
      - "6379:6379"
    volumes:
      # Use bind mount - data in your project folder
      - ./data/redis:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - news_network

networks:
  news_network:
    driver: bridge

# No volumes section needed when using bind mounts!
```

---

## Setup Instructions

### 1. Create Data Directories
```bash
# In your project root
mkdir -p data/postgres
mkdir -p data/redis

# On Linux/Mac: Set permissions (see below if issues)
chmod 755 data/postgres
chmod 755 data/redis
```

### 2. Update .gitignore
```bash
# Add to .gitignore
echo "data/" >> .gitignore

# Or add this line manually to .gitignore:
data/
```

**Why?** Database files are large and shouldn't be in git!

### 3. Start Services
```bash
docker-compose up -d
```

### 4. Verify Data Location
```bash
# Check files were created
ls -la data/postgres/
# Should see: base/, global/, pg_wal/, etc.

ls -la data/redis/
# Should see: appendonly.aof
```

---

## Permission Issues on Linux/Mac

### Problem:
```bash
docker-compose up -d
# Error: Permission denied
# PostgreSQL container can't write to ./data/postgres
```

### Why?
PostgreSQL runs as user `postgres` (UID 999) inside container
Your folder is owned by you (UID 1000)
Container can't write to it!

### Solution 1: Set Correct Permissions (Recommended)
```bash
# Make directories writable by container
sudo chown -R 999:999 data/postgres
sudo chown -R 999:999 data/redis

# Or make writable by everyone (less secure but easier)
chmod -R 777 data/postgres
chmod -R 777 data/redis
```

### Solution 2: Use User Mapping (Better)
```yaml
services:
  postgres:
    # ... other config ...
    user: "${UID}:${GID}"  # Run as your user
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
```

Then run:
```bash
UID=$(id -u) GID=$(id -g) docker-compose up -d
```

### Solution 3: Use Docker-Managed Volumes (Easiest)
```yaml
# Just use the original setup - Docker handles permissions
volumes:
  - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
    driver: local
```

---

## Comparison: Bind Mount vs Docker Volume

| Feature | Bind Mount (`./data/`) | Docker Volume (`postgres_data`) |
|---------|------------------------|----------------------------------|
| **Location** | Your project folder | Hidden Docker folder |
| **Easy to find** | ✅ Yes | ❌ No |
| **Easy to backup** | ✅ Copy folder | ❌ Need docker commands |
| **Easy to delete** | ✅ `rm -rf data/` | ❌ `docker volume rm` |
| **Permissions** | ⚠️ Can be tricky | ✅ Just works |
| **Performance** | ✅ Good | ✅ Slightly better |
| **Portability** | ⚠️ Paths must match | ✅ Works everywhere |
| **Best for** | Development | Production |

---

## My Recommendation

### For Development (Your Case):
```yaml
# Use bind mounts - easier to manage
volumes:
  - ./data/postgres:/var/lib/postgresql/data
  - ./data/redis:/data
```

**Why?**
- Easy to find your data
- Easy to delete and start fresh: `rm -rf data/`
- Easy to backup: Copy `data/` folder
- Can inspect database files if needed

### For Production:
```yaml
# Use Docker volumes - more reliable
volumes:
  - postgres_data:/var/lib/postgresql/data
  - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

**Why?**
- Better performance
- No permission issues
- Managed by Docker
- Standard practice

---

## Useful Commands with Bind Mounts

### See Your Data
```bash
# View PostgreSQL files
ls -lh data/postgres/

# View Redis files
ls -lh data/redis/
cat data/redis/appendonly.aof
```

### Backup Your Data
```bash
# Simple backup - copy folder
cp -r data/ data_backup_$(date +%Y%m%d)/

# Or create archive
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/
```

### Reset Database
```bash
# Stop containers
docker-compose down

# Delete data
rm -rf data/postgres/*
rm -rf data/redis/*

# Start fresh
docker-compose up -d
# Empty database!
```

### Move Data to Another Computer
```bash
# On computer A
tar -czf data.tar.gz data/
# Copy data.tar.gz to computer B

# On computer B
tar -xzf data.tar.gz
docker-compose up -d
# Same data!
```

---

## Complete Example: Setting Up from Scratch

```bash
# 1. Create project structure
mkdir news-recommender && cd news-recommender
mkdir -p data/postgres data/redis

# 2. Create docker-compose.yml with bind mounts
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    container_name: news_postgres
    environment:
      POSTGRES_DB: news_recommender
      POSTGRES_USER: news_user
      POSTGRES_PASSWORD: news_password_dev
    ports:
      - "5432:5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
    networks:
      - news_network
  
  redis:
    image: redis:7-alpine
    container_name: news_redis
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    command: redis-server --appendonly yes
    networks:
      - news_network

networks:
  news_network:
EOF

# 3. Add to gitignore
echo "data/" >> .gitignore

# 4. Fix permissions (Linux/Mac)
chmod -R 777 data/

# 5. Start services
docker-compose up -d

# 6. Verify
ls -la data/postgres/  # Should see database files
ls -la data/redis/     # Should see appendonly.aof

# 7. Connect to postgres
docker-compose exec postgres psql -U news_user -d news_recommender
```

---

## Troubleshooting

### Issue: "Permission denied" on Linux
```bash
# Solution: Make folder writable
sudo chmod -R 777 data/

# Or run container as your user
UID=$(id -u) GID=$(id -g) docker-compose up -d
```

### Issue: Data folder is huge
```bash
# Check size
du -sh data/*

# PostgreSQL: 
# - Small DB: ~50 MB
# - After lots of use: 500 MB+

# Solution: Vacuum database
docker-compose exec postgres psql -U news_user -d news_recommender -c "VACUUM FULL;"
```

### Issue: Want to switch from bind mount to volume
```bash
# 1. Backup current data
cp -r data/ data_backup/

# 2. Stop containers
docker-compose down

# 3. Change docker-compose.yml to use volumes
# (use the original version)

# 4. Start with volumes
docker-compose up -d

# 5. Migrate data (if needed)
docker cp data_backup/postgres/. news_postgres:/var/lib/postgresql/data/
```

---

## Updated .gitignore

```gitignore
# Database data (don't commit!)
data/
*.db
*.sqlite

# Docker volumes
postgres_data/
redis_data/

# Logs
logs/
*.log

# Python
__pycache__/
*.pyc
venv/
.env

# IDE
.vscode/
.idea/
```

---

## Bottom Section: Networks

```yaml
networks:
  news_network:
    driver: bridge
```

**What it means**: Define network for containers
- `news_network` = network name
- `driver: bridge` = default network type

**Network types**:
```yaml
driver: bridge
# Default, best for single machine
# Containers can talk to each other
# Isolated from outside

driver: host
# Use host's network directly
# No isolation
# Rarely needed

driver: overlay
# For multi-machine setups (Docker Swarm)
# Not needed for development
```

**Visualization**:
```
news_network (bridge)
├── postgres (172.18.0.2)
├── redis (172.18.0.3)
└── your-app (172.18.0.4)
    ↑
All can talk to each other by name!
```

---

## How Services Talk to Each Other

### Inside Containers (Service Names):
```python
# In your FastAPI app container
DATABASE_URL = "postgresql://user:pass@postgres:5432/db"
#                                       ↑
#                              Service name (not localhost!)

REDIS_URL = "redis://redis:6379/0"
#                    ↑
#              Service name
```

### From Your Computer (localhost):
```python
# On your computer (outside Docker)
DATABASE_URL = "postgresql://user:pass@localhost:5432/db"
#                                       ↑
#                              localhost (port mapped)

REDIS_URL = "redis://localhost:6379/0"
#                    ↑
#              localhost
```

---

## Common Commands

### Start everything:
```bash
docker-compose up
# Starts all services, shows logs

docker-compose up -d
# Starts in background (detached)
```

### Stop everything:
```bash
docker-compose down
# Stops and removes containers
# Keeps volumes (data safe!)

docker-compose down -v
# Stops and removes containers AND volumes
# DELETES ALL DATA! ⚠️
```

### View logs:
```bash
docker-compose logs
# All services

docker-compose logs postgres
# Just postgres

docker-compose logs -f
# Follow logs (like tail -f)
```

### Restart a service:
```bash
docker-compose restart postgres
# Restart just postgres

docker-compose restart
# Restart all services
```

### Check status:
```bash
docker-compose ps
# List running services

docker-compose top
# Show running processes
```

### Execute commands in containers:
```bash
# Open PostgreSQL shell
docker-compose exec postgres psql -U news_user -d news_recommender

# Open Redis shell
docker-compose exec redis redis-cli

# Run any command
docker-compose exec postgres ls /var/lib/postgresql
```

---

## Real-World Example: Full Workflow

### Day 1: First Setup
```bash
# Start services
docker-compose up -d

# Output:
# Creating network "news-recommender_news_network"
# Creating volume "news-recommender_postgres_data"
# Creating volume "news-recommender_redis_data"
# Creating news_postgres ... done
# Creating news_redis ... done

# Check they're running
docker-compose ps

# Output:
# Name                 State    Ports
# news_postgres        Up       0.0.0.0:5432->5432/tcp
# news_redis           Up       0.0.0.0:6379->6379/tcp
```

### Day 2: Working on Project
```bash
# Services still running from yesterday
docker-compose ps  # Shows "Up"

# If you shut down computer:
docker-compose up -d  # Start again, data still there!
```

### Day 3: Need to Reset Database
```bash
# Stop everything
docker-compose down

# Delete volumes (fresh start)
docker-compose down -v

# Start again (empty database)
docker-compose up -d
```

---

## Benefits of Docker Compose

### ✅ Consistency
```
Everyone on team uses same versions:
- PostgreSQL 15
- Redis 7
- Same configuration

"Works on my machine" → Works on everyone's machine!
```

### ✅ Simplicity
```bash
# One command instead of 10:
docker-compose up

# vs

docker run postgres ...
docker run redis ...
docker network create ...
docker volume create ...
# etc...
```

### ✅ Isolation
```
Multiple projects? No problem!

Project A:
  - postgres:5432
  - redis:6379

Project B:
  - postgres:5433  # Different port!
  - redis:6380

No conflicts!
```

### ✅ Easy Deployment
```yaml
# Same docker-compose.yml works:
- On your laptop (dev)
- On staging server
- On production server

Just change environment variables!
```

---

## Common Issues & Solutions

### Issue: Port already in use
```bash
Error: Bind for 0.0.0.0:5432 failed: port is already allocated
```

**Solution 1**: Stop other postgres
```bash
# Find process on port 5432
lsof -ti:5432
# Kill it
kill $(lsof -ti:5432)
```

**Solution 2**: Use different port
```yaml
ports:
  - "5433:5432"  # Use 5433 instead
```

### Issue: Permission denied
```bash
Error: Got permission denied while trying to connect to the Docker daemon
```

**Solution**:
```bash
sudo chmod 666 /var/run/docker.sock
# Or add yourself to docker group:
sudo usermod -aG docker $USER
# Then logout and login again
```

### Issue: Volume data corrupted
```bash
# Nuclear option: Delete everything and start fresh
docker-compose down -v
docker volume prune
docker-compose up -d
```

### Issue: Container won't start
```bash
# Check logs
docker-compose logs postgres

# Common cause: Old container still running
docker-compose down
docker-compose up -d
```

---

## Summary: Key Concepts

| Concept | What It Does | Example |
|---------|--------------|---------|
| **service** | One container | `postgres`, `redis` |
| **image** | What to run | `postgres:15-alpine` |
| **ports** | Expose to host | `5432:5432` |
| **volumes** | Persistent storage | `postgres_data:/var/lib/...` |
| **environment** | Container config | `POSTGRES_PASSWORD=...` |
| **networks** | Container communication | `news_network` |
| **healthcheck** | Monitor container | `pg_isready` |
| **depends_on** | Start order | `depends_on: [redis]` |

---

## Your docker-compose.yml in Plain English

```yaml
"Create two services:

1. PostgreSQL database:
   - Use PostgreSQL version 15 (lightweight)
   - Name it 'news_postgres'
   - Create a database called 'news_recommender'
   - Use username 'news_user' and password 'news_password_dev'
   - Let me access it at localhost:5432
   - Save data permanently in 'postgres_data' volume
   - Check every 10 seconds if it's ready
   - Connect to 'news_network'

2. Redis cache:
   - Use Redis version 7 (lightweight)
   - Name it 'news_redis'
   - Let me access it at localhost:6379
   - Save data permanently in 'redis_data' volume
   - Enable disk persistence (appendonly mode)
   - Check every 10 seconds if it's ready
   - Connect to 'news_network'

Make sure they can talk to each other on 'news_network'.
Keep all data safe in volumes even if containers restart."
```

---

Does this help clarify docker-compose.yml? Any specific parts you want me to explain more?