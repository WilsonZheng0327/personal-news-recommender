from fastapi import FastAPI, Response

# Cross-Origin Resource Sharing
#   allows frontend to talk to backend
#   without this, browser blocks requests
from fastapi.middleware.cors import CORSMiddleware
from config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="News Recommender API",
    version="0.1.0",
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "News Recommender API",
        "version": "0.1.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

'''
# @app.get("/health")
# async def health_check(db: Session = Depends(get_db)):
#     try:
#         # Check database connection
#         db.execute("SELECT 1")
        
#         # Check Redis connection
#         redis_client.ping()
        
#         # Check model loaded
#         if not classifier_model:
#             raise Exception("Model not loaded")
        
#         return {
#             "status": "ok",
#             "database": "connected",
#             "redis": "connected",
#             "model": "loaded"
#         }
#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "error": str(e)
#         }
'''

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn

    '''
    "backend.api.main:app"
        ↓         ↓     ↓
    package   module  variable

    Means:
    - Go to backend/api/main.py
    - Find the variable named "app"
    - That's the FastAPI instance to run
    '''
    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )