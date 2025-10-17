from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.db.database import test_postgres_connection
from app.services.embeddings import ensure_vector_store_ready
from app.services.storage import ensure_bucket_exists
from app.api.admin import router as admin_router
from app.api.compare import router as compare_router
from app.api.concepts import router as concepts_router
from app.api.evidence import router as evidence_router
from app.api.extraction import router as extraction_router
from app.api.graph import router as graph_router
from app.api.papers import router as papers_router
from app.api.relations import router as relations_router
from app.api.search import router as search_router
from app.api.sections import router as sections_router
from app.db.migrate import apply_migrations
from app.db.pool import init_pool, close_pool, get_pool


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:  # Basic error handling for MVP
            print(f"Request error: {exc}")  # Log the actual error
            return JSONResponse(status_code=500, content={"detail": str(exc) if exc else "Unknown error"})

    @app.get("/health")
    async def health():
        try:
            pool = get_pool()
        except RuntimeError as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "detail": str(exc)},
            )

        try:
            await pool.fetchval("SELECT 1;")
        except Exception as exc:  # noqa: PERF203
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "detail": f"Database health check failed: {exc}",
                },
            )

        return {"status": "ok"}

    @app.on_event("startup")
    async def on_startup():
        print("Checking Postgres connection...")
        await test_postgres_connection()
        print("Postgres connection successful")

        print("Ensuring MinIO bucket exists...")
        await ensure_bucket_exists(settings.minio_bucket_papers)
        print("MinIO bucket check completed")

        print("Ensuring Qdrant collection exists...")
        await ensure_vector_store_ready()
        print("Qdrant collection ready")

        print("Starting database migrations...")
        await apply_migrations()
        print("Migrations completed successfully")

        print("Initializing database pool...")
        await init_pool()
        print("Database pool initialized successfully")

    @app.on_event("shutdown")
    async def on_shutdown():
        try:
            await close_pool()
        except Exception as exc:  # noqa: PERF203
            print(f"Shutdown DB pool close failed: {exc}")

    # Routers
    app.include_router(papers_router, prefix=settings.api_prefix)
    app.include_router(sections_router, prefix=settings.api_prefix)
    app.include_router(concepts_router, prefix=settings.api_prefix)
    app.include_router(relations_router, prefix=settings.api_prefix)
    app.include_router(evidence_router, prefix=settings.api_prefix)
    app.include_router(search_router, prefix=settings.api_prefix)
    app.include_router(graph_router, prefix=settings.api_prefix)
    app.include_router(compare_router, prefix=settings.api_prefix)
    app.include_router(extraction_router, prefix=settings.api_prefix)
    app.include_router(admin_router, prefix=settings.api_prefix)

    return app


app = create_app()

