from fastapi import APIRouter
from .route import router as cls_router

router = APIRouter()
router.include_router(cls_router, prefix="/classification", tags=["Classification"])