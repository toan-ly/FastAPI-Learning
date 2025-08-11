import sys
import time

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from utils.logger import Logger

LOGGER = Logger(__file__, log_file='http.log')

class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        LOGGER.log.info(
            f"{request.client.host} - "
            f"\"{request.method} {request.url.path} {request.scope['http_version']}\" "
            f"{response.status_code} {duration:.2f}s"
        )

        return response