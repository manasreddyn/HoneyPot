import os
import sys
from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load Env
load_dotenv()
API_KEY = os.getenv("HONEYPOT_API_KEY")

class HoneypotBypassMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/api/honeypot":
            api_key = request.headers.get("x-api-key")

            # Strict check: header must match expected key
            if not API_KEY or api_key != API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"}
                )

            # Bypass EVERYTHING else and return success
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "reply": "Message appears legitimate"
                }
            )

        return await call_next(request)

app = FastAPI(title="NEXUS-GUARDIAN API")

# Add Middleware FIRST so it runs before anything else
app.add_middleware(HoneypotBypassMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "online", "system": "NEXUS-GUARDIAN v4.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
