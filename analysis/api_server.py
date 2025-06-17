from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import sys
import os
from typing import Dict, Any

# Add current directory to import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from beam_search_generator import UnifiedBeamSearchGenerator

app = FastAPI(title="Beam Search API", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = None  # Beam search generator instance

class BeamSearchRequest(BaseModel):
    question: str
    beam_width: int = 3
    max_length: int = 150

# （可选）保留文档用的 response model，但我们下面返回时会直接用 JSONResponse 绕过它
class BeamSearchResponse(BaseModel):
    problem: Dict[str, Any]
    beam_tree: Dict[str, Any]
    paths: list

@app.on_event("startup")
async def startup_event():
    global generator
    print("Initializing enhanced beam search generator...")
    generator = UnifiedBeamSearchGenerator()
    print("Enhanced generator is ready.")

@app.get("/")
async def root():
    return {"message": "Enhanced Beam Search API is running."}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "generator_loaded": generator is not None,
        "generator_type": "EnhancedBeamSearchGenerator"
    }

@app.post("/generate")  # 去掉 response_model，或保留也无所谓
async def generate_beam_search(request: BeamSearchRequest):
    if generator is None:
        raise HTTPException(status_code=500, detail="Generator not initialized")
    try:
        print(f"Processing question: {request.question[:50]}...")
        result = generator.generate_reasoning_beam_search(
            question=request.question,
            beam_width=request.beam_width,
            max_depth=5
        )

        # 原始返回结构
        response_data = {
            "problem": result.problem,
            "beam_tree": result.beam_tree,
            "paths": result.paths
        }

        # 先把 numpy 类型等都转成原生 Python
        encoded = jsonable_encoder(response_data)
        # 用 JSONResponse 直接返回，不再走 Pydantic response_model 校验
        return JSONResponse(content=encoded)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
