from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os
from typing import Dict, Any

# Add current directory to import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from beam_search_generator import TokenLevelBeamSearchGenerator

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

class BeamSearchResponse(BaseModel):
    problem_id: str
    original_question: str
    beam_tree: Dict[str, Any]
    completed_paths: list
    generation_stats: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    global generator
    print("Initializing beam search generator...")
    generator = TokenLevelBeamSearchGenerator()
    print("Generator is ready.")

@app.get("/")
async def root():
    return {"message": "Beam Search API is running."}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "generator_loaded": generator is not None,
        "deployment_mode": generator.deployment_mode if generator else "unknown"
    }

@app.post("/generate", response_model=BeamSearchResponse)
async def generate_beam_search(request: BeamSearchRequest):
    if generator is None:
        raise HTTPException(status_code=500, detail="Generator not initialized")

    try:
        print(f"Processing question: {request.question[:50]}...")

        result = generator.generate_token_level_beam_search(
            question=request.question,
            beam_width=request.beam_width,
            max_length=request.max_length,
            min_length=30
        )

        response_data = {
            "problem_id": result.problem_id,
            "original_question": result.original_question,
            "beam_tree": result.beam_tree,
            "completed_paths": [
                {
                    "path_id": path.path_id,
                    "token_sequence": path.token_sequence,
                    "token_texts": path.token_texts,
                    "token_probabilities": path.token_probabilities,
                    "cumulative_score": path.cumulative_score,
                    "is_complete": path.is_complete,
                    "completion_reason": path.completion_reason,
                    "step_count": path.step_count
                }
                for path in result.completed_paths
            ],
            "generation_stats": result.generation_stats
        }

        print(f"Generated {len(result.completed_paths)} paths.")
        return response_data

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Beam Search API Server...")
    print("API running at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True 
    )
