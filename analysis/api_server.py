from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
from collections import Counter
import asyncio

# Add current directory to import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_problems_data

# 导入本地 DeepSeek 生成器
try:
    from beam_search_generator import EnhancedLocalDeepSeekBeamGenerator
    LOCAL_DEEPSEEK_AVAILABLE = True
    print("✅ Local DeepSeek beam generator available")
except ImportError as e:
    LOCAL_DEEPSEEK_AVAILABLE = False
    print(f"⚠️ Local DeepSeek beam generator not available: {e}")


# 导入 O1 客户端
try:
    from enhanced_o1_client import EnhancedO1ReasoningClient
    O1_AVAILABLE = True
except ImportError:
    O1_AVAILABLE = False
    print("⚠️ O1ReasoningClient not available - O1 model will be disabled")

try:
    from analysis.test.test_o1_integration import O1UnifiedBeamSearchGenerator
    O1_BEAM_AVAILABLE = True
except ImportError:
    O1_BEAM_AVAILABLE = False
    print("⚠️ O1UnifiedBeamSearchGenerator not available; O1 beam search disabled")

# 测试 O1 客户端
print("Testing O1 import...")
try:
    from o1_reasoning_client import O1ReasoningClient
    test_client = O1ReasoningClient()
    print(f"✅ O1 client test: available={test_client.is_available()}")
except Exception as e:
    print(f"❌ O1 client test failed: {e}")

# Load data
problems_data, datasets_info = load_problems_data(max_math=5, max_engineering=5)

app = FastAPI(title="Enhanced Local DeepSeek Beam Search API", version="3.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
local_deepseek_generator = None
original_generator = None
o1_beam_generator = None
o1_client = None

class BeamSearchRequest(BaseModel):
    question: Optional[str] = None
    problem_id: Optional[str] = None
    model: str = "Local-DeepSeek-R1"  # 默认使用本地 DeepSeek
    beam_width: int = 3
    max_length: int = 150
    force_api: bool = False  # 强制使用 API 而不是本地模型

@app.on_event("startup")
async def startup_event():
    global local_deepseek_generator, original_generator, o1_client, o1_beam_generator
    
    print("🚀 Initializing Enhanced Local DeepSeek API server...")
    
    # 初始化本地 DeepSeek 生成器
    if LOCAL_DEEPSEEK_AVAILABLE:
        try:
            print("📥 Initializing local DeepSeek R1 engine...")
            local_deepseek_generator = EnhancedLocalDeepSeekBeamGenerator(
                model_path="deepseek-ai/deepseek-math-7b-instruct",
                cache_dir="/app/models",
                gpu_id=1,  # 使用 GPU 1
                use_local_engine=True,
                fallback_to_api=True
            )
            
            # 检查状态
            status = local_deepseek_generator.get_status()
            if status.get("local_engine_available", False):
                print("✅ Local DeepSeek R1 generator initialized successfully")
                print(f"   Status: {status}")
            else:
                print("⚠️ Local DeepSeek engine not available, will use API fallback")
                
        except Exception as e:
            print(f"❌ Local DeepSeek generator failed: {e}")
            local_deepseek_generator = None
    else:
        print("❌ Local DeepSeek generator not available - import failed")
    
    # 初始化 O1 客户端
    if O1_AVAILABLE:
        try:
            o1_client = EnhancedO1ReasoningClient(use_enhanced_parsing=True)
            if hasattr(o1_client, 'is_available') and o1_client.is_available():
                print("✅ O1 client initialized")
            else:
                print("⚠️ O1 client initialized but no API key")
        except Exception as e:
            print(f"❌ O1 client failed: {e}")
            o1_client = None
    
    if O1_BEAM_AVAILABLE:
        try:
            o1_beam_generator = O1UnifiedBeamSearchGenerator()
            print("✅ O1 Unified Beam Search generator initialized")
        except Exception as e:
            print(f"❌ O1 beam generator failed: {e}")
            o1_beam_generator = None
    
    print("🎉 Enhanced Local DeepSeek server startup completed.")

@app.get("/")
async def root():
    return {
        "message": "Enhanced Local DeepSeek Beam Search API is running.", 
        "version": "3.0.0",
        "features": ["Local DeepSeek R1", "Real-time Beam Search", "GPU Acceleration"]
    }

@app.get("/health")
async def health_check():
    local_status = {}
    if local_deepseek_generator:
        local_status = local_deepseek_generator.get_status()
    
    o1_available = False
    if o1_client:
        try:
            o1_available = hasattr(o1_client, 'is_available') and o1_client.is_available()
        except:
            o1_available = False
    
    return {
        "status": "healthy",
        "components": {
            "local_deepseek_generator": local_deepseek_generator is not None,
            "local_deepseek_status": local_status,
            "original_generator": original_generator is not None,
            "o1_client": o1_client is not None,
            "o1_available": o1_available
        },
        "datasets_loaded": len(datasets_info),
        "total_problems": len(problems_data),
        "api_version": "3.0.0"
    }

@app.get("/models")
async def get_available_models():
    """获取可用的推理模型"""
    
    # 检查本地 DeepSeek 状态
    local_deepseek_status = "unavailable"
    if local_deepseek_generator:
        status = local_deepseek_generator.get_status()
        if status.get("local_engine_loaded", False):
            local_deepseek_status = "available"
        elif status.get("local_engine_available", False):
            local_deepseek_status = "needs_loading"
        elif status.get("api_client_available", False):
            local_deepseek_status = "api_fallback"
    
    # 检查 O1 状态
    o1_available = False
    if o1_client:
        try:
            o1_available = hasattr(o1_client, 'is_available') and o1_client.is_available()
        except:
            o1_available = False
    
    models = [
        {
            "id": "Local-DeepSeek-R1",
            "name": "Local DeepSeek R1",
            "description": "Local DeepSeek R1 7B model running on GPU with real reasoning",
            "status": local_deepseek_status,
            "type": "local_model",
            "gpu_required": True,
            "features": ["Real reasoning", "Fast inference", "Beam search"]
        },
        {
            "id": "DeepSeek-R1",
            "name": "DeepSeek R1 (Simulated)",
            "description": "Simulated DeepSeek reasoning based on problem patterns",
            "status": "available" if original_generator else "unavailable",
            "type": "simulated"
        },
        {
            "id": "O1",
            "name": "OpenAI O1",
            "description": "Real OpenAI O1 reasoning model with thinking process",
            "status": "available" if o1_available else "unavailable",
            "type": "real_api",
            "api_required": "OPENAI_API_KEY"
        }
    ]
    
    return {"models": models}

@app.post("/generate")
async def generate_beam_search(request: BeamSearchRequest):
    try:
        # 获取问题
        if request.problem_id:
            selected_problem = None
            for p in problems_data:
                if p["id"] == request.problem_id:
                    selected_problem = p
                    break
            
            if not selected_problem:
                raise HTTPException(status_code=404, detail=f"Problem {request.problem_id} not found")
            
            question = selected_problem["question"]
            print(f"📚 Using problem from {selected_problem.get('dataset', 'unknown')}: {request.problem_id}")
            
        elif request.question:
            question = request.question
            print(f"✏️ Using user-provided question")
        else:
            raise HTTPException(status_code=400, detail="Either question or problem_id must be provided")

        # 根据模型选择生成方式
        if request.model == "Local-DeepSeek-R1":
            # 使用本地 DeepSeek R1 模型
            if not local_deepseek_generator:
                raise HTTPException(status_code=503, detail="Local DeepSeek R1 generator not available")
            
            print(f"🤖 Generating with Local DeepSeek R1...")
            result = await local_deepseek_generator.generate_reasoning_beam_search(
                question=question,
                beam_width=request.beam_width,
                max_depth=5,
                force_api=request.force_api
            )
            
            response_data = {
                "mode": "single",
                "model": "Local-DeepSeek-R1",
                "problem": result.problem,
                "beam_tree": result.beam_tree,
                "paths": result.paths,
                "model_info": {
                    "name": "Local DeepSeek R1",
                    "type": "local_model",
                    "gpu_accelerated": True,
                    "reasoning_source": result.problem.get("reasoning_source", "local_deepseek_r1")
                }
            }
            
        elif request.model == "O1":
            # 使用真实的 O1 API
            if not o1_client:
                raise HTTPException(status_code=503, detail="O1 model not available - O1ReasoningClient not imported")
            
            o1_available = False
            try:
                o1_available = hasattr(o1_client, 'is_available') and o1_client.is_available()
            except:
                pass
            
            if not o1_available:
                raise HTTPException(status_code=503, detail="O1 model not available - missing OPENAI_API_KEY")
            
            print(f"🧠 Generating with O1 model...")
            result = await o1_client.generate_reasoning(question, max_tokens=3000)
            
            if not result.get("success", False):
                raise HTTPException(status_code=500, detail=f"O1 generation failed: {result.get('error', 'Unknown error')}")
            
            response_data = {
                "mode": "single",
                "model": "O1",
                "problem": result.get("problem"),
                "beam_tree": result.get("beam_tree"),
                "paths": result.get("paths"),
                "model_info": result.get("model_info"),
                "reasoning_source": "real_o1_api"
            }
            
        elif request.model == "DeepSeek-R1":
            # 使用原有的模拟生成器
            if not original_generator:
                raise HTTPException(status_code=503, detail="Original DeepSeek generator not available")
            
            print(f"🎭 Generating with simulated DeepSeek...")
            result = await original_generator.generate_reasoning_beam_search(
                question=question,
                beam_width=request.beam_width,
                max_depth=5
            )
            
            response_data = {
                "mode": "single",
                "model": "DeepSeek-R1",
                "problem": result.problem,
                "beam_tree": result.beam_tree,
                "paths": result.paths,
                "reasoning_source": "simulated_deepseek"
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

        encoded = jsonable_encoder(response_data)
        return JSONResponse(content=encoded)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare_models(request: BeamSearchRequest):
    """对比本地 DeepSeek R1 和其他模型的推理结果"""
    
    # 对比模式：本地 DeepSeek R1 vs O1
    models_to_compare = ["Local-DeepSeek-R1", "O1"]
    
    try:
        # 获取问题
        if request.problem_id:
            selected_problem = None
            for p in problems_data:
                if p["id"] == request.problem_id:
                    selected_problem = p
                    break
            
            if not selected_problem:
                raise HTTPException(status_code=404, detail=f"Problem {request.problem_id} not found")
            
            question = selected_problem["question"]
            print(f"🔬 Comparing models for problem: {request.problem_id}")
            
        elif request.question:
            question = request.question
            print(f"🔬 Comparing models for custom question")
        else:
            raise HTTPException(status_code=400, detail="Either question or problem_id must be provided")

        # 并发生成两个模型的结果
        results = {}
        
        # 本地 DeepSeek R1 结果
        if local_deepseek_generator:
            try:
                print("🤖 Generating with Local DeepSeek R1...")
                local_result = await local_deepseek_generator.generate_reasoning_beam_search(
                    question=question,
                    beam_width=request.beam_width,
                    max_depth=5,
                    force_api=request.force_api
                )
                results["Local-DeepSeek-R1"] = {
                    "success": True,
                    "model": "Local-DeepSeek-R1",
                    "problem": local_result.problem,
                    "beam_tree": local_result.beam_tree,
                    "paths": local_result.paths,
                    "model_info": {
                        "name": "Local DeepSeek R1",
                        "type": "local_model",
                        "gpu_accelerated": True,
                        "reasoning_source": local_result.problem.get("reasoning_source", "local_deepseek_r1")
                    }
                }
                print("✅ Local DeepSeek R1 completed")
            except Exception as e:
                print(f"❌ Local DeepSeek R1 failed: {e}")
                results["Local-DeepSeek-R1"] = {
                    "success": False,
                    "error": str(e),
                    "model": "Local-DeepSeek-R1"
                }
        else:
            results["Local-DeepSeek-R1"] = {
                "success": False,
                "error": "Local DeepSeek R1 generator not available",
                "model": "Local-DeepSeek-R1"
            }
        
        # O1 真实结果  
        o1_available = False
        if o1_client:
            try:
                o1_available = hasattr(o1_client, 'is_available') and o1_client.is_available()
            except:
                pass
        
        if o1_available:
            try:
                print("🧠 Generating with O1...")
                o1_result = await o1_client.generate_reasoning(question, max_tokens=3000)
                results["O1"] = o1_result
                print("✅ O1 completed")
            except Exception as e:
                print(f"❌ O1 failed: {e}")
                results["O1"] = {
                    "success": False,
                    "error": str(e),
                    "model": "O1"
                }
        else:
            results["O1"] = {
                "success": False,
                "error": "O1 client not available - missing OPENAI_API_KEY or not imported",
                "model": "O1"
            }
        
        # 分析对比结果
        comparison_analysis = analyze_model_comparison(results)
        
        response_data = {
            "mode": "comparison",
            "question": question,
            "models": models_to_compare,
            "results": results,
            "comparison_analysis": comparison_analysis,
            "comparison_timestamp": datetime.now().isoformat()
        }
        
        encoded = jsonable_encoder(response_data)
        return JSONResponse(content=encoded)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Comparison error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_model")
async def load_local_model():
    """手动加载本地 DeepSeek 模型"""
    if not local_deepseek_generator:
        raise HTTPException(status_code=503, detail="Local DeepSeek generator not available")
    
    try:
        status = local_deepseek_generator.get_status()
        if status.get("local_engine_loaded", False):
            return {
                "message": "Model already loaded",
                "status": status
            }
        
        if not status.get("local_engine_available", False):
            raise HTTPException(status_code=503, detail="Local engine not available")
        
        # 触发模型加载
        print("📥 Manually loading DeepSeek R1 model...")
        if local_deepseek_generator.local_engine and not local_deepseek_generator.local_engine.is_loaded:
            success = local_deepseek_generator.local_engine.load_model()
            if not success:
                raise HTTPException(status_code=500, detail="Failed to load model")
        
        new_status = local_deepseek_generator.get_status()
        return {
            "message": "Model loaded successfully",
            "status": new_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

def analyze_model_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    """分析模型对比结果"""
    
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}
    
    analysis = {
        "total_models": len(results),
        "successful_models": len(successful_results),
        "failed_models": len(results) - len(successful_results),
        "model_performance": {},
        "reasoning_differences": {},
        "consensus_analysis": {}
    }
    
    if not successful_results:
        analysis["error"] = "No successful model results to compare"
        return analysis
    
    # 分析每个模型的表现
    for model, result in successful_results.items():
        paths = result.get("paths", [])
        if paths:
            best_path = max(paths, key=lambda p: p.get("score", 0))
            avg_score = sum(p.get("score", 0) for p in paths) / len(paths)
            
            analysis["model_performance"][model] = {
                "best_score": best_path.get("score", 0),
                "average_score": avg_score,
                "total_paths": len(paths),
                "best_answer": best_path.get("final_answer", "Unknown"),
                "reasoning_type": result.get("model_info", {}).get("type", "unknown")
            }
    
    # 分析推理差异
    if len(successful_results) >= 2:
        model_names = list(successful_results.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                result1 = successful_results[model1]
                result2 = successful_results[model2]
                
                # 比较推理深度
                depth1 = len(result1.get("beam_tree", {}))
                depth2 = len(result2.get("beam_tree", {}))
                
                # 比较最终答案
                paths1 = result1.get("paths", [])
                paths2 = result2.get("paths", [])
                
                answer1 = paths1[0].get("final_answer", "") if paths1 else ""
                answer2 = paths2[0].get("final_answer", "") if paths2 else ""
                
                comparison_key = f"{model1}_vs_{model2}"
                analysis["reasoning_differences"][comparison_key] = {
                    "reasoning_depth_difference": abs(depth1 - depth2),
                    "answer_agreement": answer1 == answer2,
                    "model1_depth": depth1,
                    "model2_depth": depth2,
                    "model1_answer": answer1,
                    "model2_answer": answer2
                }
    
    # 一致性分析
    answers = []
    for model, result in successful_results.items():
        paths = result.get("paths", [])
        if paths:
            best_path = max(paths, key=lambda p: p.get("score", 0))
            answers.append(best_path.get("final_answer", ""))
    
    if answers:
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0] if answer_counts else ("", 0)
        
        analysis["consensus_analysis"] = {
            "most_common_answer": most_common[0],
            "agreement_count": most_common[1],
            "total_models": len(answers),
            "agreement_percentage": (most_common[1] / len(answers)) * 100 if answers else 0,
            "all_answers": answers,
            "unique_answers": len(set(answers))
        }
    
    return analysis

@app.get("/problems")
async def get_problems():
    """获取所有可用问题"""
    return {
        "total": len(problems_data),
        "problems": [
            {
                "id": p["id"],
                "question": p["question"][:100] + "..." if len(p["question"]) > 100 else p["question"],
                "category": p["category"],
                "difficulty": p["difficulty"],
                "dataset": p["dataset"]
            }
            for p in problems_data
        ]
    }

@app.get("/datasets")
async def get_datasets():
    """获取所有可用数据集"""
    return {
        "total_datasets": len(datasets_info),
        "total_problems": len(problems_data),
        "datasets": [
            {
                "id": dataset_id,
                "name": info["name"],
                "description": info["description"], 
                "source": info["source"],
                "total_problems": info["total_problems"],
                "domain": info.get("domain", "unknown")
            }
            for dataset_id, info in datasets_info.items()
        ]
    }

@app.get("/datasets/{dataset_name}/problems")
async def get_dataset_problems(dataset_name: str):
    """获取特定数据集的问题"""
    dataset_problems = [p for p in problems_data if p["dataset"] == dataset_name]
    
    if not dataset_problems:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found or empty")
    
    return {
        "dataset": dataset_name,
        "total": len(dataset_problems),
        "problems": [
            {
                "id": p["id"],
                "question": p["question"][:100] + "..." if len(p["question"]) > 100 else p["question"],
                "category": p["category"],
                "difficulty": p["difficulty"]
            }
            for p in dataset_problems
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )