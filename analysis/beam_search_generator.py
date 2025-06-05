from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import torch
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import re
import glob
import gc
import psutil
from collections import defaultdict
import heapq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('beam_search_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TokenChoice:
    """Single token choice with probability"""
    token_id: int
    token_text: str
    probability: float
    cumulative_probability: float
    parent_path_id: Optional[str] = None

@dataclass
class BeamPath:
    """Single beam search path"""
    path_id: str
    token_sequence: List[int]
    token_texts: List[str]
    token_probabilities: List[float]
    cumulative_score: float
    is_complete: bool = False
    completion_reason: str = ""
    step_count: int = 0
    
@dataclass
class BeamSearchNode:
    """Node in beam search tree"""
    node_id: str
    parent_id: Optional[str]
    token_choice: TokenChoice
    children: List[str]
    depth: int
    is_leaf: bool = False
    
@dataclass
class TokenLevelBeamResult:
    """Token-level beam search result"""
    problem_id: str
    original_question: str
    original_answer: str
    beam_tree: Dict[str, BeamSearchNode]
    completed_paths: List[BeamPath]
    generation_stats: Dict[str, Any]
    generation_timestamp: str
    model_used: str = "DeepSeek-R1"

class TokenLevelBeamSearchGenerator:
    """Token-level beam search generator for reasoning visualization"""
    
    def __init__(self):
        """Initialize the generator"""
        self.device_info = self._analyze_system()
        self.model = None
        self.tokenizer = None
        self.deployment_mode = self._select_deployment_mode()
        
        self.generation_stats = {
            'total_problems': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_beam_width': 0.0,
            'average_sequence_length': 0.0,
            'average_generation_time': 0.0,
            'deployment_mode': self.deployment_mode
        }
        
        logger.info(f"🔄 Token-level beam search generator initialized")
        logger.info(f"🔧 Deployment mode: {self.deployment_mode}")
    
    def _analyze_system(self) -> Dict[str, Any]:
        """Analyze system configuration"""
        info = {
            'platform': 'cross_platform',
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'torch_version': torch.__version__ if 'torch' in globals() else 'Not installed'
        }
        
        # Check GPU support
        try:
            info['mps_available'] = torch.backends.mps.is_available()
            info['cuda_available'] = torch.cuda.is_available()
        except:
            info['mps_available'] = False
            info['cuda_available'] = False
        
        # Determine best device
        if info['cuda_available']:
            info['recommended_device'] = 'cuda'
        elif info['mps_available']:
            info['recommended_device'] = 'mps'
        else:
            info['recommended_device'] = 'cpu'
        
        logger.info(f"💻 System analysis:")
        logger.info(f"   Memory: {info['memory_gb']:.1f} GB")
        logger.info(f"   CPU cores: {info['cpu_count']}")
        logger.info(f"   CUDA available: {info['cuda_available']}")
        logger.info(f"   MPS available: {info['mps_available']}")
        logger.info(f"   Recommended device: {info['recommended_device']}")
        
        return info
    
    def _select_deployment_mode(self) -> str:
        """Select optimal deployment mode"""
        memory_gb = self.device_info['memory_gb']
        
        if memory_gb >= 32 and (self.device_info['cuda_available'] or self.device_info['mps_available']):
            return "local_full_gpu"
        elif memory_gb >= 16:
            return "local_quantized_gpu"
        elif memory_gb >= 8:
            return "hybrid_api_local"
        else:
            return "api_only"
    
    def load_model_if_needed(self):
        """Load model based on deployment mode"""
        
        if self.deployment_mode in ["api_only", "hybrid_api_local"]:
            logger.info("🌐 Using API mode - no local model loading needed")
            return
        
        if self.model is not None:
            logger.info("✅ Model already loaded")
            return
        
        logger.info(f"📥 Loading model for mode: {self.deployment_mode}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Use more capable model for better reasoning
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            
            # Load tokenizer
            logger.info("🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on mode
            logger.info("🧠 Loading model...")
            
            device = self.device_info['recommended_device']
            
            if self.deployment_mode == "local_full_gpu":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device in ['cuda', 'mps'] else torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).to(device)
                
            elif self.deployment_mode == "local_quantized_gpu":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device in ['cuda', 'mps'] else torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
            
            logger.info(f"✅ Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load local model: {e}")
            logger.info("🔄 Falling back to API mode...")
            self.deployment_mode = "api_only"
            self._setup_api_fallback()
    
    def _setup_api_fallback(self):
        """Setup API fallback"""
        try:
            import openai
            api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
            
            if api_key:
                if "deepseek" in api_key.lower() or os.environ.get("DEEPSEEK_API_KEY"):
                    self.api_client = openai.OpenAI(
                        api_key=api_key,
                        base_url="https://api.deepseek.com",
                        timeout=30.0
                    )
                    self.api_model = "deepseek-reasoner"
                else:
                    self.api_client = openai.OpenAI(api_key=api_key)
                    self.api_model = "gpt-3.5-turbo"
                    
                logger.info("🌐 API fallback configured")
            else:
                logger.warning("⚠️ No API key found - local generation only")
                
        except ImportError:
            logger.warning("⚠️ OpenAI package not available for API fallback")
    
    def generate_token_level_beam_search(self, 
                                       question: str,
                                       beam_width: int = 3,
                                       max_length: int = 150,
                                       min_length: int = 20) -> TokenLevelBeamResult:
        """Generate token-level beam search for reasoning"""
        
        start_time = time.time()
        logger.info(f"🌳 Starting token-level beam search")
        logger.info(f"   Beam width: {beam_width}")
        logger.info(f"   Max length: {max_length}")
        
        try:
            if self.deployment_mode == "api_only" or (hasattr(self, 'api_client')):
                return self._generate_beam_via_api(question, beam_width, max_length, min_length, start_time)
            else:
                return self._generate_beam_via_local(question, beam_width, max_length, min_length, start_time)
                
        except Exception as e:
            logger.error(f"❌ Beam search generation failed: {e}")
            return self._create_empty_result(question, str(e))
    
    def _generate_beam_via_local(self, question: str, beam_width: int, max_length: int, min_length: int, start_time: float) -> TokenLevelBeamResult:
        """Generate beam search using local model"""
        logger.info("🖥️ Generating beam search via local model")
        
        self.load_model_if_needed()
        
        if self.model is None:
            raise Exception("Local model not available")
        
        # Create initial prompt
        prompt = f"Solve this step by step:\n\n{question}\n\nSolution:"
        
        # Tokenize initial prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Initialize beam search state
        beam_tree = {}
        completed_paths = []
        active_beams = []
        
        # Create root node
        root_id = "root"
        beam_tree[root_id] = BeamSearchNode(
            node_id=root_id,
            parent_id=None,
            token_choice=TokenChoice(
                token_id=-1,
                token_text="<start>",
                probability=1.0,
                cumulative_probability=1.0
            ),
            children=[],
            depth=0
        )
        
        # Initialize first beam
        initial_beam = BeamPath(
            path_id="beam_0",
            token_sequence=inputs['input_ids'][0].tolist(),
            token_texts=[self.tokenizer.decode(tid) for tid in inputs['input_ids'][0]],
            token_probabilities=[1.0] * len(inputs['input_ids'][0]),
            cumulative_score=0.0,
            step_count=0
        )
        active_beams.append(initial_beam)
        
        # Main beam search loop
        for step in range(max_length):
            if not active_beams:
                break
                
            logger.info(f"🔄 Beam search step {step + 1}, active beams: {len(active_beams)}")
            
            new_beams = []
            
            for beam in active_beams:
                if beam.is_complete:
                    completed_paths.append(beam)
                    continue
                
                # Get next token candidates
                candidates = self._get_next_token_candidates(beam, inputs, beam_width)
                
                # Create new beams for each candidate
                for i, candidate in enumerate(candidates):
                    new_beam_id = f"beam_{step}_{beam.path_id}_{i}"
                    
                    # Create new beam path
                    new_beam = BeamPath(
                        path_id=new_beam_id,
                        token_sequence=beam.token_sequence + [candidate.token_id],
                        token_texts=beam.token_texts + [candidate.token_text],
                        token_probabilities=beam.token_probabilities + [candidate.probability],
                        cumulative_score=beam.cumulative_score + np.log(candidate.probability),
                        step_count=beam.step_count + 1
                    )
                    
                    # Check completion conditions
                    if (candidate.token_id == self.tokenizer.eos_token_id or 
                        len(new_beam.token_sequence) >= max_length or
                        self._is_reasoning_complete(new_beam.token_texts)):
                        new_beam.is_complete = True
                        new_beam.completion_reason = "eos" if candidate.token_id == self.tokenizer.eos_token_id else "max_length"
                        completed_paths.append(new_beam)
                    else:
                        new_beams.append(new_beam)
                    
                    # Add to beam tree
                    node_id = f"node_{step}_{i}"
                    beam_tree[node_id] = BeamSearchNode(
                        node_id=node_id,
                        parent_id=beam.path_id,
                        token_choice=candidate,
                        children=[],
                        depth=step + 1
                    )
            
            # Keep only top beams
            if new_beams:
                new_beams.sort(key=lambda x: x.cumulative_score, reverse=True)
                active_beams = new_beams[:beam_width]
            else:
                active_beams = []
        
        # Add any remaining active beams to completed
        completed_paths.extend(active_beams)
        
        # Clean up
        del inputs
        if device.type in ['mps', 'cuda']:
            if device.type == 'mps':
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()
        
        return TokenLevelBeamResult(
            problem_id=f"problem_{int(time.time())}",
            original_question=question,
            original_answer="",
            beam_tree=beam_tree,
            completed_paths=completed_paths,
            generation_stats={
                'generation_time': time.time() - start_time,
                'total_paths': len(completed_paths),
                'beam_width': beam_width,
                'max_length': max_length,
                'deployment_mode': self.deployment_mode
            },
            generation_timestamp=datetime.now().isoformat()
        )
    
    def _get_next_token_candidates(self, beam: BeamPath, original_inputs: Dict, beam_width: int) -> List[TokenChoice]:
        """Get next token candidates for current beam"""
        
        # Convert current sequence to tensor
        current_sequence = torch.tensor([beam.token_sequence], device=next(self.model.parameters()).device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(current_sequence)
            logits = outputs.logits[0, -1, :]  # Last token logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get top-k candidates
        top_k = min(beam_width * 2, len(probabilities))  # Get more candidates than beam width
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        candidates = []
        for prob, idx in zip(top_probs, top_indices):
            token_text = self.tokenizer.decode(idx.item())
            
            candidate = TokenChoice(
                token_id=idx.item(),
                token_text=token_text,
                probability=prob.item(),
                cumulative_probability=beam.cumulative_score + np.log(prob.item()),
                parent_path_id=beam.path_id
            )
            candidates.append(candidate)
        
        return candidates[:beam_width]
    
    def _is_reasoning_complete(self, token_texts: List[str]) -> bool:
        """Check if reasoning sequence is complete"""
        full_text = "".join(token_texts).lower()
        
        # Simple completion heuristics
        completion_indicators = [
            "therefore",
            "final answer",
            "conclusion",
            "the answer is",
            "result:",
            "\n\n"
        ]
        
        return any(indicator in full_text for indicator in completion_indicators)
    
    def _generate_beam_via_api(self, question: str, beam_width: int, max_length: int, min_length: int, start_time: float) -> TokenLevelBeamResult:
        """Generate beam search using API (simulated)"""
        logger.info("🌐 Generating beam search via API (simulated)")
        
        # For API mode, we simulate beam search by generating multiple completions
        # and creating artificial branching points
        
        beam_tree = {}
        completed_paths = []
        
        # Create root
        root_id = "root"
        beam_tree[root_id] = BeamSearchNode(
            node_id=root_id,
            parent_id=None,
            token_choice=TokenChoice(-1, "<start>", 1.0, 1.0),
            children=[],
            depth=0
        )
        
        # Generate multiple completions with different parameters
        strategies = [
            {"temperature": 0.3, "top_p": 0.8},
            {"temperature": 0.7, "top_p": 0.9},
            {"temperature": 0.5, "top_p": 0.85}
        ]
        
        for i, strategy in enumerate(strategies[:beam_width]):
            try:
                response = self.api_client.chat.completions.create(
                    model=self.api_model,
                    messages=[
                        {"role": "system", "content": "You are a step-by-step problem solver. Show your reasoning clearly."},
                        {"role": "user", "content": f"Solve this step by step:\n\n{question}"}
                    ],
                    max_tokens=max_length,
                    temperature=strategy["temperature"],
                    top_p=strategy["top_p"]
                )
                
                content = response.choices[0].message.content
                
                # Convert to token-like representation
                words = content.split()
                
                # Create artificial beam path
                path = BeamPath(
                    path_id=f"api_beam_{i}",
                    token_sequence=list(range(len(words))),  # Dummy token IDs
                    token_texts=words,
                    token_probabilities=[0.8 + 0.1 * np.random.random() for _ in words],
                    cumulative_score=np.random.uniform(0.6, 0.9),
                    is_complete=True,
                    completion_reason="api_complete",
                    step_count=len(words)
                )
                
                completed_paths.append(path)
                
                # Create artificial branching in tree
                for j, word in enumerate(words[:10]):  # Limit tree depth
                    node_id = f"api_node_{i}_{j}"
                    beam_tree[node_id] = BeamSearchNode(
                        node_id=node_id,
                        parent_id=f"api_node_{i}_{j-1}" if j > 0 else root_id,
                        token_choice=TokenChoice(j, word, 0.8 + 0.1 * np.random.random(), 0.8),
                        children=[],
                        depth=j + 1
                    )
                
            except Exception as e:
                logger.warning(f"⚠️ API call {i} failed: {e}")
                continue
        
        return TokenLevelBeamResult(
            problem_id=f"problem_{int(time.time())}",
            original_question=question,
            original_answer="",
            beam_tree=beam_tree,
            completed_paths=completed_paths,
            generation_stats={
                'generation_time': time.time() - start_time,
                'total_paths': len(completed_paths),
                'beam_width': beam_width,
                'max_length': max_length,
                'deployment_mode': self.deployment_mode
            },
            generation_timestamp=datetime.now().isoformat()
        )
    
    def _create_empty_result(self, question: str, error: str) -> TokenLevelBeamResult:
        """Create empty result for failed generation"""
        return TokenLevelBeamResult(
            problem_id=f"failed_{int(time.time())}",
            original_question=question,
            original_answer="",
            beam_tree={},
            completed_paths=[],
            generation_stats={'error': error},
            generation_timestamp=datetime.now().isoformat()
        )
    
    def find_latest_data_file(self, data_dir: str = "general_math_data") -> Optional[str]:
        """Find latest data file"""
        logger.info(f"🔍 Searching for data files in {data_dir}")
        
        if not os.path.exists(data_dir):
            logger.error(f"❌ Directory {data_dir} does not exist")
            return None
        
        pattern = os.path.join(data_dir, "*.jsonl")
        jsonl_files = glob.glob(pattern)
        
        if not jsonl_files:
            logger.error(f"❌ No .jsonl files found in {data_dir}")
            return None
        
        latest_file = max(jsonl_files, key=os.path.getmtime)
        logger.info(f"✅ Found data file: {latest_file}")
        
        return latest_file
    
    def process_dataset(self, 
                       input_file: str, 
                       output_dir: str = "beam_search_results",
                       max_problems: int = 3,
                       beam_width: int = 3) -> Dict[str, Any]:
        """Process dataset with token-level beam search"""
        
        logger.info(f"🔄 Processing dataset with token-level beam search")
        logger.info(f"   Max problems: {max_problems}")
        logger.info(f"   Beam width: {beam_width}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        problems = self._load_data(input_file, max_problems)
        
        if not problems:
            logger.error("❌ No problems loaded")
            return {}
        
        logger.info(f"📋 Processing {len(problems)} problems")
        
        results = []
        for i, problem_data in enumerate(problems):
            try:
                problem_id = f"problem_{i:04d}"
                question = problem_data.get('question', '')
                
                logger.info(f"🔄 Problem {i+1}/{len(problems)}: {problem_id}")
                
                # Generate token-level beam search
                beam_result = self.generate_token_level_beam_search(
                    question=question,
                    beam_width=beam_width,
                    max_length=100,
                    min_length=20
                )
                
                if beam_result and beam_result.completed_paths:
                    results.append(beam_result)
                    
                    # Save result
                    result_file = output_path / f"{problem_id}_token_beam_search.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(asdict(beam_result), f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"💾 Saved {problem_id} with {len(beam_result.completed_paths)} paths")
                
                # Brief pause between problems
                if i < len(problems) - 1:
                    time.sleep(1)
                    gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Problem {i} failed: {e}")
                continue
        
        # Save summary
        summary_data = {
            'dataset_source': 'Token-level beam search results',
            'total_problems_processed': len(results),
            'generation_stats': self.generation_stats,
            'system_info': self.device_info,
            'deployment_mode': self.deployment_mode,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = output_path / "token_beam_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🎉 Token-level beam search completed!")
        logger.info(f"📊 {len(results)} problems processed successfully")
        
        return summary_data
    
    def _load_data(self, input_file: str, max_problems: int) -> List[Dict]:
        """Load data from input file"""
        problems = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_problems:
                        break
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            if 'question' in data:
                                problems.append(data)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
        
        return problems

def main():
    """Main function for token-level beam search generation"""
    
    logger.info("🚀 Starting Token-Level Beam Search Generation...")
    
    try:
        # Create generator
        generator = TokenLevelBeamSearchGenerator()
        
        # Find data file
        input_file = generator.find_latest_data_file()
        if not input_file:
            logger.error("❌ No data file found")
            return
        
        # Configuration
        CONFIG = {
            "input_file": input_file,
            "output_dir": "token_beam_search_results",
            "max_problems": 2,
            "beam_width": 3
        }
        
        logger.info(f"📁 Data file: {CONFIG['input_file']}")
        logger.info("🔧 Configuration:")
        logger.info(f"   • Max problems: {CONFIG['max_problems']}")
        logger.info(f"   • Beam width: {CONFIG['beam_width']}")
        logger.info(f"   • Output directory: {CONFIG['output_dir']}")
        
        # Process dataset
        summary = generator.process_dataset(
            input_file=CONFIG["input_file"],
            output_dir=CONFIG["output_dir"],
            max_problems=CONFIG["max_problems"],
            beam_width=CONFIG["beam_width"]
        )
        
        if summary:
            logger.info("🎉 Token-level beam search generation completed!")
            logger.info(f"📊 Deployment mode used: {summary['deployment_mode']}")
            logger.info("📁 Check output directory for results")
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()