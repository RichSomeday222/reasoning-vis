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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mac_beam_search_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ReasoningStep:
    """单个推理步骤"""
    step_id: int
    content: str
    confidence: float
    token_probabilities: List[float]
    is_valid: Optional[bool] = None
    step_type: str = "reasoning"
    
@dataclass
class ReasoningBranch:
    """推理分支"""
    branch_id: str
    steps: List[ReasoningStep]
    total_score: float
    final_answer: str
    is_correct: Optional[bool] = None
    generation_params: Dict[str, Any] = None
    thinking_process: str = ""
    solution_process: str = ""
    generation_success: bool = True
    generation_time: float = 0.0
    
@dataclass
class BeamSearchResult:
    """Beam Search结果"""
    problem_id: str
    original_question: str
    original_answer: str
    original_reasoning: str
    branches: List[ReasoningBranch]
    generation_timestamp: str
    model_used: str = "DeepSeek-R1-Mac"
    beam_params: Dict[str, Any] = None
    total_generations: int = 0
    successful_generations: int = 0

class MacOptimizedDeepSeekGenerator:
    """Mac优化的DeepSeek生成器 - 支持多种部署选项"""
    
    def __init__(self):
        """初始化生成器"""
        self.device_info = self._analyze_system()
        self.model = None
        self.tokenizer = None
        self.deployment_mode = self._select_deployment_mode()
        
        self.generation_stats = {
            'total_problems': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_branches': 0,
            'average_branch_score': 0.0,
            'average_generation_time': 0.0,
            'deployment_mode': self.deployment_mode
        }
        
        logger.info(f"🍎 Mac-optimized DeepSeek generator initialized")
        logger.info(f"🔧 Deployment mode: {self.deployment_mode}")
    
    def _analyze_system(self) -> Dict[str, Any]:
        """分析系统配置"""
        info = {
            'platform': 'mac',
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'torch_version': torch.__version__ if 'torch' in globals() else 'Not installed'
        }
        
        # 检查MPS支持
        try:
            info['mps_available'] = torch.backends.mps.is_available()
            info['cuda_available'] = torch.cuda.is_available()
        except:
            info['mps_available'] = False
            info['cuda_available'] = False
        
        # 检查芯片类型
        try:
            import platform
            machine = platform.machine()
            if machine == 'arm64':
                info['chip_type'] = 'Apple Silicon (M-series)'
                info['recommended_device'] = 'mps' if info['mps_available'] else 'cpu'
            else:
                info['chip_type'] = 'Intel'
                info['recommended_device'] = 'cpu'
        except:
            info['chip_type'] = 'Unknown'
            info['recommended_device'] = 'cpu'
        
        logger.info(f"💻 System analysis:")
        logger.info(f"   Chip: {info['chip_type']}")
        logger.info(f"   Memory: {info['memory_gb']:.1f} GB")
        logger.info(f"   CPU cores: {info['cpu_count']}")
        logger.info(f"   MPS available: {info['mps_available']}")
        logger.info(f"   Recommended device: {info['recommended_device']}")
        
        return info
    
    def _select_deployment_mode(self) -> str:
        """选择最佳部署模式"""
        memory_gb = self.device_info['memory_gb']
        
        if memory_gb >= 32 and self.device_info['mps_available']:
            return "local_full_mps" 
        elif memory_gb >= 16:
            if self.device_info['mps_available']:
                return "local_quantized_mps" 
            else:
                return "local_quantized_cpu"
        elif memory_gb >= 8:
            return "hybrid_api_local"  
        else:
            return "api_only"
    
    def find_latest_data_file(self, data_dir: str = "general_math_data") -> Optional[str]:
        """查找最新的数据文件"""
        logger.info(f"🔍 Searching for data files in {data_dir}")
        
        if not os.path.exists(data_dir):
            logger.error(f"❌ Directory {data_dir} does not exist")
            logger.info("💡 Please run the downloader first: python download_deepseek.py")
            return None
        
        pattern = os.path.join(data_dir, "*.jsonl")
        jsonl_files = glob.glob(pattern)
        
        if not jsonl_files:
            logger.error(f"❌ No .jsonl files found in {data_dir}")
            return None
        
        latest_file = max(jsonl_files, key=os.path.getmtime)
        file_size_mb = os.path.getsize(latest_file) / (1024 * 1024)
        
        logger.info(f"✅ Found data file: {latest_file}")
        logger.info(f"📊 File size: {file_size_mb:.1f} MB")
        
        return latest_file
    
    def load_model_if_needed(self):
        """根据部署模式加载模型"""
        
        if self.deployment_mode in ["api_only", "hybrid_api_local"]:
            logger.info("🌐 Using API mode - no local model loading needed")
            return
        
        if self.model is not None:
            logger.info("✅ Model already loaded")
            return
        
        logger.info(f"📥 Loading model for mode: {self.deployment_mode}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 使用更小的蒸馏版本
            
            # 加载tokenizer
            logger.info("🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 根据模式加载模型
            logger.info("🧠 Loading model...")
            
            if self.deployment_mode == "local_full_mps":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).to("mps")
                
            elif self.deployment_mode == "local_quantized_mps":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                
            elif self.deployment_mode == "local_quantized_cpu":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"🎯 Device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load local model: {e}")
            logger.info("🔄 Falling back to API mode...")
            self.deployment_mode = "api_only"
            self._setup_api_fallback()
    
    def _setup_api_fallback(self):
        """设置API后备方案"""
        try:
            import openai
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            
            if api_key:
                self.api_client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                    timeout=30.0
                )
                logger.info("🌐 API fallback configured")
            else:
                logger.warning("⚠️ No API key found - local generation only")
                
        except ImportError:
            logger.warning("⚠️ OpenAI package not available for API fallback")
    
    def generate_single_branch(self, 
                             question: str,
                             strategy: str = "default",
                             strategy_params: Dict[str, Any] = None) -> Optional[ReasoningBranch]:
        """生成单个推理分支"""
        
        start_time = time.time()
        
        try:
            if self.deployment_mode == "api_only" or (self.deployment_mode == "hybrid_api_local" and hasattr(self, 'api_client')):
                return self._generate_via_api(question, strategy, strategy_params, start_time)
            else:
                return self._generate_via_local(question, strategy, strategy_params, start_time)
                
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            
            # 如果是混合模式，尝试另一种方法
            if self.deployment_mode == "hybrid_api_local":
                try:
                    if hasattr(self, 'api_client'):
                        return self._generate_via_local(question, strategy, strategy_params, start_time)
                    else:
                        return self._generate_via_api(question, strategy, strategy_params, start_time)
                except:
                    pass
            
            return self._create_failed_branch(strategy, str(e), time.time() - start_time)
    
    def _generate_via_api(self, question: str, strategy: str, strategy_params: Dict[str, Any], start_time: float) -> Optional[ReasoningBranch]:
        """通过API生成"""
        logger.info(f"🌐 Generating via API with strategy: {strategy}")
        
        prompt = self._create_simple_prompt(question, strategy)
        
        response = self.api_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a math expert. Think step by step."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            stream=False
        )
        
        content = response.choices[0].message.content
        return self._process_generated_content(content, strategy, time.time() - start_time, "api")
    
    def _generate_via_local(self, question: str, strategy: str, strategy_params: Dict[str, Any], start_time: float) -> Optional[ReasoningBranch]:
        """通过本地模型生成"""
        logger.info(f"🖥️ Generating via local model with strategy: {strategy}")
        
        self.load_model_if_needed()
        
        if self.model is None:
            raise Exception("Local model not available")
        
        prompt = self._create_simple_prompt(question, strategy)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,  # 减少token数量以节省内存
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        content = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 清理内存
        del outputs, inputs
        if device.type in ['mps', 'cuda']:
            torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()
        
        return self._process_generated_content(content, strategy, time.time() - start_time, "local")
    
    def _create_simple_prompt(self, question: str, strategy: str) -> str:
        """创建简化的prompt"""
        if strategy == "systematic":
            return f"Solve this math problem step by step:\n\n{question}\n\nSolution:"
        elif strategy == "verification":
            return f"Solve and verify:\n\n{question}\n\nShow work and check answer:"
        else:
            return f"Math problem:\n\n{question}\n\nAnswer:"
    
    def _process_generated_content(self, content: str, strategy: str, generation_time: float, source: str) -> ReasoningBranch:
        """处理生成的内容"""
        
        # 简化的内容解析
        thinking_content, solution_content = self._extract_thinking_and_solution(content)
        steps = self._parse_reasoning_steps_simple(content)
        final_answer = self._extract_final_answer_simple(content)
        
        branch_id = f"mac_{source}_{strategy}_{int(time.time() * 1000)}"
        
        return ReasoningBranch(
            branch_id=branch_id,
            steps=steps,
            total_score=0.0,
            final_answer=final_answer,
            thinking_process=thinking_content,
            solution_process=solution_content,
            generation_success=True,
            generation_time=generation_time,
            generation_params={
                'strategy': strategy,
                'source': source,
                'deployment_mode': self.deployment_mode
            }
        )
    
    def _parse_reasoning_steps_simple(self, content: str) -> List[ReasoningStep]:
        """简化的步骤解析"""
        steps = []
        
        # 简单按行分割
        lines = [line.strip() for line in content.split('\n') if line.strip() and len(line.strip()) > 5]
        
        for i, line in enumerate(lines[:8]):  # 最多8步
            steps.append(ReasoningStep(
                step_id=i,
                content=line,
                confidence=0.8,
                token_probabilities=[0.8],
                step_type="reasoning"
            ))
        
        return steps
    
    def _extract_thinking_and_solution(self, content: str) -> Tuple[str, str]:
        """提取思维和解决方案"""
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""
        solution = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        if not solution and not thinking:
            solution = content.strip()
        
        return thinking, solution
    
    def _extract_final_answer_simple(self, content: str) -> str:
        """简化的答案提取"""
        patterns = [
            r'answer[:\s]*(.+?)(?:\n|$)',
            r'result[:\s]*(.+?)(?:\n|$)',
            r'\\boxed\{([^}]+)\}',
            r'\b([A-E])\b(?:\)|\.|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:50]
        
        # 返回最后一行
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return lines[-1][:50] if lines else "No answer found"
    
    def _create_failed_branch(self, strategy: str, error: str, generation_time: float) -> ReasoningBranch:
        """创建失败的分支"""
        return ReasoningBranch(
            branch_id=f"failed_{strategy}_{int(time.time())}",
            steps=[],
            total_score=0.0,
            final_answer="Generation failed",
            thinking_process="",
            solution_process="",
            generation_success=False,
            generation_time=generation_time,
            generation_params={'strategy': strategy, 'error': error}
        )
    
    def generate_beam_search_branches(self, 
                                    problem_data: Dict[str, Any], 
                                    problem_id: str,
                                    beam_size: int = 4) -> BeamSearchResult:
        """生成多分支推理（Mac优化版）"""
        
        question = problem_data.get('question', '')
        original_answer = problem_data.get('reference_answer', '')
        original_reasoning = problem_data.get('reasoning_content', '')
        
        logger.info(f"🌳 Generating {beam_size} branches for {problem_id}")
        logger.info(f"📊 Using deployment mode: {self.deployment_mode}")
        
        branches = []
        strategies = ["systematic", "direct", "verification"]
        
        try:
            for i in range(beam_size):
                strategy = strategies[i % len(strategies)]
                
                logger.info(f"🔄 Branch {i+1}/{beam_size} ({strategy})")
                
                branch = self.generate_single_branch(
                    question=question,
                    strategy=strategy
                )
                
                if branch:
                    branches.append(branch)
                
                # 内存清理
                gc.collect()
                time.sleep(0.5)  # 避免过热
            
            # 计算分支得分
            successful_branches = [b for b in branches if b.generation_success]
            for branch in successful_branches:
                branch.total_score = self._calculate_simple_score(branch, original_answer)
            
            # 排序
            successful_branches.sort(key=lambda x: x.total_score, reverse=True)
            failed_branches = [b for b in branches if not b.generation_success]
            final_branches = successful_branches + failed_branches
            
            # 创建结果
            result = BeamSearchResult(
                problem_id=problem_id,
                original_question=question,
                original_answer=original_answer,
                original_reasoning=original_reasoning,
                branches=final_branches,
                generation_timestamp=datetime.now().isoformat(),
                beam_params={
                    'beam_size': beam_size,
                    'deployment_mode': self.deployment_mode,
                    'device_info': self.device_info
                },
                total_generations=len(branches),
                successful_generations=len(successful_branches)
            )
            
            self.generation_stats['successful_generations'] += 1
            self.generation_stats['total_branches'] += len(final_branches)
            
            logger.info(f"✅ Generated {len(final_branches)} branches ({len(successful_branches)} successful)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to generate branches: {e}")
            self.generation_stats['failed_generations'] += 1
            return None
    
    def _calculate_simple_score(self, branch: ReasoningBranch, reference_answer: str) -> float:
        """简化的分支评分"""
        score = 0.0
        
        if branch.generation_success:
            score += 0.4
        
        if branch.steps:
            score += min(len(branch.steps) / 5, 0.3)
        
        if branch.final_answer and reference_answer:
            if branch.final_answer.lower() in reference_answer.lower() or reference_answer.lower() in branch.final_answer.lower():
                score += 0.3
        
        return score
    
    def process_dataset(self, 
                       input_file: str, 
                       output_dir: str = "mac_beam_results",
                       max_problems: int = 3,
                       beam_size: int = 3) -> Dict[str, Any]:
        """处理数据集（Mac版本）"""
        
        logger.info(f"🍎 Mac processing: {max_problems} problems, {beam_size} branches each")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        problems = self._load_data(input_file, max_problems)
        
        if not problems:
            logger.error("❌ No problems loaded")
            return {}
        
        logger.info(f"📋 Processing {len(problems)} problems")
        
        results = []
        for i, problem_data in enumerate(problems):
            try:
                problem_id = f"mac_{i:04d}"
                logger.info(f"🔄 Problem {i+1}/{len(problems)}: {problem_id}")
                
                beam_result = self.generate_beam_search_branches(
                    problem_data=problem_data,
                    problem_id=problem_id,
                    beam_size=beam_size
                )
                
                if beam_result:
                    results.append(beam_result)
                    
                    # 保存结果
                    result_file = output_path / f"{problem_id}_beam_search.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(asdict(beam_result), f, indent=2, ensure_ascii=False)
                    
                    success_rate = beam_result.successful_generations / beam_result.total_generations
                    logger.info(f"💾 Saved {problem_id} (success: {success_rate:.0%})")
                
                # 问题间休息
                if i < len(problems) - 1:
                    logger.info("⏸️ Brief pause...")
                    time.sleep(2)
                    gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Problem {i} failed: {e}")
                continue
        
        # 保存汇总
        summary_data = {
            'dataset_source': 'GR.inc general-math (Mac optimized)',
            'total_problems_processed': len(results),
            'generation_stats': self.generation_stats,
            'system_info': self.device_info,
            'deployment_mode': self.deployment_mode,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = output_path / "mac_beam_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🎉 Mac processing completed!")
        logger.info(f"📊 {len(results)} problems processed successfully")
        
        return summary_data
    
    def _load_data(self, input_file: str, max_problems: int) -> List[Dict]:
        """加载数据"""
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
    """主函数（Mac版本）"""
    
    logger.info("🍎 Starting Mac-Optimized DeepSeek Beam Search Generation...")
    
    try:
        # 创建生成器
        generator = MacOptimizedDeepSeekGenerator()
        
        # 查找数据文件
        input_file = generator.find_latest_data_file()
        if not input_file:
            logger.error("❌ No data file found")
            logger.info("💡 Run the downloader first: python download_deepseek.py")
            return
        
        # Mac友好的配置
        CONFIG = {
            "input_file": input_file,
            "output_dir": "mac_deepseek_beam_results",
            "max_problems": 2,  # 保守数量
            "beam_size": 3      # 适中的分支数
        }
        
        logger.info(f"📁 Data file: {CONFIG['input_file']}")
        logger.info("🍎 Mac optimizations:")
        logger.info("   • Automatic device detection")
        logger.info("   • Memory-efficient processing")
        logger.info("   • API fallback support")
        logger.info("   • Lightweight model variants")
        
        # 处理数据集
        summary = generator.process_dataset(
            input_file=CONFIG["input_file"],
            output_dir=CONFIG["output_dir"],
            max_problems=CONFIG["max_problems"],
            beam_size=CONFIG["beam_size"]
        )
        
        if summary:
            logger.info("🎉 Mac beam search generation completed!")
            logger.info(f"📊 Deployment mode used: {summary['deployment_mode']}")
            logger.info("📁 Check output directory for results")
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        logger.info("💡 Mac troubleshooting:")
        logger.info("   1. Install: pip install torch transformers")
        logger.info("   2. For Apple Silicon: ensure MPS support")
        logger.info("   3. Fallback: set DEEPSEEK_API_KEY for API mode")
        logger.info("   4. Check available memory with Activity Monitor")
        raise

if __name__ == "__main__":
    main()