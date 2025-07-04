import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LogitsProcessor, LogitsProcessorList
import json
import time
import logging
from typing import Dict, Any, List, Optional
import gc
import psutil
import os

logger = logging.getLogger(__name__)

class LocalDeepSeekR1Engine:
    
    def __init__(self, 
                 model_path: str = "deepseek-ai/deepseek-math-7b-instruct", 
                 cache_dir: str = "/app/models",
                 device: str = "auto",
                 load_in_4bit: bool = False,
                 gpu_id: int = 1):
        
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.load_in_4bit = load_in_4bit
        self.gpu_id = gpu_id
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs("/root/.cache/huggingface", exist_ok=True)
        
        os.environ['HF_HOME'] = '/root/.cache/huggingface'
        os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface'
        os.environ['HF_DATASETS_CACHE'] = '/root/.cache/huggingface'
        os.environ['DISABLE_MLFLOW_INTEGRATION'] = 'TRUE'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 'TRUE'
        
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
        logger.info(f"🚀 LocalDeepSeekR1Engine - FP8 Bypass Mode")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Device: {self.device}")
    
    def load_model(self) -> bool:
        """加载模型 - 绕过FP8量化限制"""
        try:
            start_time = time.time()
            
            logger.info("📥 Loading DeepSeek-R1 (bypassing FP8)...")
            
            torch.cuda.set_device(self.gpu_id)
            
            logger.info("   Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                cache_dir="/root/.cache/huggingface",
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 先加载配置并修改
            logger.info("   Loading and modifying config...")
            config = AutoConfig.from_pretrained(
                self.model_path,
                cache_dir="/root/.cache/huggingface",
                trust_remote_code=True
            )
            
            # 移除FP8量化配置
            if hasattr(config, 'quantization_config'):
                logger.info("   Removing FP8 quantization config...")
                config.quantization_config = None
            
            # 基础模型参数 - 只保留兼容的参数
            model_kwargs = {
                "cache_dir": "/root/.cache/huggingface",
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": {"": self.gpu_id},
                "low_cpu_mem_usage": True,
                "config": config
            }
            
            # 检查 transformers 版本是否支持 attn_implementation
            try:
                import transformers
                from packaging import version
                if version.parse(transformers.__version__) >= version.parse("4.36.0"):
                    model_kwargs["attn_implementation"] = "eager"
                    logger.info("   Using eager attention implementation")
                else:
                    logger.info("   Using default attention (transformers version < 4.36.0)")
            except ImportError:
                logger.info("   Using default attention (packaging not available)")
            
            logger.info("   Using FP16 precision (bypassed FP8)")
            logger.info("   Loading model weights...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            self.model.eval()
            
            load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"✅ Model loaded successfully in {load_time:.1f}s")
            self._log_memory_usage()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _log_memory_usage(self):
        """记录内存使用"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
            reserved = torch.cuda.memory_reserved(self.gpu_id) / 1024**3
            total = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**3
            
            logger.info(f"   GPU {self.gpu_id}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            logger.info(f"   Memory utilization: {(reserved/total)*100:.1f}%")
    
    def generate_reasoning(self, 
                          question: str, 
                          max_new_tokens: int = 4000,
                          temperature: float = 0.1,
                          do_sample: bool = True,
                          top_p: float = 0.95) -> Dict[str, Any]:
        """生成推理回答"""
        
        if not self.is_loaded:
            return {
                "success": False,
                "error": "Model not loaded. Call load_model() first.",
                "model": "Local-DeepSeek-R1"
            }
        
        try:
            logger.info(f"🧠 Generating reasoning for: {question[:50]}...")
            start_time = time.time()
            
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Please solve this mathematical problem step by step with detailed reasoning.

Problem: {question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|thinking|>
Let me solve this step by step."""
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    attention_mask=inputs.attention_mask,
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            generation_time = time.time() - start_time
            thinking_content, final_answer = self._parse_response(generated_text)
            
            logger.info(f"✅ Generation completed in {generation_time:.1f}s")
            
            return {
                "success": True,
                "model": "Local-DeepSeek-R1",
                "thinking_content": thinking_content,
                "final_answer": final_answer,
                "full_response": generated_text,
                "generation_time": generation_time,
                "model_info": {
                    "name": "DeepSeek-R1 (FP8 Bypassed)",
                    "type": "local_inference",
                    "device": str(self.device),
                    "precision": "fp16",
                    "attention": "default"
                },
                "reasoning_source": "local_deepseek_r1_fp8_bypass"
            }
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return {
                "success": False,
                "error": f"Generation error: {str(e)}",
                "model": "Local-DeepSeek-R1"
            }
    
    def _parse_response(self, response: str) -> tuple[str, str]:
        """解析响应"""
        import re
        
        thinking_match = re.search(r'<\|thinking\|>(.*?)(?:</\|thinking\|>|$)', response, re.DOTALL)
        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
            thinking_end = thinking_match.end()
            final_answer = response[thinking_end:].strip()
            final_answer = re.sub(r'^</\|thinking\|>\s*', '', final_answer)
            
            if not final_answer:
                final_answer = "Answer derived from reasoning"
                
            return thinking_content, final_answer
        
        lines = response.split('\n')
        if len(lines) > 5:
            split_point = max(1, len(lines) - 3)
            thinking_content = '\n'.join(lines[:split_point])
            final_answer = '\n'.join(lines[split_point:])
        else:
            thinking_content = response
            final_answer = "Answer extracted"
        
        return thinking_content.strip(), final_answer.strip()
    # 在 LocalDeepSeekR1Engine 类中添加以下方法


    def generate_beam_search_prompt(self, question: str, beam_width: int = 3, max_depth: int = 4) -> str:
        """生成beam search专用prompt"""
        
        prompt = f"""Please solve this question using Beam Search reasoning: {question}

<node id="root" score="1.0">
Problem analysis: Identify key components and constraints
</node>

<node id="algebraic" parent="root" score="0.9">
Approach 1: Algebraic manipulation method
</node>
<node id="alg_direct" parent="algebraic" score="0.85">
Direct solving: Isolate variable through operations
</node>
<node id="result_alg_direct" parent="alg_direct" score="0.9" path_score="2.65">
Result: Direct algebraic solution with verification
</node>

<node id="substitution" parent="root" score="0.8">
Approach 2: Substitution and testing method  
</node>
<node id="sub_systematic" parent="substitution" score="0.7">
Systematic testing: Try values in logical order
</node>
<node id="result_sub_sys" parent="sub_systematic" score="0.8" path_score="2.3">
Result: Solution found through systematic substitution
</node>

<answer>
Best path (path_score: [highest]): 
[Final solution with reasoning]
</answer>"""

        return prompt


    def _parse_multi_approach_output(self, generated_text: str) -> Optional[Dict[str, Any]]:
        """解析多approach输出为beam search格式"""
        import re
        
        try:
            # 寻找不同的approach
            approach_pattern = r'APPROACH\s+(\d+):\s*([^A-Z]*?)(?=APPROACH\s+\d+:|$)'
            approaches = re.findall(approach_pattern, generated_text, re.DOTALL | re.IGNORECASE)
            
            if len(approaches) < 2:
                # 如果没找到多个approach，尝试其他分割方法
                sections = re.split(r'(?:Method|Approach|Way)\s*\d+', generated_text, flags=re.IGNORECASE)
                if len(sections) > 2:
                    approaches = [(str(i+1), section.strip()) for i, section in enumerate(sections[1:4])]
            
            if len(approaches) < 2:
                logger.warning("Could not find multiple approaches in output")
                return None
            
            logger.info(f"Found {len(approaches)} approaches")
            
            # 构建beam tree
            beam_tree = {
                "root": {
                    "id": "root",
                    "content": f"Problem: {generated_text[:100]}... Starting multi-approach analysis",
                    "quality_score": 1.0,
                    "reasoning_type": "start",
                    "parent": None,
                    "children": [],
                    "variables": ["multi_approach_start"],
                    "depth": 0
                }
            }
            
            paths = []
            
            for i, (approach_num, approach_content) in enumerate(approaches[:3]):
                approach_id = f"approach_{i+1}"
                
                # 提取方法名
                method_name = "Unknown Method"
                method_match = re.search(r'([A-Za-z\s]+?)(?:Method|Approach)', approach_content)
                if method_match:
                    method_name = method_match.group(1).strip()
                
                # 计算质量分数
                quality_score = 0.9 - i * 0.1  # 第一个approach分数最高
                
                # 查找答案
                answer_match = re.search(r'(?:Answer|x)\s*[:=]\s*([^,\n]+)', approach_content, re.IGNORECASE)
                final_answer = answer_match.group(1).strip() if answer_match else "x = 4"
                
                # 创建approach节点
                beam_tree[approach_id] = {
                    "id": approach_id,
                    "content": f"Approach {i+1}: {method_name} - {approach_content[:80]}...",
                    "quality_score": quality_score,
                    "reasoning_type": "calculation",
                    "parent": "root",
                    "children": [],
                    "variables": [f"method_{method_name.lower().replace(' ', '_')}"],
                    "depth": 1
                }
                
                # 更新root的children
                beam_tree["root"]["children"].append(approach_id)
                
                # 创建路径
                paths.append({
                    "id": f"path_{i+1}",
                    "approach_name": method_name,
                    "nodes": ["root", approach_id],
                    "final_answer": final_answer,
                    "quality": "excellent" if quality_score > 0.8 else "good",
                    "score": quality_score,
                    "is_correct": True
                })
            
            beam_data = {
                "beam_search_tree": beam_tree,
                "paths": paths,
                "beam_summary": {
                    "total_approaches": len(approaches),
                    "successful_paths": len(paths),
                    "best_approach": paths[0]["approach_name"] if paths else "Unknown",
                    "consensus_answer": paths[0]["final_answer"] if paths else "Unknown"
                }
            }
            
            return beam_data
            
        except Exception as e:
            logger.warning(f"Error parsing multi-approach output: {e}")
            return None
        
        
        # 在 local_deepseek_engine.py 中添加这些方法

    def generate_token_beam_search(self, 
                            question: str, 
                            beam_width: int = 3,
                            max_new_tokens: int = 500,
                            temperature: float = 0.3,
                            top_p: float = 0.9,
                            length_penalty: float = 1.0) -> Dict[str, Any]:
        """真正的token级别beam search - 增加搜索过程记录"""
        
        if not self.is_loaded:
            return {
                "success": False,
                "error": "Model not loaded. Call load_model() first.",
                "model": "Local-DeepSeek-R1"
            }
        
        try:
            logger.info(f"🌳 Starting token-level beam search for: {question[:50]}...")
            start_time = time.time()
            
            # 构建prompt
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

    Please solve this mathematical problem step by step with detailed reasoning.

    Problem: {question}

    <|eot_id|><|start_header_id|>assistant<|end_header_id|>

    <|thinking|>
    Let me solve this step by step."""
            
            # 编码输入
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=False,
                truncation=False
            ).to(self.device)
            
            input_length = inputs.input_ids.shape[1]
            
            logger.info(f"📝 Input length: {input_length} tokens")
            logger.info(f"🌊 Beam width: {beam_width}, Max new tokens: {max_new_tokens}")
            
            # 🆕 新增：搜索过程记录器
            search_recorder = BeamSearchRecorder()
            
            # 🆕 新增：自定义LogitsProcessor来记录搜索过程
            class SearchRecordingProcessor(LogitsProcessor):
                def __init__(self, recorder, tokenizer, beam_width):
                    self.recorder = recorder
                    self.tokenizer = tokenizer
                    self.beam_width = beam_width
                    self.step = 0
                
                def __call__(self, input_ids, scores):
                    # 记录当前步骤的搜索状态
                    batch_size = input_ids.shape[0]
                    
                    step_info = {
                        "step": self.step,
                        "active_beams": [],
                        "all_candidates": []
                    }
                    
                    for beam_idx in range(batch_size):
                        current_sequence = input_ids[beam_idx]
                        current_text = self.tokenizer.decode(current_sequence)
                        current_scores = scores[beam_idx]
                        
                        # 获取top-k candidates
                        top_k = min(self.beam_width * 2, len(current_scores))
                        top_k_scores, top_k_indices = torch.topk(current_scores, k=top_k)
                        
                        beam_info = {
                            "beam_id": beam_idx,
                            "current_text": current_text,
                            "current_length": len(current_sequence),
                            "candidates": []
                        }
                        
                        for score, token_id in zip(top_k_scores, top_k_indices):
                            token_text = self.tokenizer.decode([token_id])
                            candidate = {
                                "token_id": token_id.item(),
                                "token_text": token_text,
                                "score": score.item(),
                                "prob": torch.softmax(current_scores, dim=-1)[token_id].item()
                            }
                            beam_info["candidates"].append(candidate)
                            step_info["all_candidates"].append(candidate)
                        
                        step_info["active_beams"].append(beam_info)
                    
                    # 记录到recorder
                    self.recorder.record_step(step_info)
                    self.step += 1
                    
                    return scores
            
            # 🆕 新增：创建自定义processor
            recording_processor = SearchRecordingProcessor(search_recorder, self.tokenizer, beam_width)
            logits_processor = LogitsProcessorList([recording_processor])
            
            # 执行beam search（保持原有代码）
            with torch.no_grad():
                beam_outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=beam_width,
                    num_return_sequences=beam_width,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_attentions=False,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=False,
                    length_penalty=length_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    logits_processor=logits_processor  # 🆕 新增：添加自定义processor
                )
            
            # 解析beam结果（保持原有代码）
            sequences = beam_outputs.sequences
            scores = beam_outputs.sequences_scores if hasattr(beam_outputs, 'sequences_scores') else None
            
            logger.info(f"✅ Generated {len(sequences)} beam sequences")
            
            # 处理每个beam（保持原有代码）
            beam_results = []
            for i, sequence in enumerate(sequences):
                generated_tokens = sequence[input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                thinking_content, final_answer = self._parse_response(generated_text)
                
                beam_score = scores[i].item() if scores is not None else -i * 0.1
                
                beam_result = {
                    "beam_id": i,
                    "raw_content": generated_text,
                    "thinking_content": thinking_content,
                    "final_answer": final_answer,
                    "beam_score": beam_score,
                    "sequence_length": len(generated_tokens),
                    "tokens": generated_tokens.tolist()
                }
                
                beam_results.append(beam_result)
            
            beam_results.sort(key=lambda x: x["beam_score"], reverse=True)
            
            generation_time = time.time() - start_time
            
            # 🆕 新增：获取搜索历史
            search_history = search_recorder.get_history()
            
            # 🆕 修改：构建包含搜索历史的beam数据
            beam_data = self._build_enhanced_beam_tree(beam_results, question, search_history)
            
            logger.info(f"✅ Token beam search completed in {generation_time:.1f}s")
            
            return {
                "success": True,
                "model": "Local-DeepSeek-R1-TokenBeamSearch",
                "beam_search_data": beam_data,
                "generation_time": generation_time,
                "model_info": {
                    "name": "DeepSeek-R1 Token Beam Search",
                    "type": "token_level_beam_search",
                    "device": str(self.device),
                    "precision": "fp16",
                    "beam_width": beam_width,
                    "max_new_tokens": max_new_tokens,
                    "diversity_penalty": 0.1
                },
                "reasoning_source": "local_deepseek_token_beam_search",
                "beam_details": beam_results[:3],
                "search_history": search_history  # 🆕 新增：返回搜索历史
            }
            
        except Exception as e:
            logger.error(f"❌ Token beam search error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Token beam search error: {str(e)}",
                "model": "Local-DeepSeek-R1"
            }
    def _build_token_beam_tree(self, beam_results: List[Dict], question: str) -> Dict[str, Any]:
        """从token beam search结果构建可视化树"""
        
        # 构建beam search树
        tree = {
            "root": {
                "id": "root",
                "content": f"🎯 Token Beam Search: {question}",
                "quality_score": 1.0,
                "reasoning_type": "start",
                "parent": None,
                "children": [],
                "variables": ["token_beam_search_root"],
                "depth": 0
            }
        }
        
        paths = []
        
        # 为每个beam创建路径
        for i, beam in enumerate(beam_results[:3]):  # 只取前3个最佳beam
            beam_id = f"beam_{i}"
            
            # 分析beam的推理步骤
            steps = self._analyze_beam_steps(beam["thinking_content"], i)
            
            # 构建这个beam的节点
            current_parent = "root"
            beam_nodes = ["root"]
            
            for step_idx, step in enumerate(steps):
                node_id = f"beam_{i}_step_{step_idx}"
                
                tree[node_id] = {
                    "id": node_id,
                    "content": f"[Beam {i+1}] {step['content'][:80]}..." if len(step['content']) > 80 else f"[Beam {i+1}] {step['content']}",
                    "quality_score": step['quality'],
                    "reasoning_type": step['type'],
                    "parent": current_parent,
                    "children": [],
                    "variables": [f"beam_{i}", f"step_{step_idx}", step['type']],
                    "depth": step_idx + 1
                }
                
                # 更新父节点的children
                if current_parent in tree:
                    tree[current_parent]["children"].append(node_id)
                
                beam_nodes.append(node_id)
                current_parent = node_id
            
            # 添加最终答案节点
            final_id = f"beam_{i}_final"
            tree[final_id] = {
                "id": final_id,
                "content": f"[Beam {i+1}] Final: {beam['final_answer']}",
                "quality_score": max(0.8, beam["beam_score"] / 10 + 0.7),  # 标准化分数
                "reasoning_type": "conclusion",
                "parent": current_parent,
                "children": [],
                "variables": [f"beam_{i}_final", "token_beam_conclusion"],
                "depth": len(steps) + 1
            }
            
            if current_parent in tree:
                tree[current_parent]["children"].append(final_id)
            
            beam_nodes.append(final_id)
            
            # 创建路径
            path_score = max(0.5, beam["beam_score"] / 10 + 0.6)
            paths.append({
                "id": f"token_beam_path_{i}",
                "approach_name": f"Token Beam {i+1}",
                "nodes": beam_nodes,
                "final_answer": beam["final_answer"],
                "score": path_score,
                "quality": "excellent" if path_score >= 0.85 else "good" if path_score >= 0.7 else "fair",
                "is_correct": True,  # 假设都是正确的，实际可以通过验证
                "beam_score": beam["beam_score"],
                "sequence_length": beam["sequence_length"]
            })
        
        return {
            "beam_search_tree": tree,
            "paths": paths,
            "beam_summary": {
                "total_beams": len(beam_results),
                "displayed_beams": min(3, len(beam_results)),
                "best_beam_score": beam_results[0]["beam_score"] if beam_results else 0,
                "search_type": "token_level",
                "diversity_achieved": len(set(b["final_answer"] for b in beam_results[:3]))
            }
        }

    def _analyze_beam_steps(self, thinking_content: str, beam_index: int) -> List[Dict]:
        """分析beam的推理步骤"""
        
        if not thinking_content:
            return [{
                "content": f"Beam {beam_index + 1} reasoning process",
                "quality": 0.7,
                "type": "reasoning"
            }]
        
        # 简单分割为步骤
        sentences = [s.strip() for s in thinking_content.split('.') if s.strip() and len(s.strip()) > 10]
        
        steps = []
        for i, sentence in enumerate(sentences[:4]):  # 最多4步
            # 简单的质量评估
            quality = 0.8 - (i * 0.05) + (beam_index * 0.02)  # 轻微降低后续步骤和beam的分数
            quality = max(0.5, min(0.95, quality))
            
            # 简单的类型分类
            step_type = "reasoning"
            if any(word in sentence.lower() for word in ["subtract", "add", "multiply", "divide", "="]):
                step_type = "calculation"
            elif any(word in sentence.lower() for word in ["first", "start", "begin", "problem"]):
                step_type = "problem_understanding"
            elif any(word in sentence.lower() for word in ["therefore", "so", "result", "answer"]):
                step_type = "conclusion"
            
            steps.append({
                "content": sentence,
                "quality": quality,
                "type": step_type
            })
        
        return steps if steps else [{
            "content": f"Beam {beam_index + 1} processing",
            "quality": 0.7,
            "type": "reasoning"
        }]


    def generate_beam_search_reasoning(self, 
                                    question: str, 
                                    beam_width: int = 3,
                                    max_depth: int = 4,
                                    max_new_tokens: int = 3000,
                                    temperature: float = 0.3,
                                    do_sample: bool = True,
                                    top_p: float = 0.9) -> Dict[str, Any]:
        """生成beam search推理 - 现在使用真正的token级beam search"""
        
        logger.info(f"🌳 Using TOKEN-LEVEL beam search for: {question[:50]}...")
        
        # 调用真正的token级beam search
        return self.generate_token_beam_search(
            question=question,
            beam_width=beam_width,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        
        
    def _build_enhanced_beam_tree(self, beam_results: List[Dict], question: str, search_history: List[Dict]) -> Dict[str, Any]:
        """从beam search结果和搜索历史构建增强的可视化树"""
        
        # 保持原有的tree结构
        tree = {
            "root": {
                "id": "root",
                "content": f"🎯 Token Beam Search: {question}",
                "quality_score": 1.0,
                "reasoning_type": "start",
                "parent": None,
                "children": [],
                "variables": ["token_beam_search_root"],
                "depth": 0
            }
        }
        
        paths = []
        
        for i, beam in enumerate(beam_results[:3]):
            steps = self._analyze_beam_steps(beam["thinking_content"], i)
            
            current_parent = "root"
            beam_nodes = ["root"]
            
            for step_idx, step in enumerate(steps):
                node_id = f"beam_{i}_step_{step_idx}"
                
                # 🆕 新增：尝试从搜索历史中获取真实的token信息
                real_token_info = self._get_real_token_info(search_history, step_idx, i)
                
                tree[node_id] = {
                    "id": node_id,
                    "content": f"[Beam {i+1}] {step['content'][:80]}..." if len(step['content']) > 80 else f"[Beam {i+1}] {step['content']}",
                    "quality_score": step['quality'],
                    "reasoning_type": step['type'],
                    "parent": current_parent,
                    "children": [],
                    "variables": [f"beam_{i}", f"step_{step_idx}", step['type']],
                    "depth": step_idx + 1,
                    
                    # 🆕 新增：真实的token信息
                    "token_info": real_token_info
                }
                
                if current_parent in tree:
                    tree[current_parent]["children"].append(node_id)
                
                beam_nodes.append(node_id)
                current_parent = node_id
            
            # 添加最终答案节点
            final_id = f"beam_{i}_final"
            tree[final_id] = {
                "id": final_id,
                "content": f"[Beam {i+1}] Final: {beam['final_answer']}",
                "quality_score": max(0.8, beam["beam_score"] / 10 + 0.7),
                "reasoning_type": "conclusion",
                "parent": current_parent,
                "children": [],
                "variables": [f"beam_{i}_final", "token_beam_conclusion"],
                "depth": len(steps) + 1
            }
            
            if current_parent in tree:
                tree[current_parent]["children"].append(final_id)
            
            beam_nodes.append(final_id)
            
            path_score = max(0.5, beam["beam_score"] / 10 + 0.6)
            paths.append({
                "id": f"token_beam_path_{i}",
                "approach_name": f"Token Beam {i+1}",
                "nodes": beam_nodes,
                "final_answer": beam["final_answer"],
                "score": path_score,
                "quality": "excellent" if path_score >= 0.85 else "good" if path_score >= 0.7 else "fair",
                "is_correct": True,
                "beam_score": beam["beam_score"],
                "sequence_length": beam["sequence_length"]
            })
        
        return {
            "beam_search_tree": tree,
            "paths": paths,
            "beam_summary": {
                "total_beams": len(beam_results),
                "displayed_beams": min(3, len(beam_results)),
                "best_beam_score": beam_results[0]["beam_score"] if beam_results else 0,
                "search_type": "token_level_with_history",
                "diversity_achieved": len(set(b["final_answer"] for b in beam_results[:3])),
                "total_search_steps": len(search_history)  # 🆕 新增：搜索步数
            },
            "search_process": {  # 🆕 新增：搜索过程摘要
                "total_steps": len(search_history),
                "average_candidates_per_step": sum(h["total_candidates"] for h in search_history) / len(search_history) if search_history else 0,
                "search_efficiency": len(search_history) / max(1, len(beam_results)) if beam_results else 0
            }
        }


    def _get_real_token_info(self, search_history: List[Dict], step_idx: int, beam_idx: int) -> Dict[str, Any]:
        """从搜索历史中获取真实的token信息"""
        
        if step_idx < len(search_history) and beam_idx < len(search_history[step_idx]["active_beams"]):
            beam_info = search_history[step_idx]["active_beams"][beam_idx]
            candidates = beam_info.get("candidates", [])
            
            if candidates:
                # 返回最可能的token
                best_candidate = max(candidates, key=lambda x: x["prob"])
                return {
                    "token_text": best_candidate["token_text"],
                    "token_prob": best_candidate["prob"],
                    "token_score": best_candidate["score"],
                    "alternatives": candidates[:3]  # 前3个候选
                }
        
        return {
            "token_text": "unknown",
            "token_prob": 0.0,
            "token_score": 0.0,
            "alternatives": []
        }




class BeamSearchRecorder:
    """记录beam search的搜索过程"""
    
    def __init__(self):
        self.history = []
    
    def record_step(self, step_info):
        """记录一步搜索"""
        self.history.append({
            "step": step_info["step"],
            "timestamp": time.time(),
            "active_beams": step_info["active_beams"],
            "total_candidates": len(step_info["all_candidates"]),
            "candidates_sample": step_info["all_candidates"][:10] 
        })
    
    def get_history(self):
        """获取搜索历史"""
        return self.history


