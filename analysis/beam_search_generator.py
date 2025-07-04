import torch
import json
import time
import logging
import numpy as np
import os
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

try:
    from local_deepseek_engine import LocalDeepSeekR1Engine
    LOCAL_DEEPSEEK_AVAILABLE = True
except ImportError:
    LOCAL_DEEPSEEK_AVAILABLE = False
    print("⚠️ Local DeepSeek engine not available")

try:
    from deepseek_reasoning_client import DeepSeekReasoningClient
    DEEPSEEK_CLIENT_AVAILABLE = True
except ImportError:
    DEEPSEEK_CLIENT_AVAILABLE = False
    print("⚠️ DeepSeek client not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FrontendBeamNode:
    """Node format matching frontend requirements"""
    id: str
    content: str
    reasoning_type: str
    quality_score: float
    probability: float
    parent: Optional[str]
    children: List[str]
    variables: List[str]
    depth: int

@dataclass
class FrontendPath:
    """Path format matching frontend requirements"""
    id: str
    nodes: List[str]
    quality: str
    score: float
    is_correct: bool
    final_answer: str

@dataclass
class FrontendBeamResult:
    """Complete result format for frontend"""
    problem: Dict[str, Any]
    beam_tree: Dict[str, Dict[str, Any]]
    paths: List[Dict[str, Any]]

class LocalDeepSeekReasoningParser:
    """解析本地 DeepSeek R1 推理结果"""
    
    def __init__(self):
        # 推理类型关键词
        self.reasoning_keywords = {
            'problem_understanding': [
                'understand', 'analyze', 'this problem', 'we need to',
                'first', 'looking at', 'the question', 'given'
            ],
            'calculation': [
                'calculate', 'compute', 'sum', 'multiply', 'divide', 'equals',
                'formula', 'substitute', 'simplify', 'solve', 'step'
            ],
            'verification': [
                'check', 'verify', 'confirm', 'make sure',
                'does this make sense', 'validation', 'double check'
            ],
            'correction': [
                'wait', 'actually', 'no', 'wrong', 'reconsider',
                'error', 'correction', 'instead', 'mistake'
            ],
            'conclusion': [
                'therefore', 'so', 'final answer', 'answer is', 'conclusion',
                'result', 'hence', 'summary'
            ]
        }
    
    def parse_local_deepseek_result(self, deepseek_result: Dict[str, Any]) -> Tuple[Dict[str, FrontendBeamNode], List[FrontendPath]]:
        """解析本地 DeepSeek 结果为 beam search 格式"""
        
        if not deepseek_result.get("success", False):
            # 返回错误节点
            error_tree = {
                "root": FrontendBeamNode(
                    id="root",
                    content=f"Local DeepSeek Error: {deepseek_result.get('error', 'Unknown error')}",
                    reasoning_type="error",
                    quality_score=0.0,
                    probability=0.0,
                    parent=None,
                    children=[],
                    variables=["error"],
                    depth=0
                )
            }
            return error_tree, []
        
        thinking_content = deepseek_result.get("thinking_content", "")
        final_answer = deepseek_result.get("final_answer", "")
        
        # 分割思考过程为步骤
        thinking_steps = self._split_thinking_text(thinking_content)
        
        # 构建 beam tree
        beam_tree = self._build_beam_tree_from_steps(thinking_steps, final_answer)
        
        # 提取路径
        paths = self._extract_paths_from_tree(beam_tree)
        
        return beam_tree, paths
    
    def _split_thinking_text(self, text: str) -> List[str]:
        """智能分割思考文本为步骤"""
        if not text:
            return ["No thinking process available"]
        
        # 按照段落、句子等分割
        patterns = [
            r'\n\n+',  # 段落分割
            r'\.\s+(?=[A-Z])',  # 句子分割（大写字母开头）
            r'(?<=\.)\s*(?=Step|Now|Next|So|The|First|Second|Third)',  # 步骤标识
            r'(?<=:)\s*(?=[A-Z])',  # 冒号后分割
        ]
        
        sentences = [text]
        for pattern in patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = new_sentences
        
        # 清理和过滤
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 20:  # 过滤太短的句子
                sentence = re.sub(r'\s+', ' ', sentence)  # 标准化空白
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences if cleaned_sentences else ["Reasoning step processed"]
    
    def _build_beam_tree_from_steps(self, steps: List[str], final_answer: str) -> Dict[str, FrontendBeamNode]:
        """从步骤构建 beam tree"""
        tree = {}
        
        # 根节点
        tree["root"] = FrontendBeamNode(
            id="root",
            content="[Local DeepSeek R1] Starting mathematical problem analysis",
            reasoning_type="start",
            quality_score=1.0,
            probability=1.0,
            parent=None,
            children=[],
            variables=["local_deepseek_start"],
            depth=0
        )
        
        previous_node_id = "root"
        
        # 处理每个思考步骤
        for i, step in enumerate(steps):
            node_id = f"step_{i}"
            
            # 分类步骤类型
            step_type = self._classify_step_type(step)
            
            # 计算质量分数
            quality_score = self._calculate_step_quality(step, step_type)
            
            # 创建节点
            node = FrontendBeamNode(
                id=node_id,
                content=f"[Local DeepSeek] {step[:100]}..." if len(step) > 100 else f"[Local DeepSeek] {step}",
                reasoning_type=step_type,
                quality_score=quality_score,
                probability=quality_score * 0.9,
                parent=previous_node_id,
                children=[],
                variables=[f"local_step_{i}", step_type],
                depth=i + 1
            )
            
            tree[node_id] = node
            
            # 更新父节点的 children
            if previous_node_id in tree:
                tree[previous_node_id].children.append(node_id)
            
            previous_node_id = node_id
        
        # 添加最终答案节点
        final_node_id = "final_answer"
        final_node = FrontendBeamNode(
            id=final_node_id,
            content=f"[Local DeepSeek] Final Answer: {final_answer}",
            reasoning_type="conclusion",
            quality_score=0.95,
            probability=0.92,
            parent=previous_node_id,
            children=[],
            variables=["local_final_answer", "conclusion"],
            depth=len(steps) + 1
        )
        
        tree[final_node_id] = final_node
        
        if previous_node_id in tree:
            tree[previous_node_id].children.append(final_node_id)
        
        return tree
    
    def _classify_step_type(self, text: str) -> str:
        """分类步骤类型"""
        text_lower = text.lower()
        
        scores = {}
        for step_type, keywords in self.reasoning_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[step_type] = score
        
        if not scores:
            return 'reasoning'
        
        return max(scores, key=scores.get)
    
    def _calculate_step_quality(self, text: str, step_type: str) -> float:
        """计算步骤质量分数"""
        base_score = 0.6
        
        # 长度奖励
        length_bonus = min(0.2, len(text) / 200)
        
        # 数学关键词奖励
        math_keywords = ['=', '+', '-', '*', '/', 'equation', 'formula', 'calculate']
        math_score = sum(0.03 for keyword in math_keywords if keyword in text.lower())
        
        # 确定性关键词
        certainty_keywords = ['clearly', 'obviously', 'definitely', 'therefore']
        uncertainty_keywords = ['maybe', 'perhaps', 'might', 'possibly']
        
        certainty_score = sum(0.05 for keyword in certainty_keywords if keyword in text.lower())
        uncertainty_penalty = sum(0.05 for keyword in uncertainty_keywords if keyword in text.lower())
        
        # 步骤类型奖励
        type_bonus = {
            'calculation': 0.15,
            'verification': 0.1,
            'conclusion': 0.2,
            'correction': -0.05,
            'problem_understanding': 0.08
        }.get(step_type, 0.05)
        
        final_score = (base_score + length_bonus + math_score + certainty_score + 
                      type_bonus - uncertainty_penalty)
        
        return max(0.2, min(0.98, final_score))
    
    def _extract_paths_from_tree(self, beam_tree: Dict[str, FrontendBeamNode]) -> List[FrontendPath]:
        """从 tree 提取路径"""
        paths = []
        
        # 找到主路径（从 root 到叶子节点）
        def find_main_path(node_id: str, current_path: List[str]):
            node = beam_tree[node_id]
            current_path = current_path + [node_id]
            
            if not node.children:  # 叶子节点
                # 计算路径质量
                path_scores = [beam_tree[nid].quality_score for nid in current_path]
                avg_score = float(np.mean(path_scores))
                
                # 确定质量标签
                if avg_score >= 0.85:
                    quality = "excellent"
                elif avg_score >= 0.7:
                    quality = "good"
                elif avg_score >= 0.5:
                    quality = "fair"
                else:
                    quality = "poor"
                
                # 提取最终答案
                final_answer = self._extract_answer_from_content(node.content)
                
                # 创建路径
                path = FrontendPath(
                    id=f"local_path_{len(paths)}",
                    nodes=current_path,
                    quality=quality,
                    score=avg_score,
                    is_correct=avg_score > 0.75,
                    final_answer=final_answer
                )
                paths.append(path)
            else:
                # 继续遍历子节点
                for child_id in node.children:
                    if child_id in beam_tree:
                        find_main_path(child_id, current_path)
        
        find_main_path("root", [])
        
        # 按分数排序
        paths.sort(key=lambda p: p.score, reverse=True)
        return paths[:3]  # 返回前3条路径
    
    def _extract_answer_from_content(self, content: str) -> str:
        """从内容中提取答案"""
        # 尝试提取选择题答案
        answer_match = re.search(r'\b([A-E])\)', content)
        if answer_match:
            return answer_match.group(1)
        
        # 尝试提取数字答案
        number_match = re.search(r'answer.*?(\d+(?:\.\d+)?)', content, re.IGNORECASE)
        if number_match:
            return number_match.group(1)
        
        # 尝试提取等式
        equals_match = re.search(r'x\s*=\s*(\d+(?:\.\d+)?)', content)
        if equals_match:
            return f"x = {equals_match.group(1)}"
        
        return "A"  # 默认返回

class ProblemClassifier:
    """Classify mathematical problem types"""
    
    def __init__(self):
        self.patterns = {
            "sequence": [
                "sequence", "series", "sum", "term", "arithmetic", "geometric",
                "first.*terms", "nth term", "Σ", "summation", "consecutive"
            ],
            "geometry": [
                "triangle", "circle", "angle", "area", "perimeter", "volume",
                "rectangle", "square", "polygon", "diameter", "radius"
            ],
            "algebra": [
                "equation", "solve", "variable", "polynomial", "factor",
                "quadratic", "linear", "system", "inequality", "roots"
            ],
            "probability": [
                "probability", "chance", "random", "dice", "coin", "card",
                "outcome", "event", "sample space", "conditional"
            ],
            "calculus": [
                "derivative", "integral", "limit", "continuous", "differential",
                "rate of change", "optimization", "maximum", "minimum"
            ]
        }
    
    def classify_problem(self, question: str) -> str:
        """Classify problem type based on keywords"""
        question_lower = question.lower()
        
        scores = {}
        for problem_type, keywords in self.patterns.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                scores[problem_type] = score
        
        if not scores:
            return "general"
        
        return max(scores, key=scores.get)

class EnhancedLocalDeepSeekBeamGenerator:
    """增强的本地 DeepSeek Beam Search 生成器"""
    
    def __init__(self, 
                 model_path: str = "deepseek-ai/deepseek-math-7b-instruct",
                 cache_dir: str = "/app/models",
                 gpu_id: int = 1,
                 use_local_engine: bool = True,
                 fallback_to_api: bool = True):
        """
        Args:
            model_path: 本地模型路径
            cache_dir: 缓存目录
            gpu_id: GPU ID
            use_local_engine: 是否使用本地引擎
            fallback_to_api: 是否回退到 API
        """
        self.device = self._setup_device()
        self.classifier = ProblemClassifier()
        self.parser = LocalDeepSeekReasoningParser()
        self.use_local_engine = use_local_engine
        self.fallback_to_api = fallback_to_api
        
        # 初始化本地 DeepSeek 引擎
        self.local_engine = None
        if use_local_engine and LOCAL_DEEPSEEK_AVAILABLE:
            try:
                self.local_engine = LocalDeepSeekR1Engine(
                    model_path=model_path,
                    cache_dir=cache_dir,
                    gpu_id=gpu_id
                )
                logger.info("✅ Local DeepSeek engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize local DeepSeek engine: {e}")
                if not fallback_to_api:
                    raise
                self.local_engine = None
        
        # 初始化 API 客户端作为备用
        self.deepseek_client = None
        if fallback_to_api and DEEPSEEK_CLIENT_AVAILABLE:
            try:
                self.deepseek_client = DeepSeekReasoningClient()
                if self.deepseek_client.is_available():
                    logger.info("✅ DeepSeek API client available as fallback")
            except Exception as e:
                logger.error(f"❌ Failed to initialize DeepSeek client: {e}")
                self.deepseek_client = None
        
        logger.info(f"Enhanced Local DeepSeek generator initialized")
        logger.info(f"  Local engine: {self.local_engine is not None}")
        logger.info(f"  API fallback: {self.deepseek_client is not None and self.deepseek_client.is_available()}")
    
    def _setup_device(self):
        """Setup compute device"""
        if torch.cuda.is_available():
            device = torch.device('cuda:1')  # 使用 GPU 1
            logger.info(f"Using GPU: {torch.cuda.get_device_name(1)}")
            return device
        else:
            return torch.device('cpu')
    
    async def generate_reasoning_beam_search(
        self, 
        question: str, 
        beam_width: int = 3,
        max_depth: int = 5,
        force_api: bool = False
    ) -> FrontendBeamResult:
        """生成 beam search 推理"""
        
        start_time = time.time()
        
        # 确定使用哪种方法
        use_local = (
            self.use_local_engine and 
            self.local_engine and 
            not force_api
        )
        
        if use_local:
            logger.info("🤖 Using local DeepSeek R1 engine")
            try:
                result = await self._generate_with_local_engine(question)
                generation_time = time.time() - start_time
                logger.info(f"✅ Local DeepSeek completed in {generation_time:.2f}s")
                return result
            except Exception as e:
                logger.error(f"❌ Local DeepSeek failed: {e}")
                if self.fallback_to_api and self.deepseek_client and self.deepseek_client.is_available():
                    logger.info("🔄 Falling back to DeepSeek API")
                    return await self._generate_with_api(question)
                else:
                    raise
        else:
            if self.deepseek_client and self.deepseek_client.is_available():
                logger.info("🌐 Using DeepSeek API")
                result = await self._generate_with_api(question)
                generation_time = time.time() - start_time
                logger.info(f"✅ DeepSeek API completed in {generation_time:.2f}s")
                return result
            else:
                raise Exception("No DeepSeek engine available (local or API)")
    
    async def _generate_with_local_engine(self, question: str) -> FrontendBeamResult:
        """使用本地引擎生成推理"""
        
        # 确保模型已加载
        if not self.local_engine.is_loaded:
            logger.info("📥 Loading local DeepSeek model...")
            if not self.local_engine.load_model():
                raise Exception("Failed to load local DeepSeek model")
        
        # 生成推理
        deepseek_result = self.local_engine.generate_reasoning(
            question, 
            max_new_tokens=3000,
            temperature=0.1
        )
        
        if not deepseek_result.get("success", False):
            raise Exception(f"Local DeepSeek generation failed: {deepseek_result.get('error', 'Unknown error')}")
        
        # 解析结果为 beam search 格式
        beam_tree, paths = self.parser.parse_local_deepseek_result(deepseek_result)
        
        # 分类问题类型
        problem_type = self.classifier.classify_problem(question)
        
        # 构建返回结果
        result = FrontendBeamResult(
            problem={
                "question": question,
                "options": self._extract_options(question),
                "problem_type": problem_type,
                "reasoning_source": "local_deepseek_r1"
            },
            beam_tree={k: asdict(v) for k, v in beam_tree.items()},
            paths=[asdict(p) for p in paths]
        )
        
        return result
    
    async def _generate_with_api(self, question: str) -> FrontendBeamResult:
        """使用 API 生成推理"""
        
        # 调用 DeepSeek API
        deepseek_result = await self.deepseek_client.generate_reasoning(
            question, 
            max_tokens=3000,
            temperature=0.1
        )
        
        if not deepseek_result.get("success", False):
            raise Exception(f"DeepSeek API error: {deepseek_result.get('error', 'Unknown error')}")
        
        # 转换为前端格式
        beam_tree = {}
        for node_id, node_data in deepseek_result["beam_tree"].items():
            beam_tree[node_id] = {
                "id": node_data["id"],
                "content": node_data["content"],
                "reasoning_type": node_data["reasoning_type"],
                "quality_score": float(node_data["quality_score"]),
                "probability": float(node_data["probability"]),
                "parent": node_data["parent"],
                "children": node_data["children"],
                "variables": node_data["variables"],
                "depth": int(node_data["depth"])
            }
        
        # 转换路径
        paths = []
        for path_data in deepseek_result["paths"]:
            paths.append({
                "id": path_data["id"],
                "nodes": path_data["nodes"],
                "quality": path_data["quality"],
                "score": float(path_data["score"]),
                "is_correct": bool(path_data["is_correct"]),
                "final_answer": path_data["final_answer"]
            })
        
        # 提取问题信息
        problem_type = self.classifier.classify_problem(question)
        
        result = FrontendBeamResult(
            problem={
                "question": question,
                "options": self._extract_options(question),
                "problem_type": problem_type,
                "reasoning_source": "deepseek_api"
            },
            beam_tree=beam_tree,
            paths=paths
        )
        
        return result
    
    def _extract_options(self, question: str) -> List[str]:
        """从问题中提取选项"""
        options = []
        
        patterns = [
            r'([A-E]\)\s*[^A-E\n]*?)(?=[A-E]\)|$)',
            r'([A-E][\.\)]\s*[^A-E\n]*?)(?=[A-E][\.\)]|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.DOTALL)
            if matches:
                for match in matches:
                    option = match.strip()
                    if option and len(option) > 2:
                        option = re.sub(r'\s+', ' ', option)
                        options.append(option)
                break
        
        if not options:
            lines = question.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^[A-E][\.\)]\s*.+', line):
                    options.append(line)
        
        if not options:
            options = ["A) Option A", "B) Option B", "C) Option C", "D) Option D"]
        
        return options[:5]
    
    def is_available(self) -> bool:
        """检查生成器是否可用"""
        return (
            (self.local_engine is not None) or 
            (self.deepseek_client is not None and self.deepseek_client.is_available())
        )
    
    def get_status(self) -> Dict[str, Any]:
        """获取生成器状态"""
        local_loaded = False
        if self.local_engine:
            local_loaded = self.local_engine.is_loaded
        
        api_available = False
        if self.deepseek_client:
            api_available = self.deepseek_client.is_available()
        
        status = {
            "local_engine_available": self.local_engine is not None,
            "local_engine_loaded": local_loaded,
            "api_client_available": api_available,
            "current_mode": "local" if local_loaded else "api" if api_available else "unavailable",
            "fallback_enabled": self.fallback_to_api
        }
        
        return status

# 向后兼容性包装器
def UnifiedBeamSearchGenerator(*args, **kwargs):
    """向后兼容性包装器"""
    return EnhancedLocalDeepSeekBeamGenerator(*args, **kwargs)

# 新的统一生成器，同时支持本地和 API
class EnhancedUnifiedBeamSearchGenerator(EnhancedLocalDeepSeekBeamGenerator):
    """统一的增强 Beam Search 生成器，支持本地 DeepSeek 和 API"""
    pass

async def main():
    """测试增强的本地 DeepSeek 生成器"""
    
    print("🧪 Testing Enhanced Local DeepSeek Beam Search Generator")
    print("=" * 60)
    
    # 测试问题
    test_questions = [
        "Solve: 2x + 5 = 13",
        "Find the area of a triangle with sides 3, 4, and 5.",
        "What is 15% of 200?",
        "Find the sum of the first 10 natural numbers."
    ]
    
    try:
        # 初始化生成器
        generator = EnhancedLocalDeepSeekBeamGenerator(
            model_path="deepseek-ai/deepseek-math-7b-instruct",
            cache_dir="/app/models",
            gpu_id=1,
            use_local_engine=True,
            fallback_to_api=True
        )
        
        # 检查状态
        status = generator.get_status()
        print(f"📊 Generator Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        if not generator.is_available():
            print("❌ Generator not available")
            return
        
        # 测试每个问题
        for i, question in enumerate(test_questions):
            print(f"\n📝 Question {i+1}: {question}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                result = await generator.generate_reasoning_beam_search(
                    question, 
                    beam_width=3,
                    max_depth=5
                )
                generation_time = time.time() - start_time
                
                print(f"✅ Generated in {generation_time:.2f}s")
                print(f"   Problem type: {result.problem.get('problem_type', 'unknown')}")
                print(f"   Reasoning source: {result.problem.get('reasoning_source', 'unknown')}")
                print(f"   Tree nodes: {len(result.beam_tree)}")
                print(f"   Solution paths: {len(result.paths)}")
                
                # 显示最佳路径
                if result.paths:
                    best_path = result.paths[0]
                    print(f"   🏆 Best path: {best_path['quality']} quality (score: {best_path['score']:.2f})")
                    print(f"       Answer: {best_path['final_answer']}")
                    print(f"       Steps: {len(best_path['nodes'])} reasoning steps")
                
                # 保存结果
                filename = f"local_deepseek_result_{i+1}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    result_dict = asdict(result)
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
                print(f"   📁 Result saved: {filename}")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Failed to initialize generator: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("🎉 Local DeepSeek testing completed!")

if __name__ == "__main__":
    asyncio.run(main())