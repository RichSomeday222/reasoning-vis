import os
import re
import json
import asyncio
import logging
import aiohttp
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ThinkingStep:
    """思考步骤 - 与O1保持一致的数据结构"""
    content: str
    step_type: str  
    confidence: float
    is_correction: bool = False
    parent_step: Optional[int] = None

@dataclass
class DeepSeekBeamNode:
    """DeepSeek推理节点 - 与O1BeamNode保持一致的结构"""
    id: str
    content: str
    reasoning_type: str
    quality_score: float
    probability: float
    parent: Optional[str]
    children: List[str]
    variables: List[str]
    depth: int
    original_text: str
    is_correction: bool = False

class DeepSeekReasoningParser:
    """DeepSeek推理解析器"""
    
    def __init__(self):
        # 推理类型关键词
        self.reasoning_keywords = {
            'problem_understanding': [
                'let me understand', 'let me analyze', 'this problem asks', 'we need to',
                'first, i need to', 'looking at this', 'the question is', 'we have'
            ],
            'calculation': [
                'calculate', 'compute', 'sum', 'multiply', 'divide', 'equals',
                'formula', 'substitute', 'simplify', 'solve', 'apply'
            ],
            'verification': [
                'let me check', 'verify', 'double check', 'confirm', 'make sure',
                'does this make sense', 'checking', 'validation'
            ],
            'correction': [
                'wait', 'actually', 'no', 'that\'s wrong', 'let me reconsider',
                'i made an error', 'correction', 'instead', 'rather', 'mistake'
            ],
            'conclusion': [
                'therefore', 'so', 'final answer', 'the answer is', 'conclusion',
                'result', 'hence', 'in summary'
            ]
        }
    
    def parse_thinking_content(self, thinking_text: str) -> List[ThinkingStep]:
        """解析thinking内容为步骤"""
        sentences = self._split_thinking_text(thinking_text)
        
        steps = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 15:
                continue
            
            step_type = self._classify_step_type(sentence)
            confidence = self._calculate_confidence(sentence, step_type)
            is_correction = self._is_correction_step(sentence)
            
            step = ThinkingStep(
                content=sentence.strip(),
                step_type=step_type,
                confidence=confidence,
                is_correction=is_correction,
                parent_step=i-1 if i > 0 else None
            )
            steps.append(step)
        
        return steps
    
    def _split_thinking_text(self, text: str) -> List[str]:
        """智能分割thinking文本"""
        patterns = [
            r'\n\n+',
            r'\.\s+(?=[A-Z])',
            r'(?<=\.)\s*(?=Let me|I need|Now|Next|So|The|First|Second|Third)',
            r'(?<=\.)\s*(?=\d+\.)',
            r'(?<=:)\s*(?=[A-Z])',
        ]
        
        sentences = [text]
        for pattern in patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = new_sentences
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 15:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
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
    
    def _calculate_confidence(self, text: str, step_type: str) -> float:
        """计算置信度分数"""
        base_score = 0.5
        length_bonus = min(0.25, len(text) / 250)
        
        math_keywords = ['=', '+', '-', '*', '/', 'formula', 'equation', 'calculate', 'solve']
        math_score = sum(0.04 for keyword in math_keywords if keyword in text.lower())
        
        certainty_keywords = ['clearly', 'obviously', 'definitely', 'certainly', 'indeed']
        uncertainty_keywords = ['maybe', 'perhaps', 'might', 'possibly', 'i think', 'seems']
        
        certainty_score = sum(0.08 for keyword in certainty_keywords if keyword in text.lower())
        uncertainty_penalty = sum(0.08 for keyword in uncertainty_keywords if keyword in text.lower())
        
        type_bonus = {
            'calculation': 0.2,
            'verification': 0.15,
            'conclusion': 0.25,
            'correction': -0.1,
            'problem_understanding': 0.12
        }.get(step_type, 0.05)
        
        final_score = (base_score + length_bonus + math_score + certainty_score + 
                      type_bonus - uncertainty_penalty)
        
        return max(0.15, min(0.98, final_score))
    
    def _is_correction_step(self, text: str) -> bool:
        """判断是否为纠正步骤"""
        correction_indicators = [
            'wait', 'actually', 'no', 'wrong', 'mistake', 'error', 
            'let me reconsider', 'that\'s not right', 'correction',
            'i need to fix', 'let me redo'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in correction_indicators)
    
    def build_beam_tree(self, thinking_steps: List[ThinkingStep], final_answer: str) -> Dict[str, DeepSeekBeamNode]:
        """构建beam search树"""
        beam_tree = {}
        
        # 根节点
        beam_tree["root"] = DeepSeekBeamNode(
            id="root",
            content="[Real DeepSeek API] Starting problem analysis",
            reasoning_type="start",
            quality_score=1.0,
            probability=1.0,
            parent=None,
            children=[],
            variables=["real_api_analysis"],
            depth=0,
            original_text="",
            is_correction=False
        )
        
        # 处理思考步骤
        previous_node_id = "root"
        main_path_nodes = ["root"]
        
        for i, step in enumerate(thinking_steps):
            node_id = f"step_{i}"
            
            main_node = DeepSeekBeamNode(
                id=node_id,
                content=f"[Real API] {step.content[:80]}..." if len(step.content) > 80 else f"[Real API] {step.content}",
                reasoning_type=step.step_type,
                quality_score=step.confidence,
                probability=step.confidence * 0.9,
                parent=previous_node_id,
                children=[],
                variables=[f"real_step_{i}_analysis"],
                depth=len(main_path_nodes),
                original_text=step.content,
                is_correction=step.is_correction
            )
            
            beam_tree[node_id] = main_node
            
            # 更新父节点的children
            if previous_node_id in beam_tree:
                beam_tree[previous_node_id].children.append(node_id)
            
            main_path_nodes.append(node_id)
            previous_node_id = node_id
        
        # 添加最终答案节点
        final_node_id = "final_answer"
        final_node = DeepSeekBeamNode(
            id=final_node_id,
            content=f"[Real API] Final Answer: {final_answer}",
            reasoning_type="conclusion",
            quality_score=0.95,
            probability=0.92,
            parent=previous_node_id,
            children=[],
            variables=["real_final_answer", "real_conclusion"],
            depth=len(main_path_nodes),
            original_text=final_answer,
            is_correction=False
        )
        
        beam_tree[final_node_id] = final_node
        
        if previous_node_id in beam_tree:
            beam_tree[previous_node_id].children.append(final_node_id)
        
        return beam_tree
    
    def extract_paths(self, beam_tree: Dict[str, DeepSeekBeamNode]) -> List[Dict[str, Any]]:
        """提取推理路径"""
        paths = []
        
        # 主路径
        main_path_nodes = []
        current_node = "root"
        
        while current_node:
            main_path_nodes.append(current_node)
            node = beam_tree[current_node]
            
            if node.children:
                best_child = max(node.children, 
                               key=lambda child_id: beam_tree[child_id].quality_score)
                current_node = best_child
            else:
                current_node = None
        
        main_path_scores = [beam_tree[node_id].quality_score for node_id in main_path_nodes]
        main_path_avg = sum(main_path_scores) / len(main_path_scores)
        
        main_path = {
            "id": "real_api_main_path",
            "nodes": main_path_nodes,
            "quality": "excellent" if main_path_avg >= 0.85 else "good" if main_path_avg >= 0.65 else "fair",
            "score": main_path_avg,
            "is_correct": main_path_avg > 0.75,
            "final_answer": "Real DeepSeek API Response",
            "path_type": "real_api_reasoning"
        }
        
        paths.append(main_path)
        return paths

class DeepSeekReasoningClient:
    """真实的DeepSeek API客户端"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url or "https://api.deepseek.com"
        
        if not self.api_key:
            logger.warning("⚠️ No DeepSeek API key found in DEEPSEEK_API_KEY environment variable")
            self.api_key = None
        else:
            logger.info("✅ DeepSeek API key found")
        
        self.parser = DeepSeekReasoningParser()
        
        if self.api_key:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "DeepSeek-Reasoning-Client/1.0"
            }
        else:
            self.headers = {}
    
    def is_available(self) -> bool:
        """检查API是否可用"""
        return bool(self.api_key)
    
    async def generate_reasoning(self, question: str, **kwargs) -> Dict[str, Any]:
        """使用真实DeepSeek API生成推理"""
        
        if not self.is_available():
            return {
                "success": False,
                "error": "DeepSeek API key not available. Set DEEPSEEK_API_KEY environment variable.",
                "model": "DeepSeek"
            }
        
        try:
            logger.info(f"🤖 Calling REAL DeepSeek API for: {question[:50]}...")
            
            # 构建请求数据
            request_data = {
                "model": "deepseek-reasoner",
                "messages": [
                    {
                        "role": "user", 
                        "content": f"Please solve this mathematical problem step by step with detailed reasoning:\n\n{question}"
                    }
                ],
                "max_tokens": kwargs.get("max_tokens", 4000),
                "temperature": kwargs.get("temperature", 0.1),
                "stream": False
            }
            
            logger.info(f"📤 Sending request to {self.base_url}/chat/completions")
            
            # 发送API请求
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=90)
                ) as response:
                    
                    response_text = await response.text()
                    logger.info(f"📥 API response status: {response.status}")
                    
                    if response.status != 200:
                        logger.error(f"❌ DeepSeek API error {response.status}: {response_text}")
                        return {
                            "success": False,
                            "error": f"DeepSeek API error {response.status}: {response_text}",
                            "model": "DeepSeek"
                        }
                    
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse JSON response: {e}")
                        return {
                            "success": False,
                            "error": f"Invalid JSON response: {e}",
                            "model": "DeepSeek"
                        }
            
            # 解析响应
            if "choices" not in response_data or not response_data["choices"]:
                logger.error(f"❌ Invalid API response format: {response_data}")
                return {
                    "success": False,
                    "error": "Invalid API response format - no choices",
                    "model": "DeepSeek"
                }
            
            choice = response_data["choices"][0]
            
            # 提取推理内容
            if "reasoning_content" in choice:
                # DeepSeek Reasoner 模型的格式
                thinking_content = choice["reasoning_content"]
                final_answer = choice["message"]["content"]
                logger.info(f"📊 Found reasoning_content: {len(thinking_content)} chars")
            else:
                # 普通聊天模型的格式
                full_content = choice["message"]["content"]
                thinking_content, final_answer = self._split_content(full_content)
                logger.info(f"📊 Split content: {len(thinking_content)} chars thinking, {len(final_answer)} chars answer")
            
            # 解析thinking过程
            thinking_steps = self.parser.parse_thinking_content(thinking_content)
            logger.info(f"🧠 Parsed {len(thinking_steps)} thinking steps")
            
            # 构建beam tree
            beam_tree = self.parser.build_beam_tree(thinking_steps, final_answer)
            
            # 提取路径
            paths = self.parser.extract_paths(beam_tree)
            
            # 构建返回结果
            result = {
                "success": True,
                "model": "DeepSeek",
                "problem": {
                    "question": question,
                    "model": "DeepSeek-Reasoner",
                    "options": self._extract_options_from_question(question)
                },
                "beam_tree": {k: asdict(v) for k, v in beam_tree.items()},
                "paths": paths,
                "model_info": {
                    "name": "DeepSeek Reasoner (Real API)",
                    "type": "real_api",
                    "model_id": request_data["model"],
                    "thinking_steps": len(thinking_steps),
                    "tokens_used": response_data.get("usage", {}).get("total_tokens", 0),
                    "api_endpoint": self.base_url
                },
                "raw_response": response_data,
                "thinking_content": thinking_content,
                "reasoning_source": "real_deepseek_api"
            }
            
            logger.info(f"🎉 Real DeepSeek API completed successfully: {len(beam_tree)} nodes, {len(paths)} paths")
            return result
            
        except asyncio.TimeoutError:
            logger.error("❌ DeepSeek API timeout")
            return {
                "success": False,
                "error": "DeepSeek API timeout (>90 seconds)",
                "model": "DeepSeek"
            }
        except Exception as e:
            logger.error(f"❌ Real DeepSeek API error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Real DeepSeek API error: {str(e)}",
                "model": "DeepSeek"
            }
    
    def _split_content(self, content: str) -> Tuple[str, str]:
        """分离思考过程和最终答案"""
        # 尝试找到思考和答案的分界
        patterns = [
            r"(.*?)(?:Final Answer|Answer|Conclusion):\s*(.*)",
            r"(.*?)(?:Therefore|So|Hence),?\s*(.*)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                thinking = match.group(1).strip()
                answer = match.group(2).strip()
                return thinking, answer
        
        # 如果没有明确分界，使用整个内容作为思考过程
        lines = content.split('\n')
        if len(lines) > 1:
            thinking = '\n'.join(lines[:-1])
            answer = lines[-1]
        else:
            thinking = content
            answer = "Answer extracted from reasoning"
        
        return thinking, answer
    
    def _extract_options_from_question(self, question: str) -> List[str]:
        """从问题中提取选项"""
        option_pattern = r'([A-E]\)[^A-E]*?)(?=[A-E]\)|$)'
        matches = re.findall(option_pattern, question, re.DOTALL)
        
        options = []
        for match in matches:
            option = match.strip()
            if option:
                options.append(option)
        
        return options if options else ["A) Option A", "B) Option B", "C) Option C", "D) Option D"]