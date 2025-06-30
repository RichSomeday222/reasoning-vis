import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import openai

logger = logging.getLogger(__name__)

@dataclass
class ThinkingStep:
    """思考步骤"""
    content: str
    step_type: str  # reasoning, calculation, verification, correction
    confidence: float
    is_correction: bool = False
    parent_step: Optional[int] = None

@dataclass
class O1BeamNode:
    """O1推理节点"""
    id: str
    content: str
    reasoning_type: str
    quality_score: float
    probability: float
    parent: Optional[str]
    children: List[str]
    variables: List[str]
    depth: int
    original_text: str  # 保留原始thinking文本
    is_correction: bool = False

class O1ReasoningParser:
    """O1推理解析器"""
    
    def __init__(self):
        # 推理类型关键词
        self.reasoning_keywords = {
            'problem_understanding': [
                'let me understand', 'let me see', 'this is a', 'the problem is',
                'i need to', 'first', 'looking at', 'this asks', 'we have'
            ],
            'calculation': [
                'calculate', 'compute', 'sum', 'multiply', 'divide', 'equals',
                'formula', 'substitute', 'simplify', 'solve'
            ],
            'verification': [
                'let me check', 'verify', 'double check', 'confirm', 'make sure',
                'does this make sense', 'checking'
            ],
            'correction': [
                'wait', 'actually', 'no', 'that\'s wrong', 'let me reconsider',
                'i made an error', 'correction', 'instead', 'rather'
            ],
            'conclusion': [
                'therefore', 'so', 'final answer', 'the answer is', 'conclusion',
                'result', 'hence'
            ]
        }
    
    def parse_thinking_content(self, thinking_text: str) -> List[ThinkingStep]:
        """解析thinking内容为步骤"""
        
        # 按句号、换行等分割thinking内容
        sentences = self._split_thinking_text(thinking_text)
        
        steps = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:  # 跳过太短的片段
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
        
        # 先按明显的分隔符分割
        # 处理多种分隔模式
        patterns = [
            r'\n\n+',  # 双换行
            r'\.\s+(?=[A-Z])',  # 句号后跟大写字母
            r'(?<=\.)\s*(?=Let me|I need|Now|Next|So|The)',  # 特定开头词
            r'(?<=\.)\s*(?=\d+\.)',  # 编号列表
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
            if sentence and len(sentence) > 10:
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
        
        # 基于长度
        length_bonus = min(0.3, len(text) / 200)
        
        # 基于数学内容
        math_keywords = ['=', '+', '-', '*', '/', 'formula', 'equation', 'calculate']
        math_score = sum(0.05 for keyword in math_keywords if keyword in text.lower())
        
        # 基于确定性语言
        certainty_keywords = ['clearly', 'obviously', 'definitely', 'certainly']
        uncertainty_keywords = ['maybe', 'perhaps', 'might', 'possibly', 'i think']
        
        certainty_score = sum(0.1 for keyword in certainty_keywords if keyword in text.lower())
        uncertainty_penalty = sum(0.1 for keyword in uncertainty_keywords if keyword in text.lower())
        
        # 基于步骤类型
        type_bonus = {
            'calculation': 0.2,
            'verification': 0.15,
            'conclusion': 0.25,
            'correction': -0.1,
            'problem_understanding': 0.1
        }.get(step_type, 0)
        
        final_score = base_score + length_bonus + math_score + certainty_score - uncertainty_penalty + type_bonus
        
        return max(0.1, min(1.0, final_score))
    
    def _is_correction_step(self, text: str) -> bool:
        """判断是否为纠正步骤"""
        correction_indicators = [
            'wait', 'actually', 'no', 'wrong', 'mistake', 'error', 
            'let me reconsider', 'that\'s not right', 'correction'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in correction_indicators)
    
    def build_beam_tree(self, thinking_steps: List[ThinkingStep], final_answer: str) -> Dict[str, O1BeamNode]:
        """构建beam search树"""
        
        beam_tree = {}
        
        # 根节点
        beam_tree["root"] = O1BeamNode(
            id="root",
            content="[O1] Starting to analyze the problem",
            reasoning_type="start",
            quality_score=1.0,
            probability=1.0,
            parent=None,
            children=[],
            variables=["problem_analysis"],
            depth=0,
            original_text="",
            is_correction=False
        )
        
        # 处理思考步骤
        previous_node_id = "root"
        main_path_nodes = ["root"]
        correction_branches = []
        
        for i, step in enumerate(thinking_steps):
            node_id = f"step_{i}"
            
            # 判断是否为纠正分支
            if step.is_correction and i > 0:
                # 创建纠正分支
                correction_branch_id = f"correction_{i}"
                parent_id = main_path_nodes[-2] if len(main_path_nodes) > 1 else "root"
                
                correction_node = O1BeamNode(
                    id=correction_branch_id,
                    content=f"[O1] {step.content[:80]}..." if len(step.content) > 80 else f"[O1] {step.content}",
                    reasoning_type=step.step_type,
                    quality_score=max(0.3, step.confidence * 0.8),  # 纠正步骤质量略低
                    probability=step.confidence * 0.7,
                    parent=parent_id,
                    children=[],
                    variables=[f"correction_{i}"],
                    depth=len([n for n in main_path_nodes if not n.startswith("correction")]),
                    original_text=step.content,
                    is_correction=True
                )
                
                beam_tree[correction_branch_id] = correction_node
                
                # 更新父节点的children
                if parent_id in beam_tree:
                    beam_tree[parent_id].children.append(correction_branch_id)
                
                correction_branches.append(correction_branch_id)
                
            else:
                # 主路径节点
                main_node = O1BeamNode(
                    id=node_id,
                    content=f"[O1] {step.content[:80]}..." if len(step.content) > 80 else f"[O1] {step.content}",
                    reasoning_type=step.step_type,
                    quality_score=step.confidence,
                    probability=step.confidence * 0.9,
                    parent=previous_node_id,
                    children=[],
                    variables=[f"step_{i}_analysis"],
                    depth=len(main_path_nodes),
                    original_text=step.content,
                    is_correction=False
                )
                
                beam_tree[node_id] = main_node
                
                # 更新父节点的children
                if previous_node_id in beam_tree:
                    beam_tree[previous_node_id].children.append(node_id)
                
                main_path_nodes.append(node_id)
                previous_node_id = node_id
        
        # 添加最终答案节点
        final_node_id = "final_answer"
        final_node = O1BeamNode(
            id=final_node_id,
            content=f"[O1] Final Answer: {final_answer}",
            reasoning_type="conclusion",
            quality_score=0.95,
            probability=0.9,
            parent=previous_node_id,
            children=[],
            variables=["final_answer", "conclusion"],
            depth=len(main_path_nodes),
            original_text=final_answer,
            is_correction=False
        )
        
        beam_tree[final_node_id] = final_node
        
        # 更新倒数第二个节点的children
        if previous_node_id in beam_tree:
            beam_tree[previous_node_id].children.append(final_node_id)
        
        return beam_tree
    
    def extract_paths(self, beam_tree: Dict[str, O1BeamNode]) -> List[Dict[str, Any]]:
        """提取推理路径"""
        paths = []
        
        # 主路径（最高质量路径）
        main_path_nodes = []
        current_node = "root"
        
        while current_node:
            main_path_nodes.append(current_node)
            node = beam_tree[current_node]
            
            # 选择质量最高的子节点
            if node.children:
                best_child = max(node.children, 
                               key=lambda child_id: beam_tree[child_id].quality_score)
                current_node = best_child
            else:
                current_node = None
        
        # 计算主路径质量
        main_path_scores = [beam_tree[node_id].quality_score for node_id in main_path_nodes]
        main_path_avg = sum(main_path_scores) / len(main_path_scores)
        
        main_path = {
            "id": "main_path",
            "nodes": main_path_nodes,
            "quality": "excellent" if main_path_avg >= 0.8 else "good" if main_path_avg >= 0.6 else "fair",
            "score": main_path_avg,
            "is_correct": main_path_avg > 0.7,
            "final_answer": "Extracted from O1",
            "path_type": "main_reasoning"
        }
        
        paths.append(main_path)
        
        # 添加包含纠正的替代路径
        correction_nodes = [node_id for node_id, node in beam_tree.items() if node.is_correction]
        if correction_nodes:
            # 构建包含纠正的路径
            alt_path_nodes = ["root"]
            for node_id in correction_nodes[:3]:  # 最多包含3个纠正
                alt_path_nodes.append(node_id)
            
            if alt_path_nodes:
                alt_path_scores = [beam_tree[node_id].quality_score for node_id in alt_path_nodes]
                alt_path_avg = sum(alt_path_scores) / len(alt_path_scores)
                
                alt_path = {
                    "id": "correction_path",
                    "nodes": alt_path_nodes,
                    "quality": "fair",
                    "score": alt_path_avg,
                    "is_correct": False,
                    "final_answer": "Alternative approach",
                    "path_type": "correction_branch"
                }
                
                paths.append(alt_path)
        
        return paths

class O1ReasoningClient:
    """O1推理客户端"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key found")
            self.client = None
        else:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            logger.info("✅ O1 client initialized")
        
        self.parser = O1ReasoningParser()
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.client is not None
    
    async def generate_reasoning(self, question: str, **kwargs) -> Dict[str, Any]:
        """使用O1生成推理"""
        
        if not self.client:
            return {
                "success": False,
                "error": "O1 client not available - missing API key",
                "model": "O1"
            }
        
        try:
            logger.info(f"🧠 Generating O1 reasoning for question: {question[:50]}...")
            
            # 调用O1 API
            response = await self.client.chat.completions.create(
                model="o1-preview",  # 或 "o1-mini"
                messages=[
                    {
                        "role": "user",
                        "content": f"Please solve this problem step by step, showing your detailed reasoning:\n\n{question}"
                    }
                ],
                max_completion_tokens=kwargs.get("max_tokens", 3000)
            )
            
            # 获取完整回答
            full_response = response.choices[0].message.content
            logger.info(f"✅ O1 response received ({len(full_response)} chars)")
            
            # 提取thinking和最终答案
            thinking_content, final_answer = self._extract_thinking_and_answer(full_response)
            
            if not thinking_content:
                logger.warning("No thinking content found in O1 response")
                thinking_content = full_response  # 使用整个回答作为thinking
                final_answer = "See reasoning above"
            
            # 解析thinking过程
            thinking_steps = self.parser.parse_thinking_content(thinking_content)
            logger.info(f"📊 Parsed {len(thinking_steps)} thinking steps")
            
            # 构建beam tree
            beam_tree = self.parser.build_beam_tree(thinking_steps, final_answer)
            
            # 提取路径
            paths = self.parser.extract_paths(beam_tree)
            
            # 构建返回结果
            result = {
                "success": True,
                "model": "O1",
                "problem": {
                    "question": question,
                    "model": "O1",
                    "options": self._extract_options_from_question(question)
                },
                "beam_tree": {k: asdict(v) for k, v in beam_tree.items()},
                "paths": paths,
                "model_info": {
                    "name": "OpenAI O1",
                    "type": "real_api",
                    "model_id": "o1-preview",
                    "thinking_steps": len(thinking_steps)
                },
                "raw_response": full_response,
                "thinking_content": thinking_content
            }
            
            logger.info(f"🎉 O1 reasoning generated: {len(beam_tree)} nodes, {len(paths)} paths")
            return result
            
        except Exception as e:
            logger.error(f"❌ O1 API error: {str(e)}")
            return {
                "success": False,
                "error": f"O1 API error: {str(e)}",
                "model": "O1"
            }
    
    def _extract_thinking_and_answer(self, response: str) -> Tuple[str, str]:
        """提取thinking内容和最终答案"""
        
        # O1的thinking通常在<thinking>标签内，但有时格式不同
        thinking_patterns = [
            r'<thinking>(.*?)</thinking>',
            r'<think>(.*?)</think>',
            r'Let me think through this step by step\.(.*?)(?=\n\n|\nSo |Therefore |The answer is)',
            r'I need to.*?(?=\n\n|\nSo |\nTherefore |\nThe answer is)'
        ]
        
        thinking_content = ""
        for pattern in thinking_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                thinking_content = match.group(1).strip()
                break
        
        # 如果没有找到thinking标签，尝试其他方法
        if not thinking_content:
            # 查找明显的推理部分
            lines = response.split('\n')
            reasoning_lines = []
            
            for line in lines:
                if any(keyword in line.lower() for keyword in [
                    'let me', 'i need to', 'first', 'then', 'so', 'therefore',
                    'step', 'calculate', 'solve'
                ]):
                    reasoning_lines.append(line)
            
            if reasoning_lines:
                thinking_content = '\n'.join(reasoning_lines)
        
        # 提取最终答案
        answer_patterns = [
            r'(?:The answer is|Final answer|Answer:|Therefore,?)\s*([A-E]\).*?)(?:\n|$)',
            r'([A-E]\).*?)(?:\n|$)',
            r'(?:Therefore|So|Hence),?\s*(.*?)(?:\n|$)'
        ]
        
        final_answer = "Answer not clearly identified"
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                final_answer = match.group(1).strip()
                break
        
        return thinking_content, final_answer
    
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

# 使用示例和测试函数
async def test_o1_reasoning():
    """测试O1推理"""
    
    # 需要设置 OPENAI_API_KEY 环境变量
    client = O1ReasoningClient()
    
    if not client.is_available():
        print("❌ O1 client not available - please set OPENAI_API_KEY")
        return
    
    test_question = """Let S be the sum of the first nine terms of the sequence x+a, x²+2a, x³+3a, ... Then S equals:
A) (50a+x+x⁸)/(x+1)
B) 50a-(x+x¹⁰)/(x-1)
C) (x⁹-1)/(x+1)+45a
D) (x¹⁰-x)/(x-1)+45a"""
    
    print("🧪 Testing O1 reasoning...")
    result = await client.generate_reasoning(test_question)
    
    if result["success"]:
        print("✅ O1 reasoning successful!")
        print(f"Generated {len(result['beam_tree'])} nodes")
        print(f"Found {len(result['paths'])} reasoning paths")
        print(f"Thinking steps: {result['model_info']['thinking_steps']}")
        
        # 保存结果用于查看
        with open("o1_reasoning_test.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("📁 Results saved to o1_reasoning_test.json")
        
    else:
        print(f"❌ O1 reasoning failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(test_o1_reasoning())