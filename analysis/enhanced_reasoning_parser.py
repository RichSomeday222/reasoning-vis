import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# 保持原有数据结构完全不变
@dataclass
class ThinkingStep:
    """思考步骤 - 保持原有结构"""
    content: str
    step_type: str  
    confidence: float
    is_correction: bool = False
    parent_step: Optional[int] = None

@dataclass
class O1BeamNode:
    """O1推理节点 - 保持原有结构"""
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

class EnhancedBeamTreeBuilder:
    """专门构建增强beam tree的类 - 不改变接口，只优化内部逻辑"""
    
    def __init__(self):
        # 识别关键推理转折点的模式
        self.decision_patterns = [
            # 方法选择点
            r'(?i)(could use|might try|approach|method|way to solve)',
            # 计算分叉点  
            r'(?i)(calculate|compute|substitute|equals|=|\+|\-|\*|\/)',
            # 验证检查点
            r'(?i)(check|verify|make sure|confirm|does this make sense)',
            # 错误纠正点
            r'(?i)(wait|actually|mistake|wrong|let me reconsider)',
            # 假设探索点
            r'(?i)(assume|suppose|if we|let\'s say|consider that)',
        ]
        
        # 推理质量指标
        self.quality_indicators = {
            'mathematical': ['formula', 'equation', '=', 'calculate', 'solve'],
            'logical': ['because', 'therefore', 'since', 'thus', 'hence'],
            'verification': ['check', 'verify', 'confirm', 'validate'],
            'uncertainty': ['maybe', 'perhaps', 'might', 'possibly', 'not sure'],
            'confidence': ['clearly', 'obviously', 'definitely', 'certainly']
        }

    def identify_key_decision_points(self, steps: List[ThinkingStep]) -> List[Dict[str, Any]]:
        """识别关键决策点 - 更保守和精准"""
        decision_points = []
        
        for i, step in enumerate(steps):
            content_lower = step.content.lower()
            
            # 只选择真正重要的决策点
            decision_score = 0
            decision_type = None
            
            # 检查每种决策模式
            for j, pattern in enumerate(self.decision_patterns):
                if re.search(pattern, step.content, re.IGNORECASE):
                    pattern_names = ['method_choice', 'calculation', 'verification', 'correction', 'hypothesis']
                    decision_type = pattern_names[j]
                    decision_score += 1
                    break
            
            # 只有高质量、长内容的步骤才考虑分支
            if (decision_score > 0 and 
                len(step.content) > 40 and 
                step.confidence > 0.6 and
                i > 0):  # 不在开头分支
                
                decision_points.append({
                    'position': i,
                    'type': decision_type,
                    'confidence': step.confidence,
                    'content_length': len(step.content),
                    'priority': self._calculate_decision_priority(step, decision_type)
                })
        
        # 排序并只保留最重要的2-3个决策点
        decision_points.sort(key=lambda x: x['priority'], reverse=True)
        return decision_points[:3]

    def _calculate_decision_priority(self, step: ThinkingStep, decision_type: str) -> float:
        """计算决策点的优先级"""
        base_priority = {
            'calculation': 0.9,     # 计算最重要
            'verification': 0.8,    # 验证其次
            'correction': 0.85,     # 纠错也很重要
            'method_choice': 0.7,   # 方法选择
            'hypothesis': 0.6       # 假设探索
        }.get(decision_type, 0.5)
        
        # 考虑步骤质量
        quality_bonus = (step.confidence - 0.5) * 0.3
        
        # 考虑内容丰富度
        length_bonus = min(0.2, len(step.content) / 500)
        
        return base_priority + quality_bonus + length_bonus

    def generate_what_if_branches(self, steps: List[ThinkingStep], decision_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成 'what-if' 替代分支 - 高质量少量分支"""
        what_if_branches = []
        
        for decision in decision_points[:2]:  # 最多2个分支点
            step_idx = decision['position']
            decision_type = decision['type']
            original_step = steps[step_idx]
            
            # 根据决策类型生成对应的what-if分支
            if decision_type == 'calculation':
                what_if_branches.append({
                    'parent_step': step_idx,
                    'branch_type': 'calculation_alternative',
                    'content': '[Alternative Calculation] What if we use a different calculation method here?',
                    'reasoning_type': 'calculation_verification',
                    'quality_score': original_step.confidence * 0.9,
                    'probability': 0.8,
                    'is_hypothetical': True,
                    'branch_reason': 'Exploring alternative calculation approach'
                })
                
            elif decision_type == 'method_choice':
                what_if_branches.append({
                    'parent_step': step_idx,
                    'branch_type': 'method_alternative', 
                    'content': '[Alternative Method] What if we approach this problem differently?',
                    'reasoning_type': 'alternative_approach',
                    'quality_score': original_step.confidence * 0.85,
                    'probability': 0.75,
                    'is_hypothetical': True,
                    'branch_reason': 'Exploring different solution strategy'
                })
                
            elif decision_type == 'verification':
                what_if_branches.append({
                    'parent_step': step_idx,
                    'branch_type': 'verification_deep',
                    'content': '[Deep Verification] Let me double-check this step more thoroughly...',
                    'reasoning_type': 'deep_verification',
                    'quality_score': original_step.confidence * 0.95,
                    'probability': 0.9,
                    'is_hypothetical': True,
                    'branch_reason': 'Enhanced verification and validation'
                })
        
        return what_if_branches

    def build_enhanced_tree(self, thinking_steps: List[ThinkingStep], final_answer: str) -> Dict[str, O1BeamNode]:
        """构建增强的beam tree - 保持接口不变"""
        beam_tree = {}
        
        # 1. 创建根节点（保持不变）
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
        
        # 2. 识别关键决策点
        decision_points = self.identify_key_decision_points(thinking_steps)
        
        # 3. 生成what-if分支
        what_if_branches = self.generate_what_if_branches(thinking_steps, decision_points)
        
        # 4. 构建主推理路径
        previous_node_id = "root"
        main_path_nodes = ["root"]
        
        for i, step in enumerate(thinking_steps):
            node_id = f"step_{i}"
            
            # 创建主路径节点
            main_node = O1BeamNode(
                id=node_id,
                content=f"[O1] {step.content[:100]}..." if len(step.content) > 100 else f"[O1] {step.content}",
                reasoning_type=step.step_type,
                quality_score=step.confidence,
                probability=step.confidence * 0.9,
                parent=previous_node_id,
                children=[],
                variables=[f"step_{i}_var"],
                depth=len(main_path_nodes),
                original_text=step.content,
                is_correction=step.is_correction
            )
            
            beam_tree[node_id] = main_node
            
            # 更新父节点的children
            if previous_node_id in beam_tree:
                beam_tree[previous_node_id].children.append(node_id)
            
            # 5. 在关键点添加what-if分支
            relevant_branches = [b for b in what_if_branches if b['parent_step'] == i]
            
            for j, branch in enumerate(relevant_branches):
                branch_id = f"what_if_{i}_{j}_{branch['branch_type']}"
                
                branch_node = O1BeamNode(
                    id=branch_id,
                    content=branch['content'],
                    reasoning_type=branch['reasoning_type'],
                    quality_score=branch['quality_score'],
                    probability=branch['probability'],
                    parent=previous_node_id,  # 从主路径的前一个节点分叉
                    children=[],
                    variables=[f"what_if_{i}_{j}"],
                    depth=len(main_path_nodes),
                    original_text=f"Hypothetical branch: {branch['branch_reason']}",
                    is_correction=False
                )
                
                beam_tree[branch_id] = branch_node
                
                # 更新父节点的children
                if previous_node_id in beam_tree:
                    beam_tree[previous_node_id].children.append(branch_id)
            
            main_path_nodes.append(node_id)
            previous_node_id = node_id
        
        # 6. 添加最终答案节点（保持不变）
        final_node_id = "final_answer"
        final_node = O1BeamNode(
            id=final_node_id,
            content=f"[O1] Final Answer: {final_answer}",
            reasoning_type="conclusion",
            quality_score=0.95,
            probability=0.9,
            parent=previous_node_id,
            children=[],
            variables=["final_answer"],
            depth=len(main_path_nodes),
            original_text=final_answer,
            is_correction=False
        )
        
        beam_tree[final_node_id] = final_node
        
        if previous_node_id in beam_tree:
            beam_tree[previous_node_id].children.append(final_node_id)
        
        return beam_tree

class EnhancedO1ReasoningParser:
    """增强版O1推理解析器 - 使用新的BeamTreeBuilder"""
    
    def __init__(self):
        # 保持原有的推理关键词不变
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
        
        # 初始化增强的tree builder
        self.tree_builder = EnhancedBeamTreeBuilder()

    def parse_thinking_content(self, thinking_text: str) -> List[ThinkingStep]:
        """解析thinking内容 - 保持接口不变，优化内部逻辑"""
        
        # 改进的文本分割 - 更智能的分句
        sentences = self._smart_split_thinking(thinking_text)
        
        steps = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 20:  # 过滤太短的句子
                continue
            
            step_type = self._classify_step_type(sentence)
            confidence = self._calculate_enhanced_confidence(sentence, step_type)
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

    def _smart_split_thinking(self, text: str) -> List[str]:
        """智能分割thinking文本"""
        
        # 第一步：按明显分隔符粗分割
        rough_segments = re.split(r'\n\n+|(?<=\.)\s+(?=[A-Z])|(?<=\.)\s*(?=Let me|I need|Now|Next|So)', text)
        
        # 第二步：合并过短的片段
        refined_segments = []
        current_segment = ""
        
        for segment in rough_segments:
            segment = segment.strip()
            if len(segment) < 30 and current_segment:
                current_segment += " " + segment
            else:
                if current_segment:
                    refined_segments.append(current_segment)
                current_segment = segment
        
        if current_segment:
            refined_segments.append(current_segment)
        
        # 第三步：过滤和清理
        final_segments = []
        for segment in refined_segments:
            if len(segment) >= 20 and not re.match(r'^[^\w]*$', segment):
                final_segments.append(segment)
        
        return final_segments

    def _calculate_enhanced_confidence(self, text: str, step_type: str) -> float:
        """计算增强的置信度 - 更准确的评分"""
        
        # 基础分数
        base_score = 0.4
        
        # 长度奖励（更保守）
        length_bonus = min(0.15, len(text) / 400)
        
        # 数学内容奖励
        math_indicators = ['=', 'formula', 'calculate', 'equation', '+', '-', '*', '/']
        math_score = min(0.2, sum(0.03 for indicator in math_indicators if indicator in text.lower()))
        
        # 逻辑连接词奖励
        logic_indicators = ['because', 'therefore', 'since', 'thus', 'hence', 'so']
        logic_score = min(0.15, sum(0.03 for indicator in logic_indicators if indicator in text.lower()))
        
        # 确定性语言
        certainty_words = ['clearly', 'obviously', 'definitely']
        uncertainty_words = ['maybe', 'perhaps', 'might', 'possibly', 'not sure']
        
        certainty_bonus = min(0.1, sum(0.05 for word in certainty_words if word in text.lower()))
        uncertainty_penalty = min(0.2, sum(0.07 for word in uncertainty_words if word in text.lower()))
        
        # 步骤类型奖励
        type_bonus = {
            'calculation': 0.15,
            'verification': 0.12,
            'conclusion': 0.18,
            'problem_understanding': 0.1,
            'correction': -0.05
        }.get(step_type, 0.05)
        
        # 计算最终分数
        final_score = (base_score + length_bonus + math_score + logic_score + 
                      certainty_bonus + type_bonus - uncertainty_penalty)
        
        return max(0.1, min(0.95, final_score))

    def build_beam_tree(self, thinking_steps: List[ThinkingStep], final_answer: str) -> Dict[str, O1BeamNode]:
        """构建beam tree - 使用增强的tree builder"""
        return self.tree_builder.build_enhanced_tree(thinking_steps, final_answer)

    def extract_paths(self, beam_tree: Dict[str, O1BeamNode]) -> List[Dict[str, Any]]:
        """提取路径 - 保持接口不变，优化路径选择"""
        paths = []
        
        # 主路径
        main_path = self._extract_main_path(beam_tree)
        if main_path:
            paths.append(main_path)
        
        # what-if分支路径
        what_if_paths = self._extract_what_if_paths(beam_tree)
        paths.extend(what_if_paths)
        
        return paths[:4]  # 最多4条路径

    def _extract_main_path(self, beam_tree: Dict[str, O1BeamNode]) -> Dict[str, Any]:
        """提取主推理路径"""
        main_path_nodes = []
        current_node = "root"
        
        while current_node:
            main_path_nodes.append(current_node)
            node = beam_tree[current_node]
            
            if node.children:
                # 选择非what-if的最高质量子节点
                main_children = [child_id for child_id in node.children 
                               if not child_id.startswith('what_if_')]
                if main_children:
                    best_child = max(main_children, 
                                   key=lambda child_id: beam_tree[child_id].quality_score)
                    current_node = best_child
                else:
                    current_node = None
            else:
                current_node = None
        
        scores = [beam_tree[node_id].quality_score for node_id in main_path_nodes]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "id": "main_reasoning",
            "nodes": main_path_nodes,
            "quality": "excellent" if avg_score >= 0.8 else "good" if avg_score >= 0.6 else "fair",
            "score": avg_score,
            "is_correct": avg_score > 0.7,
            "final_answer": "Primary reasoning path",
            "path_type": "main_reasoning"
        }

    def _extract_what_if_paths(self, beam_tree: Dict[str, O1BeamNode]) -> List[Dict[str, Any]]:
        """提取what-if分支路径"""
        what_if_paths = []
        
        # 找到所有what-if节点
        what_if_nodes = [node_id for node_id in beam_tree.keys() if node_id.startswith('what_if_')]
        
        # 按reasoning_type分组
        what_if_groups = {}
        for node_id in what_if_nodes:
            node = beam_tree[node_id]
            reasoning_type = node.reasoning_type
            if reasoning_type not in what_if_groups:
                what_if_groups[reasoning_type] = []
            what_if_groups[reasoning_type].append(node_id)
        
        # 为每组创建一条路径
        for reasoning_type, node_ids in what_if_groups.items():
            best_node_id = max(node_ids, key=lambda nid: beam_tree[nid].quality_score)
            best_node = beam_tree[best_node_id]
            
            # 构建从root到what-if节点的路径
            path_nodes = ["root"]
            
            # 找到what-if节点的父节点链
            current = best_node.parent
            parent_chain = []
            while current and current != "root":
                parent_chain.append(current)
                current = beam_tree[current].parent
            
            path_nodes.extend(reversed(parent_chain))
            path_nodes.append(best_node_id)
            
            path_scores = [beam_tree[node_id].quality_score for node_id in path_nodes]
            avg_score = sum(path_scores) / len(path_scores)
            
            what_if_paths.append({
                "id": f"what_if_{reasoning_type}",
                "nodes": path_nodes,
                "quality": "good" if avg_score >= 0.7 else "fair",
                "score": avg_score,
                "is_correct": avg_score > 0.6,
                "final_answer": f"Alternative: {reasoning_type}",
                "path_type": f"what_if_{reasoning_type}"
            })
        
        return what_if_paths[:3]  # 最多3条what-if路径

    # 保持原有的辅助方法不变
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

    def _is_correction_step(self, text: str) -> bool:
        """判断是否为纠正步骤"""
        correction_indicators = [
            'wait', 'actually', 'no', 'wrong', 'mistake', 'error', 
            'let me reconsider', 'that\'s not right', 'correction'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in correction_indicators)