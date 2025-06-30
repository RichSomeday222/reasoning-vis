import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# 保持原有的数据结构不变
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

class EnhancedO1ReasoningParser:
    """增强版O1推理解析器 - 新增功能，不改原有接口"""
    
    def __init__(self):
        # 保持原有的推理关键词
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
        
        # 新增：更细粒度的推理类型
        self.enhanced_reasoning_types = {
            'problem_setup': ['understand', 'given', 'find', 'prove', 'show that', 'determine'],
            'method_selection': ['approach', 'method', 'strategy', 'could use', 'way to', 'technique'],
            'hypothesis': ['assume', 'suppose', 'let us say', 'if we', 'consider', 'what if'],
            'derivation': ['substitute', 'apply', 'use formula', 'follow from', 'according to'],
            'calculation': ['compute', 'calculate', 'equals', 'multiply', 'add', 'subtract'],
            'verification': ['check', 'verify', 'confirm', 'does this make sense', 'validate'],
            'error_detection': ['wait', 'mistake', 'wrong', 'error', 'that\'s not right'],
            'correction': ['actually', 'instead', 'should be', 'let me redo', 'correction'],
            'synthesis': ['combine', 'putting together', 'overall', 'in summary'],
            'conclusion': ['therefore', 'final answer', 'result is', 'so the answer']
        }
        
        # 新增：分支生成关键词
        self.branch_indicators = {
            'method_choice': ['could use', 'might try', 'another way', 'alternatively', 'or we could'],
            'hypothesis_test': ['assume', 'suppose', 'if we', 'let\'s say', 'consider'],
            'calculation_fork': ['calculate', 'compute', '=', 'substitute'],
            'verification_point': ['check', 'verify', 'make sure', 'confirm'],
            'uncertainty': ['not sure', 'maybe', 'perhaps', 'might be', 'could be']
        }

    def enhanced_split_thinking_text(self, text: str) -> List[str]:
        """增强版思维文本分割 - 更智能的分割策略"""
        
        # 第一阶段：粗分割
        rough_segments = self._rough_split(text)
        
        # 第二阶段：语义边界检测
        refined_segments = []
        for segment in rough_segments:
            refined_segments.extend(self._detect_semantic_boundaries(segment))
        
        # 第三阶段：清理和过滤
        final_segments = self._clean_and_filter_segments(refined_segments)
        
        return final_segments
    
    def _rough_split(self, text: str) -> List[str]:
        """粗分割 - 基于明显的分隔符"""
        patterns = [
            r'\n\n+',  # 双换行
            r'(?<=\.)\s*(?=[A-Z][a-z])',  # 句号后跟大写字母开头的单词
            r'(?<=\.)\s*(?=Let me|I need|Now|Next|So|The|Actually|Wait)',  # 特定开头词
            r'(?<=\.)\s*(?=\d+\.|\d+\))',  # 编号列表
            r'(?<=:)\s*(?=[A-Z])',  # 冒号后跟大写字母
        ]
        
        segments = [text]
        for pattern in patterns:
            new_segments = []
            for segment in segments:
                split_parts = re.split(pattern, segment)
                new_segments.extend([part.strip() for part in split_parts if part.strip()])
            segments = new_segments
        
        return segments
    
    def _detect_semantic_boundaries(self, segment: str) -> List[str]:
        """语义边界检测 - 识别完整的推理单元"""
        
        # 如果segment很短，直接返回
        if len(segment) < 50:
            return [segment]
        
        # 检测推理单元的开始和结束模式
        reasoning_unit_patterns = [
            # 假设-推导-结论模式
            (r'(assume|suppose|if we)', r'(therefore|thus|so we get)'),
            # 计算-验证模式  
            (r'(calculate|compute|substitute)', r'(equals|gives us|result)'),
            # 问题-方法-解答模式
            (r'(need to|want to|let me)', r'(done|finished|completed)')
        ]
        
        boundaries = []
        current_pos = 0
        
        for start_pattern, end_pattern in reasoning_unit_patterns:
            start_match = re.search(start_pattern, segment[current_pos:], re.IGNORECASE)
            if start_match:
                start_pos = current_pos + start_match.start()
                
                # 查找对应的结束
                remaining_text = segment[start_pos:]
                end_match = re.search(end_pattern, remaining_text, re.IGNORECASE)
                
                if end_match:
                    end_pos = start_pos + end_match.end()
                    
                    # 添加边界
                    if start_pos > current_pos:
                        boundaries.append(segment[current_pos:start_pos].strip())
                    boundaries.append(segment[start_pos:end_pos].strip())
                    current_pos = end_pos
        
        # 添加剩余部分
        if current_pos < len(segment):
            remaining = segment[current_pos:].strip()
            if remaining:
                boundaries.append(remaining)
        
        # 如果没有找到特殊模式，返回原segment
        return boundaries if boundaries else [segment]
    
    def _clean_and_filter_segments(self, segments: List[str]) -> List[str]:
        """清理和过滤分割结果"""
        cleaned = []
        
        for segment in segments:
            segment = segment.strip()
            
            # 过滤太短的片段
            if len(segment) < 15:
                continue
            
            # 过滤只有标点符号的片段
            if re.match(r'^[^\w]+$', segment):
                continue
            
            # 合并过短的相邻片段
            if cleaned and len(cleaned[-1]) < 30 and len(segment) < 50:
                cleaned[-1] = cleaned[-1] + " " + segment
            else:
                cleaned.append(segment)
        
        return cleaned

    def identify_branch_opportunities(self, steps: List[ThinkingStep]) -> List[Dict[str, Any]]:
        """识别分支机会点 - 更加选择性和克制"""
        opportunities = []
        
        # 限制总分支数量
        max_branches_per_tree = 3
        branch_count = 0
        
        for i, step in enumerate(steps):
            if branch_count >= max_branches_per_tree:
                break
                
            content_lower = step.content.lower()
            step_length = len(step.content)
            
            # 只在重要的推理步骤生成分支
            if step_length < 30:  # 跳过太短的步骤
                continue
            
            # 优先级1: 计算验证分支点 (最重要)
            if any(ind in content_lower for ind in self.branch_indicators['calculation_fork']):
                if ('=' in step.content and any(op in step.content for op in ['+', '-', '*', '/', '^'])) or 'calculate' in content_lower:
                    opportunities.append({
                        'position': i,
                        'type': 'calculation_verification',
                        'confidence': 0.9,
                        'indicator': 'mathematical_operation',
                        'description': 'Calculation step needs verification',
                        'priority': 1
                    })
                    branch_count += 1
                    continue
            
            # 优先级2: 方法选择分支点 (中等重要)
            method_indicators = ['approach', 'method', 'could use', 'way to']
            if any(indicator in content_lower for indicator in method_indicators):
                # 确保不是在描述已选方法，而是真的在选择
                if any(choice_word in content_lower for choice_word in ['could', 'might', 'another', 'alternative']):
                    opportunities.append({
                        'position': i,
                        'type': 'method_choice',
                        'confidence': 0.8,
                        'indicator': 'method_selection',
                        'description': 'Multiple solution approaches available',
                        'priority': 2
                    })
                    branch_count += 1
                    continue
            
            # 优先级3: 错误检测和纠正 (重要)
            error_indicators = ['wait', 'actually', 'mistake', 'wrong', 'let me reconsider']
            if any(indicator in content_lower for indicator in error_indicators):
                opportunities.append({
                    'position': i,
                    'type': 'error_correction',
                    'confidence': 0.85,
                    'indicator': 'error_detection',
                    'description': 'Error detected, exploring correction',
                    'priority': 1
                })
                branch_count += 1
                continue
        
        # 按优先级和置信度排序，只保留最重要的
        opportunities.sort(key=lambda x: (x['priority'], x['confidence']), reverse=True)
        return opportunities[:max_branches_per_tree]

    def generate_enhanced_branches(self, step: ThinkingStep, opportunity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成增强的分支节点 - 减少分支数量，提高质量"""
        branches = []
        branch_type = opportunity['type']
        base_confidence = step.confidence
        
        if branch_type == 'calculation_verification':
            # 只生成一个最重要的验证分支
            branches.append({
                'content': f"[Calculation Check] Let me double-check this arithmetic step...",
                'reasoning_type': 'calculation_verification',
                'quality_score': base_confidence * 0.95,
                'probability': base_confidence * 0.92,
                'variables': [f"calc_check_{opportunity['position']}"],
                'is_correction': False,
                'branch_reason': 'Verifying mathematical accuracy'
            })
        
        elif branch_type == 'method_choice':
            # 只生成一个替代方法分支
            branches.append({
                'content': f"[Alternative Method] What if we try a different approach here?",
                'reasoning_type': 'alternative_method',
                'quality_score': base_confidence * 0.8,
                'probability': base_confidence * 0.75,
                'variables': [f"alternative_approach_{opportunity['position']}"],
                'is_correction': False,
                'branch_reason': 'Exploring alternative solution method'
            })
        
        elif branch_type == 'error_correction':
            # 错误纠正分支
            branches.append({
                'content': f"[Error Correction] Let me reconsider this step more carefully...",
                'reasoning_type': 'error_correction',
                'quality_score': base_confidence * 0.9,
                'probability': base_confidence * 0.85,
                'variables': [f"error_fix_{opportunity['position']}"],
                'is_correction': True,
                'branch_reason': 'Correcting identified error'
            })
        
        return branches

    def enhanced_build_beam_tree(self, thinking_steps: List[ThinkingStep], final_answer: str) -> Dict[str, O1BeamNode]:
        """增强版beam树构建 - 生成更丰富的分支"""
        
        beam_tree = {}
        
        # 根节点保持不变
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
        
        # 识别分支机会
        branch_opportunities = self.identify_branch_opportunities(thinking_steps)
        
        # 构建主路径
        previous_node_id = "root"
        main_path_nodes = ["root"]
        
        for i, step in enumerate(thinking_steps):
            node_id = f"step_{i}"
            
            # 创建主路径节点
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
                is_correction=step.is_correction
            )
            
            beam_tree[node_id] = main_node
            
            # 更新父节点的children
            if previous_node_id in beam_tree:
                beam_tree[previous_node_id].children.append(node_id)
            
            # 检查是否有分支机会
            relevant_opportunities = [opp for opp in branch_opportunities if opp['position'] == i]
            
            for opp in relevant_opportunities:
                if opp['confidence'] > 0.6:  # 只生成高置信度的分支
                    branches = self.generate_enhanced_branches(step, opp)
                    
                    for j, branch_data in enumerate(branches):
                        branch_id = f"branch_{i}_{j}_{opp['type']}"
                        
                        branch_node = O1BeamNode(
                            id=branch_id,
                            content=branch_data['content'],
                            reasoning_type=branch_data['reasoning_type'],
                            quality_score=branch_data['quality_score'],
                            probability=branch_data['probability'],
                            parent=previous_node_id,  # 从主路径分叉
                            children=[],
                            variables=branch_data['variables'],
                            depth=len(main_path_nodes),
                            original_text=f"Generated branch: {branch_data['branch_reason']}",
                            is_correction=branch_data['is_correction']
                        )
                        
                        beam_tree[branch_id] = branch_node
                        
                        # 更新父节点的children
                        if previous_node_id in beam_tree:
                            beam_tree[previous_node_id].children.append(branch_id)
            
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
        
        if previous_node_id in beam_tree:
            beam_tree[previous_node_id].children.append(final_node_id)
        
        return beam_tree

    def enhanced_extract_paths(self, beam_tree: Dict[str, O1BeamNode]) -> List[Dict[str, Any]]:
        """增强版路径提取 - 提取少而精的路径"""
        paths = []
        
        # 主路径（最重要）
        main_path = self._extract_main_path(beam_tree)
        if main_path:
            paths.append(main_path)
        
        # 只选择最好的1-2条分支路径
        branch_paths = self._extract_best_branch_paths(beam_tree)
        paths.extend(branch_paths)
        
        return paths[:3]  # 最多3条路径，保持简洁
    
    def _extract_best_branch_paths(self, beam_tree: Dict[str, O1BeamNode]) -> List[Dict[str, Any]]:
        """只提取最佳的分支路径"""
        branch_paths = []
        
        # 查找所有分支节点
        branch_nodes = [node_id for node_id in beam_tree.keys() if node_id.startswith('branch_')]
        
        if not branch_nodes:
            return []
        
        # 选择质量最高的分支
        best_branch = max(branch_nodes, key=lambda node_id: beam_tree[node_id].quality_score)
        branch_node = beam_tree[best_branch]
        
        # 构建从根到分支的路径
        branch_path_nodes = ["root"]
        
        # 找到分支的父节点路径
        parent_chain = []
        current = branch_node.parent
        
        while current and current != "root":
            parent_chain.append(current)
            current = beam_tree[current].parent
        
        branch_path_nodes.extend(reversed(parent_chain))
        branch_path_nodes.append(best_branch)
        
        branch_scores = [beam_tree[node_id].quality_score for node_id in branch_path_nodes]
        branch_avg = sum(branch_scores) / len(branch_scores)
        
        branch_paths.append({
            "id": f"branch_path",
            "nodes": branch_path_nodes,
            "quality": "good" if branch_avg >= 0.7 else "fair",
            "score": branch_avg,
            "is_correct": branch_avg > 0.6,
            "final_answer": f"Alternative approach: {branch_node.reasoning_type}",
            "path_type": f"branch_exploration"
        })
        
        return branch_paths
    
    def _extract_main_path(self, beam_tree: Dict[str, O1BeamNode]) -> Dict[str, Any]:
        """提取主推理路径"""
        main_path_nodes = []
        current_node = "root"
        
        while current_node:
            main_path_nodes.append(current_node)
            node = beam_tree[current_node]
            
            if node.children:
                # 选择质量最高的主路径子节点（排除分支）
                main_children = [child_id for child_id in node.children 
                               if not child_id.startswith('branch_')]
                if main_children:
                    best_child = max(main_children, 
                                   key=lambda child_id: beam_tree[child_id].quality_score)
                    current_node = best_child
                else:
                    current_node = None
            else:
                current_node = None
        
        main_path_scores = [beam_tree[node_id].quality_score for node_id in main_path_nodes]
        main_path_avg = sum(main_path_scores) / len(main_path_scores)
        
        return {
            "id": "main_reasoning_path",
            "nodes": main_path_nodes,
            "quality": "excellent" if main_path_avg >= 0.8 else "good" if main_path_avg >= 0.6 else "fair",
            "score": main_path_avg,
            "is_correct": main_path_avg > 0.7,
            "final_answer": "Primary reasoning approach",
            "path_type": "main_reasoning"
        }
    
    def _extract_branch_paths(self, beam_tree: Dict[str, O1BeamNode]) -> List[Dict[str, Any]]:
        """提取分支路径"""
        branch_paths = []
        
        # 查找所有分支节点
        branch_nodes = [node_id for node_id in beam_tree.keys() if node_id.startswith('branch_')]
        
        # 按分支类型分组
        branch_types = {}
        for branch_id in branch_nodes:
            node = beam_tree[branch_id]
            branch_type = node.reasoning_type
            if branch_type not in branch_types:
                branch_types[branch_type] = []
            branch_types[branch_type].append(branch_id)
        
        # 为每种分支类型创建一条路径
        for branch_type, branch_node_ids in branch_types.items():
            if len(branch_node_ids) > 0:
                # 选择质量最高的分支
                best_branch = max(branch_node_ids, 
                                key=lambda node_id: beam_tree[node_id].quality_score)
                
                # 构建从根到分支的路径
                branch_path_nodes = ["root"]
                
                # 找到分支的父节点路径
                branch_node = beam_tree[best_branch]
                parent_chain = []
                current = branch_node.parent
                
                while current and current != "root":
                    parent_chain.append(current)
                    current = beam_tree[current].parent
                
                branch_path_nodes.extend(reversed(parent_chain))
                branch_path_nodes.append(best_branch)
                
                branch_scores = [beam_tree[node_id].quality_score for node_id in branch_path_nodes]
                branch_avg = sum(branch_scores) / len(branch_scores)
                
                branch_paths.append({
                    "id": f"branch_{branch_type}_path",
                    "nodes": branch_path_nodes,
                    "quality": "good" if branch_avg >= 0.7 else "fair",
                    "score": branch_avg,
                    "is_correct": branch_avg > 0.6,
                    "final_answer": f"Alternative: {branch_type}",
                    "path_type": f"branch_{branch_type}"
                })
        
        return branch_paths[:3]  # 最多3条分支路径
    
    def _extract_verification_paths(self, beam_tree: Dict[str, O1BeamNode]) -> List[Dict[str, Any]]:
        """提取验证路径"""
        verification_paths = []
        
        # 查找验证相关的节点
        verification_nodes = []
        for node_id, node in beam_tree.items():
            if 'verification' in node.reasoning_type or 'check' in node.reasoning_type:
                verification_nodes.append(node_id)
        
        if verification_nodes:
            # 创建验证路径
            best_verification = max(verification_nodes,
                                  key=lambda node_id: beam_tree[node_id].quality_score)
            
            # 构建验证路径
            verification_path_nodes = ["root"]
            
            # 找到验证节点的完整路径
            verification_node = beam_tree[best_verification]
            path_chain = []
            current = verification_node.parent
            
            while current and current != "root":
                path_chain.append(current)
                current = beam_tree[current].parent
            
            verification_path_nodes.extend(reversed(path_chain))
            verification_path_nodes.append(best_verification)
            
            verification_scores = [beam_tree[node_id].quality_score for node_id in verification_path_nodes]
            verification_avg = sum(verification_scores) / len(verification_scores)
            
            verification_paths.append({
                "id": "verification_path",
                "nodes": verification_path_nodes,
                "quality": "excellent" if verification_avg >= 0.8 else "good",
                "score": verification_avg,
                "is_correct": verification_avg > 0.75,
                "final_answer": "Verification and validation",
                "path_type": "verification"
            })
        
        return verification_paths

    # 保持原有接口兼容性
    def parse_thinking_content(self, thinking_text: str) -> List[ThinkingStep]:
        """保持原有接口 - 使用增强的分割方法"""
        # 使用增强的分割方法
        sentences = self.enhanced_split_thinking_text(thinking_text)
        
        steps = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 15:  # 调整最小长度
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
    
    def build_beam_tree(self, thinking_steps: List[ThinkingStep], final_answer: str) -> Dict[str, O1BeamNode]:
        """保持原有接口 - 使用增强的树构建"""
        return self.enhanced_build_beam_tree(thinking_steps, final_answer)
    
    def extract_paths(self, beam_tree: Dict[str, O1BeamNode]) -> List[Dict[str, Any]]:
        """保持原有接口 - 使用增强的路径提取"""
        return self.enhanced_extract_paths(beam_tree)
    
    # 保持原有的辅助方法
    def _classify_step_type(self, text: str) -> str:
        """分类步骤类型 - 先尝试增强类型，再用原有类型"""
        text_lower = text.lower()
        
        # 首先尝试增强的推理类型
        enhanced_scores = {}
        for step_type, keywords in self.enhanced_reasoning_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                enhanced_scores[step_type] = score
        
        if enhanced_scores:
            return max(enhanced_scores, key=enhanced_scores.get)
        
        # 回退到原有的分类方法
        scores = {}
        for step_type, keywords in self.reasoning_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[step_type] = score
        
        if not scores:
            return 'reasoning'
        
        return max(scores, key=scores.get)
    
    def _calculate_confidence(self, text: str, step_type: str) -> float:
        """计算置信度分数 - 更严格和真实的评分"""
        # 降低基础分数
        base_score = 0.3  # 从0.5降到0.3
        
        # 长度奖励 - 更保守
        length_bonus = min(0.2, len(text) / 300)  # 从200调到300，奖励更少
        
        # 数学内容奖励 - 减少奖励
        math_keywords = ['=', '+', '-', '*', '/', 'formula', 'equation', 'calculate']
        math_score = sum(0.03 for keyword in math_keywords if keyword in text.lower())  # 从0.05降到0.03
        
        # 确定性语言奖励/惩罚
        certainty_keywords = ['clearly', 'obviously', 'definitely', 'certainly']
        uncertainty_keywords = ['maybe', 'perhaps', 'might', 'possibly', 'i think', 'not sure', 'unclear']
        
        certainty_score = sum(0.08 for keyword in certainty_keywords if keyword in text.lower())  # 从0.1降到0.08
        uncertainty_penalty = sum(0.15 for keyword in uncertainty_keywords if keyword in text.lower())  # 从0.1增到0.15
        
        # 新增：错误指示词惩罚
        error_keywords = ['wrong', 'mistake', 'error', 'incorrect', 'failed']
        error_penalty = sum(0.2 for keyword in error_keywords if keyword in text.lower())
        
        # 新增：重复内容惩罚
        repeated_phrases = len(re.findall(r'\b(\w+)\b.*\b\1\b', text.lower()))
        repetition_penalty = min(0.1, repeated_phrases * 0.02)
        
        # 调整步骤类型奖励 - 更保守
        type_bonus = {
            'calculation': 0.15,        # 从0.2降到0.15
            'verification': 0.12,       # 从0.15降到0.12
            'conclusion': 0.18,         # 从0.25降到0.18
            'correction': -0.05,        # 从-0.1调到-0.05
            'problem_understanding': 0.08,  # 从0.1降到0.08
            'method_selection': 0.1,    # 从0.15降到0.1
            'hypothesis': 0.05,         # 从0.1降到0.05
            'derivation': 0.12,         # 从0.18降到0.12
            'error_correction': -0.1,   # 新增：错误纠正惩罚
            'uncertainty_resolution': 0.0  # 新增：不确定性解决无奖励
        }.get(step_type, 0)
        
        # 新增：内容质量评估
        quality_indicators = ['because', 'therefore', 'since', 'thus', 'hence']
        quality_bonus = sum(0.05 for indicator in quality_indicators if indicator in text.lower())
        
        # 新增：简短内容惩罚
        if len(text) < 20:
            short_penalty = 0.2
        elif len(text) < 40:
            short_penalty = 0.1
        else:
            short_penalty = 0.0
        
        # 计算最终分数
        final_score = (base_score + 
                      length_bonus + 
                      math_score + 
                      certainty_score + 
                      quality_bonus + 
                      type_bonus - 
                      uncertainty_penalty - 
                      error_penalty - 
                      repetition_penalty - 
                      short_penalty)
        
        # 更严格的边界 - 让分数分布更合理
        return max(0.05, min(0.95, final_score))  # 从[0.1, 1.0]调到[0.05, 0.95]
    
    def _is_correction_step(self, text: str) -> bool:
        """判断是否为纠正步骤 - 保持原有逻辑"""
        correction_indicators = [
            'wait', 'actually', 'no', 'wrong', 'mistake', 'error', 
            'let me reconsider', 'that\'s not right', 'correction'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in correction_indicators)