import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import openai

# 导入原有的类
from o1_reasoning_client import O1ReasoningClient, ThinkingStep, O1BeamNode
# 导入增强版解析器
from enhanced_reasoning_parser import EnhancedO1ReasoningParser

logger = logging.getLogger(__name__)

class EnhancedO1ReasoningClient(O1ReasoningClient):
    """增强版O1推理客户端 - 继承原有功能，添加增强特性"""
    
    def __init__(self, api_key: Optional[str] = None, use_enhanced_parsing: bool = True):
        # 调用父类初始化
        super().__init__(api_key)
        
        # 选择使用增强版还是原版解析器
        if use_enhanced_parsing:
            self.parser = EnhancedO1ReasoningParser()
            logger.info("✅ Using enhanced O1 reasoning parser")
        else:
            # 保持原有的解析器
            logger.info("ℹ️ Using original O1 reasoning parser")
        
        self.enhanced_mode = use_enhanced_parsing
    
    async def generate_reasoning(self, question: str, **kwargs) -> Dict[str, Any]:
        """增强版推理生成 - 保持原有接口，内部使用增强功能"""
        
        if not self.client:
            return {
                "success": False,
                "error": "O1 client not available - missing API key",
                "model": "O1"
            }
        
        try:
            logger.info(f"🧠 Generating {'enhanced ' if self.enhanced_mode else ''}O1 reasoning for question: {question[:50]}...")
            
            # 使用增强的prompt来获得更适合beam search的思考过程
            enhanced_prompt = self._create_enhanced_prompt(question) if self.enhanced_mode else question
            
            # 调用O1 API
            response = await self.client.chat.completions.create(
                model="o1-preview",
                messages=[
                    {
                        "role": "user", 
                        "content": enhanced_prompt
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
                thinking_content = full_response
                final_answer = "See reasoning above"
            
            # 使用增强版解析器解析thinking过程
            thinking_steps = self.parser.parse_thinking_content(thinking_content)
            logger.info(f"📊 Parsed {len(thinking_steps)} thinking steps (enhanced: {self.enhanced_mode})")
            
            # 构建增强版beam tree
            beam_tree = self.parser.build_beam_tree(thinking_steps, final_answer)
            
            # 提取增强版路径
            paths = self.parser.extract_paths(beam_tree)
            
            # 添加增强功能的统计信息
            enhancement_stats = self._calculate_enhancement_stats(beam_tree, paths)
            
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
                    "thinking_steps": len(thinking_steps),
                    "enhanced_parsing": self.enhanced_mode,
                    "enhancement_stats": enhancement_stats
                },
                "raw_response": full_response,
                "thinking_content": thinking_content
            }
            
            enhancement_msg = f" with {enhancement_stats['total_branches']} branches" if self.enhanced_mode else ""
            logger.info(f"🎉 O1 reasoning generated: {len(beam_tree)} nodes, {len(paths)} paths{enhancement_msg}")
            return result
            
        except Exception as e:
            logger.error(f"❌ O1 API error: {str(e)}")
            return {
                "success": False,
                "error": f"O1 API error: {str(e)}",
                "model": "O1"
            }
    
    def _create_enhanced_prompt(self, question: str) -> str:
        """创建增强版prompt，引导O1生成更适合beam search的思考过程"""
        
        enhanced_prompt = f"""Please solve this problem using a structured reasoning approach that explores multiple perspectives:

{question}

When working through this problem:
1. First, clearly understand what's being asked
2. Consider 2-3 different approaches or methods that could work
3. For your chosen approach, work step-by-step and show your reasoning
4. At key decision points, briefly mention alternative choices you could have made
5. Double-check your calculations and reasoning
6. If you notice any potential errors or uncertainties, address them explicitly
7. Provide a clear final answer

Please show your complete thinking process, including any moments where you reconsider your approach or verify your work."""

        return enhanced_prompt
    
    def _calculate_enhancement_stats(self, beam_tree: Dict[str, O1BeamNode], paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算增强功能的统计信息"""
        
        if not self.enhanced_mode:
            return {"enhanced_parsing": False}
        
        # 统计不同类型的节点
        node_types = {}
        branch_nodes = 0
        verification_nodes = 0
        
        for node_id, node in beam_tree.items():
            node_type = node.reasoning_type
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            if node_id.startswith('branch_'):
                branch_nodes += 1
            
            if 'verification' in node_type or 'check' in node_type:
                verification_nodes += 1
        
        # 统计路径类型
        path_types = {}
        for path in paths:
            path_type = path.get('path_type', 'unknown')
            path_types[path_type] = path_types.get(path_type, 0) + 1
        
        # 计算平均质量分数
        all_scores = [node.quality_score for node in beam_tree.values()]
        avg_quality = sum(all_scores) / len(all_scores) if all_scores else 0
        
        return {
            "enhanced_parsing": True,
            "total_nodes": len(beam_tree),
            "total_branches": branch_nodes,
            "verification_nodes": verification_nodes,
            "node_type_distribution": node_types,
            "path_type_distribution": path_types,
            "average_quality_score": round(avg_quality, 3),
            "reasoning_depth": max(node.depth for node in beam_tree.values()) if beam_tree else 0
        }

# 便捷的工厂函数
def create_enhanced_o1_client(api_key: Optional[str] = None, enhanced: bool = True) -> EnhancedO1ReasoningClient:
    """创建增强版O1客户端的便捷函数"""
    return EnhancedO1ReasoningClient(api_key=api_key, use_enhanced_parsing=enhanced)

# 向后兼容的类别名
EnhancedO1Client = EnhancedO1ReasoningClient

# 使用示例
async def test_enhanced_o1():
    """测试增强版O1推理"""
    
    # 创建增强版客户端
    enhanced_client = create_enhanced_o1_client(enhanced=True)
    
    # 创建原版客户端用于对比
    original_client = create_enhanced_o1_client(enhanced=False)
    
    if not enhanced_client.is_available():
        print("❌ O1 client not available - please set OPENAI_API_KEY")
        return
    
    test_question = """A triangle has sides of length 3, 4, and 5. 
    A) What is the area of this triangle?
    B) Is this a right triangle?
    C) What is the perimeter?"""
    
    print("🧪 Testing enhanced vs original O1 reasoning...")
    
    # 测试增强版
    print("\n🚀 Enhanced version:")
    enhanced_result = await enhanced_client.generate_reasoning(test_question)
    
    if enhanced_result["success"]:
        stats = enhanced_result["model_info"]["enhancement_stats"]
        print(f"✅ Enhanced: {stats['total_nodes']} nodes, {stats['total_branches']} branches")
        print(f"   Path types: {list(stats['path_type_distribution'].keys())}")
        print(f"   Avg quality: {stats['average_quality_score']}")
    
    # 测试原版
    print("\n📝 Original version:")
    original_result = await original_client.generate_reasoning(test_question)
    
    if original_result["success"]:
        print(f"✅ Original: {len(original_result['beam_tree'])} nodes, {len(original_result['paths'])} paths")
    
    # 保存对比结果
    comparison = {
        "enhanced": enhanced_result if enhanced_result["success"] else None,
        "original": original_result if original_result["success"] else None,
        "test_question": test_question
    }
    
    with open("enhanced_vs_original_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print("📁 Comparison saved to enhanced_vs_original_comparison.json")

if __name__ == "__main__":
    asyncio.run(test_enhanced_o1())