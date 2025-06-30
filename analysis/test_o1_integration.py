import asyncio
import logging
import json
from typing import List, Dict, Any, Optional

from o1_reasoning_client import O1ReasoningClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BeamSearchResult:
    """Container for beam search output."""
    def __init__(self,
                 question: str,
                 beam_trees: List[Dict[str, Any]],
                 beam_paths: List[List[Dict[str, Any]]]):
        self.question = question
        self.beam_trees = beam_trees        # list of beam_tree dicts
        self.beam_paths = beam_paths        # list of paths lists

class O1UnifiedBeamSearchGenerator:
    """Unified beam search generator using O1ReasoningClient."""

    def __init__(self, api_key: Optional[str] = None):
        # 初始化 O1 客户端和解析器
        self.o1_client = O1ReasoningClient(api_key)
        self.parser = self.o1_client.parser
        if not self.o1_client.is_available():
            logger.warning("O1ReasoningClient not available; beam search will fail")

    async def _generate_multiple(self, question: str, beam_width: int) -> List[Dict[str, Any]]:
        """
        并行调用 O1 API beam_width 次，获取多条 chain-of-thought。
        Returns:
            List of result dicts from generate_reasoning
        """
        tasks = [
            self.o1_client.generate_reasoning(question)
            for _ in range(beam_width)
        ]
        # gather 并发结果
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # 过滤失败
        valid = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Error in generate_reasoning task: {res}")
            elif res.get("success"):
                valid.append(res)
        return valid

    async def generate_reasoning_beam_search(self,
                                       question: str,
                                       beam_width: int = 3,
                                       max_depth: int = 5) -> BeamSearchResult:
        """
        对问题进行 beam search 推理：
        1. 并行调用模型 beam_width 次
        2. 对每条思路构建 beam 树，提取 paths
        3. 按 paths[0]["score"] 排序，取 top beam_width

        Returns:
            BeamSearchResult 包含 question、beam_trees 和 beam_paths
        """
        if not self.o1_client.is_available():
            raise RuntimeError("O1ReasoningClient not available for beam search")

        # 并发调用模型
        results = await self._generate_multiple(question, beam_width)

        # 构建每条候选的树和路径
        candidates = []  # type: List[Dict[str, Any]]
        for res in results:
            thinking = res.get("thinking_content", "")
            final_ans = res.get("raw_response", "")

            steps = self.parser.parse_thinking_content(thinking)
            tree = self.parser.build_beam_tree(steps, final_ans)
            paths = self.parser.extract_paths(tree)

            # 主路径评分
            score = paths[0]["score"] if paths else 0.0
            candidates.append({"tree": tree, "paths": paths, "score": score})

        # 排序取 top K
        candidates.sort(key=lambda c: c["score"], reverse=True)
        top_k = candidates[:beam_width]

        beam_trees = [c["tree"] for c in top_k]
        beam_paths = [c["paths"] for c in top_k]

        return BeamSearchResult(question, beam_trees, beam_paths)

# 测试函数
async def test_beam_search():
    """测试 UnifiedBeamSearchGenerator 的 beam search 功能"""
    question = (
        "Let S be the sum of the first nine terms of the sequence x+a, x²+2a, x³+3a, ... "
        "Then S equals:\n"
        "A) (50a+x+x⁸)/(x+1)\n"
        "B) 50a-(x+x¹⁰)/(x-1)\n"
        "C) (x⁹-1)/(x+1)+45a\n"
        "D) (x¹⁰-x)/(x-1)+45a"
    )
    generator = O1UnifiedBeamSearchGenerator()
    result = await generator.generate_reasoning_beam_search(question, beam_width=3)

    # 打印测试结果
    print("Question:", result.question)
    for idx, paths in enumerate(result.beam_paths, 1):
        print(f"\nBeam {idx} Paths:")
        for p in paths:
            print(f"  Path ID: {p['id']}, Score: {p['score']}")
            print(f"    Nodes: {p['nodes']}")

if __name__ == "__main__":
    asyncio.run(test_beam_search())
