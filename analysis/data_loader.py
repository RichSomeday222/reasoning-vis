from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    
    def load_math_sample(self, file_path: str = None, max_problems: int = 5) -> List[Dict[str, Any]]:
        """Load sample problems from math JSONL files"""
        if file_path is None:
            math_dir = Path("general_math_data")
            if math_dir.exists():
                jsonl_files = list(math_dir.glob("sample_*.jsonl"))
                if not jsonl_files:
                    jsonl_files = list(math_dir.glob("*.jsonl"))

                if jsonl_files:
                    file_path = jsonl_files[0]
                    logger.info(f"Auto-detected math file: {file_path}")
                else:
                    logger.warning("No math JSONL files found")
                    return []
            else:
                logger.warning("Math data directory not found")
                return []
        else:
            file_path = Path(file_path)

        problems = []

        try:
            if not file_path.exists():
                logger.warning(f"Math file not found: {file_path}")
                return problems

            with open(file_path, 'r', encoding='utf-8') as f:
                count = 0
                for line_num, line in enumerate(f, 1):
                    if count >= max_problems:
                        break

                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            problem = self._convert_math_to_standard(data, count)
                            if problem:
                                problems.append(problem)
                                count += 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON decode error on line {line_num}: {e}")
                            continue

            logger.info(f"📊 Loaded {len(problems)} math problems")
            return problems

        except Exception as e:
            logger.error(f"Error loading math sample: {e}")
            return problems

    def _convert_math_to_standard(self, data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Convert math data into standard format"""
        try:
            question = data.get("question", "")
            if not question:
                return None

            difficulty = "medium"
            if len(question) > 600:
                difficulty = "hard"
            elif len(question) < 200:
                difficulty = "easy"

            category = self._classify_math_category(question)

            problem = {
                "id": f"math_{data.get('id', index)}",
                "question": question,
                "category": category,
                "difficulty": difficulty,
                "dataset": "math",
                "metadata": {
                    "source_file": "general_math_data",
                    "original_id": data.get("id"),
                    "has_answer": "answer_content" in data,
                    "has_reasoning": "reasoning_content" in data,
                    "verifier_score": data.get("verifier_score"),
                    "community_score": data.get("community_score"),
                    "domain": "mathematics"
                }
            }

            if "answer_content" in data:
                problem["answer_content"] = data["answer_content"]

            if "reasoning_content" in data:
                problem["reasoning_content"] = data["reasoning_content"]

            return problem

        except Exception as e:
            logger.warning(f"Error converting math data: {e}")
            return None

    def _classify_math_category(self, question: str) -> str:
        """Classify the math question category"""
        question_lower = question.lower()

        if any(word in question_lower for word in ["sequence", "series", "sum", "term"]):
            return "sequence"
        elif any(word in question_lower for word in ["triangle", "circle", "area", "geometry"]):
            return "geometry"
        elif any(word in question_lower for word in ["derivative", "integral", "limit", "calculus"]):
            return "calculus"
        elif any(word in question_lower for word in ["probability", "chance", "coin", "dice"]):
            return "probability"
        elif any(word in question_lower for word in ["equation", "solve", "algebra"]):
            return "algebra"
        else:
            return "math"

    def load_engineering_sample(self, file_path: str = None, max_problems: int = 5) -> List[Dict[str, Any]]:
        """Load sample problems from engineering JSONL files"""
        if file_path is None:
            engineering_dir = Path("engineering_data")
            if engineering_dir.exists():
                jsonl_files = list(engineering_dir.glob("sample_*.jsonl"))
                if not jsonl_files:
                    jsonl_files = list(engineering_dir.glob("engineering_*.jsonl"))

                if jsonl_files:
                    file_path = jsonl_files[0]
                    logger.info(f"Auto-detected engineering file: {file_path}")
                else:
                    logger.warning("No engineering JSONL files found")
                    return []
            else:
                logger.warning("Engineering data directory not found")
                return []
        else:
            file_path = Path(file_path)

        problems = []

        try:
            if not file_path.exists():
                logger.warning(f"Engineering file not found: {file_path}")
                return problems

            with open(file_path, 'r', encoding='utf-8') as f:
                count = 0
                for line_num, line in enumerate(f, 1):
                    if count >= max_problems:
                        break

                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            problem = self._convert_engineering_to_standard(data, count)
                            if problem:
                                problems.append(problem)
                                count += 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON decode error on line {line_num}: {e}")
                            continue

            logger.info(f"🔧 Loaded {len(problems)} engineering problems")
            return problems

        except Exception as e:
            logger.error(f"Error loading engineering sample: {e}")
            return problems

    def _convert_engineering_to_standard(self, data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Convert engineering data into standard format"""
        try:
            question = data.get("question", "")
            if not question:
                return None

            difficulty = "medium"
            if len(question) > 800:
                difficulty = "hard"
            elif len(question) < 300:
                difficulty = "easy"

            category = self._classify_engineering_category(question)

            problem = {
                "id": f"engineering_{data.get('id', index)}",
                "question": question,
                "category": category,
                "difficulty": difficulty,
                "dataset": "engineering",
                "metadata": {
                    "source_file": "engineering_data",
                    "original_id": data.get("id"),
                    "has_answer": "answer_content" in data,
                    "has_reasoning": "reasoning_content" in data,
                    "verifier_score": data.get("verifier_score"),
                    "community_score": data.get("community_score"),
                    "domain": "engineering"
                }
            }

            if "answer_content" in data:
                problem["answer_content"] = data["answer_content"]

            if "reasoning_content" in data:
                problem["reasoning_content"] = data["reasoning_content"]

            return problem

        except Exception as e:
            logger.warning(f"Error converting engineering data: {e}")
            return None

    def _classify_engineering_category(self, question: str) -> str:
        """Classify the engineering question category"""
        question_lower = question.lower()

        if any(word in question_lower for word in ["circuit", "electrical", "voltage", "current", "resistance"]):
            return "electrical"
        elif any(word in question_lower for word in ["structure", "beam", "load", "stress", "material"]):
            return "structural"
        elif any(word in question_lower for word in ["fluid", "flow", "pressure", "pump", "pipe"]):
            return "fluid_mechanics"
        elif any(word in question_lower for word in ["heat", "thermal", "temperature", "energy", "thermodynamics"]):
            return "thermal"
        elif any(word in question_lower for word in ["control", "system", "feedback", "signal", "transfer"]):
            return "control_systems"
        elif any(word in question_lower for word in ["software", "algorithm", "code", "program", "computer"]):
            return "software"
        else:
            return "engineering"

    def load_all_problems(self, max_math: int = 5, max_engineering: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Load all problem data (only math and engineering, no manual)"""
        all_problems = []
        datasets_info = {}

        # 🆕 只加载真实数据，不要手工数据
        math_problems = self.load_math_sample(max_problems=max_math)
        if math_problems:
            all_problems.extend(math_problems)
            datasets_info["math"] = {
                "name": "Math Reasoning",
                "description": f"Mathematical reasoning problems from GR.inc ({len(math_problems)} samples)",
                "source": "gr_inc",
                "total_problems": len(math_problems),
                "domain": "mathematics"
            }

        engineering_problems = self.load_engineering_sample(max_problems=max_engineering)
        if engineering_problems:
            all_problems.extend(engineering_problems)
            datasets_info["engineering"] = {
                "name": "Engineering Reasoning", 
                "description": f"Engineering reasoning problems from GR.inc ({len(engineering_problems)} samples)",
                "source": "gr_inc",
                "total_problems": len(engineering_problems),
                "domain": "engineering"
            }

        logger.info(f"📚 Total loaded: {len(all_problems)} problems from {len(datasets_info)} datasets")
        logger.info(f"   - Math: {len(math_problems) if math_problems else 0}")
        logger.info(f"   - Engineering: {len(engineering_problems) if engineering_problems else 0}")

        return all_problems, datasets_info

def load_problems_data(max_math: int = 5, max_engineering: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load all problems data"""
    loader = DataLoader()
    return loader.load_all_problems(max_math=max_math, max_engineering=max_engineering)