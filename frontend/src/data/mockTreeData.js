export const mockTreeData = {
  problem: {
    question: "Let S be the sum of the first nine terms of the sequence x+a, x²+2a, x³+3a, ... Then S equals:",
    options: ["A) (50a+x+x⁸)/(x+1)", "B) 50a-(x+x¹⁰)/(x-1)", "C) (x⁹-1)/(x+1)+45a", "D) (x¹⁰-x)/(x-1)+45a"]
  },
  beam_tree: {
    "root": {
      id: "root",
      content: "Problem Analysis",
      reasoning_type: "start",
      quality_score: 1.0,
      probability: 1.0,
      parent: null,
      children: ["approach_1", "approach_2", "approach_3"],
      variables: ["S = sum of 9 terms", "sequence: x^k + ka"],
      depth: 0
    },
    "approach_1": {
      id: "approach_1", 
      content: "Identify general term: x^k + ka",
      reasoning_type: "problem_understanding",
      quality_score: 0.9,
      probability: 0.75,
      parent: "root",
      children: ["split_sum_1", "direct_calc_1"],
      variables: ["general_term = x^k + ka", "k = 1,2,...,9"],
      depth: 1
    },
    "approach_2": {
      id: "approach_2",
      content: "Try substitution x = 1",
      reasoning_type: "problem_understanding", 
      quality_score: 0.4,
      probability: 0.6,
      parent: "root",
      children: ["substitute_1"],
      variables: ["x = 1", "simplified approach"],
      depth: 1
    },
    "approach_3": {
      id: "approach_3",
      content: "Look for pattern in coefficients",
      reasoning_type: "problem_understanding",
      quality_score: 0.7,
      probability: 0.65,
      parent: "root", 
      children: ["pattern_analysis"],
      variables: ["coefficients: 1,2,3,...", "pattern recognition"],
      depth: 1
    },
    "split_sum_1": {
      id: "split_sum_1",
      content: "S = Σ(x^k) + aΣ(k)",
      reasoning_type: "calculation",
      quality_score: 0.95,
      probability: 0.85,
      parent: "approach_1",
      children: ["geometric_series", "arithmetic_series"],
      variables: ["sum_powers = Σ(x^k)", "sum_coeffs = Σ(k)", "k=1 to 9"],
      depth: 2
    },
    "direct_calc_1": {
      id: "direct_calc_1",
      content: "Calculate term by term",
      reasoning_type: "calculation",
      quality_score: 0.6,
      probability: 0.45,
      parent: "approach_1",
      children: ["manual_sum"],
      variables: ["term_1 = x + a", "term_2 = x² + 2a", "..."],
      depth: 2
    },
    "substitute_1": {
      id: "substitute_1",
      content: "S = 9 + 45a when x = 1",
      reasoning_type: "calculation",
      quality_score: 0.3,
      probability: 0.4,
      parent: "approach_2",
      children: ["wrong_conclusion"],
      variables: ["x = 1", "S = 9 + 45a"],
      depth: 2
    },
    "pattern_analysis": {
      id: "pattern_analysis", 
      content: "Coefficients: 1,2,3,...,9",
      reasoning_type: "calculation",
      quality_score: 0.8,
      probability: 0.7,
      parent: "approach_3",
      children: ["sum_formula"],
      variables: ["coeffs = [1,2,3,...,9]", "sum_coeffs = 45"],
      depth: 2
    },
    "geometric_series": {
      id: "geometric_series",
      content: "Σ(x^k) = x(1-x^9)/(1-x)",
      reasoning_type: "calculation", 
      quality_score: 0.9,
      probability: 0.8,
      parent: "split_sum_1",
      children: ["simplify_geo"],
      variables: ["geometric_sum", "r = x", "n = 9"],
      depth: 3
    },
    "arithmetic_series": {
      id: "arithmetic_series",
      content: "Σ(k) = 9×10/2 = 45",
      reasoning_type: "calculation",
      quality_score: 0.95,
      probability: 0.9,
      parent: "split_sum_1", 
      children: ["combine_results"],
      variables: ["sum_k = 45", "k = 1 to 9"],
      depth: 3
    },
    "simplify_geo": {
      id: "simplify_geo",
      content: "= x(x^9-1)/(x-1) = (x^10-x)/(x-1)",
      reasoning_type: "calculation",
      quality_score: 0.98,
      probability: 0.85,
      parent: "geometric_series",
      children: ["final_answer_correct"],
      variables: ["numerator = x^10 - x", "denominator = x - 1"],
      depth: 4
    },
    "combine_results": {
      id: "combine_results", 
      content: "S = (x^10-x)/(x-1) + 45a",
      reasoning_type: "conclusion",
      quality_score: 0.97,
      probability: 0.92,
      parent: "arithmetic_series",
      children: ["final_answer_correct"],
      variables: ["geometric_part", "arithmetic_part = 45a"],
      depth: 4
    },
    "final_answer_correct": {
      id: "final_answer_correct",
      content: "Answer: D) (x^10-x)/(x-1) + 45a",
      reasoning_type: "conclusion",
      quality_score: 0.99,
      probability: 0.95,
      parent: "combine_results",
      children: [],
      variables: ["final_answer = D", "confidence = high"],
      depth: 5
    },
    "manual_sum": {
      id: "manual_sum",
      content: "S = (x+a) + (x²+2a) + ...",
      reasoning_type: "calculation",
      quality_score: 0.7,
      probability: 0.5,
      parent: "direct_calc_1",
      children: ["incomplete_calc"],
      variables: ["manual_expansion", "tedious_approach"],
      depth: 3
    },
    "wrong_conclusion": {
      id: "wrong_conclusion",
      content: "Choose C) (x^9-1)/(x+1)+45a",
      reasoning_type: "conclusion",
      quality_score: 0.2,
      probability: 0.3,
      parent: "substitute_1",
      children: [],
      variables: ["wrong_answer = C", "faulty_reasoning"],
      depth: 3
    },
    "sum_formula": {
      id: "sum_formula",
      content: "Need to handle x^k terms properly",
      reasoning_type: "calculation",
      quality_score: 0.8,
      probability: 0.6,
      parent: "pattern_analysis",
      children: ["partial_solution"],
      variables: ["x_terms", "coefficient_terms"],
      depth: 3
    },
    "incomplete_calc": {
      id: "incomplete_calc",
      content: "This approach is too complex...",
      reasoning_type: "conclusion",
      quality_score: 0.4,
      probability: 0.2,
      parent: "manual_sum",
      children: [],
      variables: ["incomplete", "abandoned"],
      depth: 4
    },
    "partial_solution": {
      id: "partial_solution",
      content: "Partially correct but incomplete",
      reasoning_type: "conclusion", 
      quality_score: 0.6,
      probability: 0.4,
      parent: "sum_formula",
      children: [],
      variables: ["partial_answer", "needs_more_work"],
      depth: 4
    }
  },
  paths: [
    {
      id: "path_excellent",
      nodes: ["root", "approach_1", "split_sum_1", "geometric_series", "simplify_geo", "final_answer_correct"],
      quality: "excellent",
      score: 0.96,
      is_correct: true,
      final_answer: "D"
    },
    {
      id: "path_good",
      nodes: ["root", "approach_1", "split_sum_1", "arithmetic_series", "combine_results", "final_answer_correct"], 
      quality: "good",
      score: 0.91,
      is_correct: true,
      final_answer: "D"
    },
    {
      id: "path_poor",
      nodes: ["root", "approach_2", "substitute_1", "wrong_conclusion"],
      quality: "poor", 
      score: 0.32,
      is_correct: false,
      final_answer: "C"
    }
  ]
};