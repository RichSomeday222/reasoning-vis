import time
from local_deepseek_engine import LocalDeepSeekR1Engine

def test_persistent_deepseek():
    print("Testing Fully Persistent DeepSeek-R1...")
    
    engine = LocalDeepSeekR1Engine(
        cache_dir="/app/models",
        load_in_4bit=False,
        gpu_id=1
    )
    
    print("Loading model...")
    load_start = time.time()
    
    if engine.load_model():
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.1f}s!")
        
        question = "Solve: 2x + 5 = 13"
        print(f"Testing: {question}")
        
        result = engine.generate_reasoning(question, max_new_tokens=2000)
        
        if result["success"]:
            print(f"Success! Time: {result['generation_time']:.1f}s")
            print(f"Thinking: {result['thinking_content'][:150]}...")
            print(f"Answer: {result['final_answer'][:100]}...")
        else:
            print(f"Failed: {result['error']}")
        
        print("Test completed! Model will persist across restarts.")
        
    else:
        print("Failed to load model")

if __name__ == "__main__":
    test_persistent_deepseek()
