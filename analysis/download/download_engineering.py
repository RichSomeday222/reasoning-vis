import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import agi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_engineering_dataset(api_key=None, model="DeepSeek-R1", output_dir="engineering_data"):
    """下载engineering数据集"""
    
    # 获取API密钥
    api_key = api_key or os.environ.get("AGI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set AGI_API_KEY environment variable or provide api_key")
    
    print(f"🔧 Downloading engineering dataset with model: {model}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化客户端
    client = agi.Client(api_key)
    
    try:
        # 下载数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"engineering_{model}_{timestamp}.jsonl"
        
        logger.info(f"📡 Downloading engineering data...")
        data_response = client.data.get(
            task='engineering',
            model=model,
            save_as=output_filename,
            download_dir=str(output_path)
        )
        
        logger.info(f"✅ Download completed!")
        logger.info(f"📊 Response: {data_response}")
        
        # 检查下载的文件
        possible_files = [
            output_path / output_filename,
            Path("gr_data") / output_filename,
            Path("gr_data") / f"engineering_{model}.jsonl",
            Path("gr_data") / f"task_data_engineering_{model}.jsonl"
        ]
        
        downloaded_file = None
        for possible_file in possible_files:
            if possible_file.exists():
                logger.info(f"📁 Found downloaded file: {possible_file}")
                downloaded_file = possible_file
                break
        
        if downloaded_file:
            # 移动到正确位置
            final_output_file = output_path / output_filename
            if downloaded_file != final_output_file:
                downloaded_file.rename(final_output_file)
                logger.info(f"📁 Moved file to: {final_output_file}")
            
            # 创建样本文件（前50行）
            sample_file = output_path / f"sample_{output_filename}"
            create_sample_file(final_output_file, sample_file, max_lines=50)
            
            logger.info(f"🎉 Engineering dataset downloaded successfully!")
            logger.info(f"📄 Full dataset: {final_output_file}")
            logger.info(f"📄 Sample file: {sample_file}")
            
            return True
            
        else:
            logger.error("❌ No downloaded file found")
            return False
            
    except Exception as e:
        logger.error(f"💥 Download failed: {str(e)}")
        return False

def create_sample_file(source_file, sample_file, max_lines=50):
    """创建样本文件（前N行）"""
    try:
        with open(source_file, 'r', encoding='utf-8') as src:
            with open(sample_file, 'w', encoding='utf-8') as dst:
                for i, line in enumerate(src):
                    if i >= max_lines:
                        break
                    dst.write(line)
        
        logger.info(f"📄 Created sample file with {max_lines} lines: {sample_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample file: {e}")

def analyze_engineering_data(file_path):
    """分析engineering数据集"""
    try:
        problems = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > 10:  # 只分析前10行
                    break
                    
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        problems.append(data)
                    except json.JSONDecodeError:
                        continue
        
        print("\n" + "="*60)
        print("🔧 ENGINEERING DATASET ANALYSIS")
        print("="*60)
        print(f"Sample size: {len(problems)} problems")
        
        if problems:
            # 显示第一个问题的结构
            first_problem = problems[0]
            print(f"\nFirst problem structure:")
            for key, value in first_problem.items():
                if isinstance(value, str):
                    value_preview = value[:100] + "..." if len(value) > 100 else value
                else:
                    value_preview = str(value)
                print(f"  {key}: {value_preview}")
            
            # 统计信息
            has_question = sum(1 for p in problems if 'question' in p)
            has_reasoning = sum(1 for p in problems if 'reasoning_content' in p)
            has_answer = sum(1 for p in problems if 'answer_content' in p)
            
            print(f"\nContent statistics:")
            print(f"  Problems with question: {has_question}/{len(problems)}")
            print(f"  Problems with reasoning: {has_reasoning}/{len(problems)}")
            print(f"  Problems with answer: {has_answer}/{len(problems)}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")

def main():
    """主函数"""
    print("🚀 Starting engineering dataset download...")
    
    # 下载engineering数据集
    success = download_engineering_dataset(
        model="DeepSeek-R1",
        output_dir="engineering_data"
    )
    
    if success:
        # 分析下载的数据
        data_dir = Path("engineering_data")
        jsonl_files = list(data_dir.glob("engineering_*.jsonl"))
        
        if jsonl_files:
            latest_file = max(jsonl_files, key=lambda x: x.stat().st_mtime)
            print(f"\n📊 Analyzing downloaded data: {latest_file}")
            analyze_engineering_data(latest_file)
            
            print(f"\n🎯 Next steps:")
            print(f"1. Check the engineering data quality")
            print(f"2. Update data_loader.py to support engineering dataset")
            print(f"3. Test API with both math and engineering problems")
        
    else:
        print("❌ Download failed. Please check your API key and try again.")

if __name__ == "__main__":
    main()