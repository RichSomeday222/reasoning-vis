import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import time
import agi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gr_inc_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GRIncDataDownloader:
    """GR.inc Data Downloader and Processor"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize downloader"""
        
        self.api_key = api_key or os.environ.get("AGI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Please set AGI_API_KEY environment variable or provide api_key")
        print("Loaded API Key:", self.api_key)

        self.client = agi.Client(self.api_key)
        self.download_stats = {
            'total_samples': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'start_time': None,
            'end_time': None
        }

    def download_general_math_data(self,
                                    model: str = "DeepSeek-R1",
                                    output_dir: str = "data",
                                    max_retries: int = 3) -> bool:
        """Download general-math dataset"""

        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"general_math_{model}_{timestamp}.jsonl"

            logger.info(f"🔽 Downloading general-math data from model: {model}")
            logger.info(f"📁 Output directory: {output_path}")
            logger.info(f"📄 Output filename: {output_filename}")

            self.download_stats['start_time'] = datetime.now()

            for attempt in range(max_retries):
                try:
                    logger.info(f"🔄 Attempt {attempt + 1}/{max_retries}")
                    
                    logger.info("📡 Calling API with save_as parameter...")
                    data_response = self.client.data.get(
                        task='general-math',
                        model=model,
                        save_as=output_filename,
                        download_dir=str(output_path)
                    )
                    
                    logger.info(f"✅ API call successful!")
                    logger.info(f"📊 Response: {data_response}")
                    
                    possible_files = [
                        output_path / output_filename,
                        Path("gr_data") / output_filename,
                        Path("gr_data") / f"general-math_{model}.jsonl",
                        Path("gr_data") / f"task_data_general-math_{model}.jsonl"
                    ]
                    
                    downloaded_file = None
                    for possible_file in possible_files:
                        if possible_file.exists():
                            logger.info(f"📁 Found file at: {possible_file}")
                            downloaded_file = possible_file
                            break
                    
                    if downloaded_file:
                        final_output_file = output_path / output_filename
                        if downloaded_file != final_output_file:
                            downloaded_file.rename(final_output_file)
                            logger.info(f"📁 Moved file to: {final_output_file}")
                        else:
                            final_output_file = downloaded_file
                            logger.info(f"📁 File already in correct location: {final_output_file}")
                        
                        if self.verify_downloaded_file(final_output_file):
                            logger.info(f"✅ Download completed successfully: {final_output_file}")
                            stats = self.analyze_downloaded_data(final_output_file)
                            self.download_stats.update(stats)
                            return True
                        else:
                            logger.error("❌ Downloaded file verification failed")
                            return False
                    else:
                        logger.warning("🔍 No downloaded file found, trying method 2...")
                        
                        logger.info("📡 Trying direct data retrieval...")
                        data_response = self.client.data.get(
                            task='general-math',
                            model=model
                        )
                        
                        if data_response:
                            manual_output_file = output_path / output_filename
                            self._save_data_to_file(data_response, manual_output_file)
                            
                            if self.verify_downloaded_file(manual_output_file):
                                logger.info(f"✅ Manual save successful: {manual_output_file}")
                                stats = self.analyze_downloaded_data(manual_output_file)
                                self.download_stats.update(stats)
                                return True
                    
                    break
                    
                except Exception as e:
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2)

            self.download_stats['end_time'] = datetime.now()
            return False

        except Exception as e:
            logger.error(f"💥 Failed to download general-math data: {str(e)}")
            return False

    def _save_data_to_file(self, data_response, output_file: Path):
        """Save API response data to file"""
        logger.info(f"💾 Saving data to {output_file}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                if isinstance(data_response, list):
                    for item in data_response:
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
                    logger.info(f"📝 Saved {len(data_response)} records as JSONL")
                    
                elif isinstance(data_response, dict):
                    json.dump(data_response, f, ensure_ascii=False)
                    f.write('\n')
                    logger.info("📝 Saved 1 record as JSONL")
                    
                elif isinstance(data_response, str):
                    if data_response.startswith('http'):
                        logger.info("📥 Response is URL, downloading file...")
                        import requests
                        response = requests.get(data_response)
                        response.raise_for_status()
                        f.write(response.text)
                        logger.info("📝 Downloaded and saved content from URL")
                    else:
                        f.write(data_response)
                        logger.info("📝 Saved string data")
                    
                else:
                    json.dump(str(data_response), f, ensure_ascii=False)
                    f.write('\n')
                    logger.info(f"📝 Saved data as string (type: {type(data_response)})")
                    
        except Exception as e:
            logger.error(f"❌ Failed to save data to file: {str(e)}")
            raise

    def verify_downloaded_file(self, file_path: Path) -> bool:
        """Verify if the downloaded file is valid"""
        try:
            if not file_path.exists():
                logger.error(f"❌ File does not exist: {file_path}")
                return False
                
            file_size = file_path.stat().st_size
            if file_size == 0:
                logger.error(f"❌ File is empty: {file_path}")
                return False

            logger.info(f"📊 File size: {file_size / 1024:.2f} KB")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.error("❌ File content is empty")
                    return False
                
                lines = content.split('\n')
                valid_lines = 0
                
                for i, line in enumerate(lines[:5]):
                    if line.strip():
                        try:
                            json.loads(line.strip())
                            valid_lines += 1
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Line {i+1} is not valid JSON: {line[:50]}...")
                
                if valid_lines > 0:
                    logger.info(f"✅ File format verified: {valid_lines} valid JSON lines found")
                    return True
                else:
                    logger.warning("⚠️ No valid JSON lines found, but treating as valid")
                    return True

        except Exception as e:
            logger.error(f"❌ File verification failed: {str(e)}")
            return False

    def analyze_downloaded_data(self, file_path: Path) -> Dict[str, Any]:
        """Analyze downloaded data"""

        logger.info("🔍 Analyzing downloaded data...")

        stats = {
            'total_samples': 0,
            'problem_types': {},
            'difficulty_levels': {},
            'has_reasoning_traces': 0,
            'avg_reasoning_length': 0,
            'file_size_mb': 0
        }

        try:
            stats['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
            reasoning_lengths = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            stats['total_samples'] += 1

                            problem_type = self.classify_problem_type(data)
                            stats['problem_types'][problem_type] = stats['problem_types'].get(problem_type, 0) + 1

                            if self.has_reasoning_trace(data):
                                stats['has_reasoning_traces'] += 1
                                length = self.get_reasoning_length(data)
                                if length > 0:
                                    reasoning_lengths.append(length)

                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Line {line_num}: Invalid JSON format")
                            continue

                    if line_num % 1000 == 0:
                        logger.info(f"📊 Processed {line_num} lines...")

            if reasoning_lengths:
                stats['avg_reasoning_length'] = sum(reasoning_lengths) / len(reasoning_lengths)

            self.print_data_analysis(stats)
            return stats

        except Exception as e:
            logger.error(f"❌ Data analysis failed: {str(e)}")
            return stats

    def classify_problem_type(self, data: Dict) -> str:
        """Classify problem type"""
        text_fields = ['problem', 'question', 'prompt', 'input', 'text', 'content']
        text_content = ""
        
        for field in text_fields:
            if field in data and data[field]:
                text_content = str(data[field]).lower()
                break
        
        if not text_content:
            return 'unknown'

        if any(word in text_content for word in ['prove', 'show that', 'demonstrate']):
            return 'proof'
        elif any(word in text_content for word in ['calculate', 'compute', 'find the value']):
            return 'calculation'
        elif any(word in text_content for word in ['solve', 'equation', 'system']):
            return 'equation_solving'
        elif any(word in text_content for word in ['geometry', 'triangle', 'circle', 'polygon']):
            return 'geometry'
        elif any(word in text_content for word in ['probability', 'statistics', 'random']):
            return 'probability'
        else:
            return 'general'

    def has_reasoning_trace(self, data: Dict) -> bool:
        """Check if reasoning trace exists"""
        fields = ['reasoning', 'trace', 'steps', 'solution_steps', 'thinking', 'explanation', 'solution', 'rationale']
        return any(field in data and data[field] for field in fields)

    def get_reasoning_length(self, data: Dict) -> int:
        """Get length of reasoning trace"""
        fields = ['reasoning', 'trace', 'steps', 'solution_steps', 'thinking', 'explanation', 'solution', 'rationale']
        for field in fields:
            if field in data and data[field]:
                content = data[field]
                if isinstance(content, str):
                    return len(content.split())
                elif isinstance(content, list):
                    return len(content)
        return 0

    def print_data_analysis(self, stats: Dict):
        """Print data analysis results"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 DATA ANALYSIS RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"File size: {stats['file_size_mb']:.2f} MB")
        if stats['total_samples'] > 0:
            logger.info(f"Samples with reasoning traces: {stats['has_reasoning_traces']} "
                        f"({stats['has_reasoning_traces'] / stats['total_samples'] * 100:.1f}%)")
        if stats['avg_reasoning_length'] > 0:
            logger.info(f"Average reasoning length: {stats['avg_reasoning_length']:.1f} words")

        if stats['problem_types']:
            logger.info("\n📋 Problem Types:")
            for ptype, count in sorted(stats['problem_types'].items(), key=lambda x: x[1], reverse=True):
                if stats['total_samples'] > 0:
                    percentage = count / stats['total_samples'] * 100
                    logger.info(f"  • {ptype}: {count} ({percentage:.1f}%)")

    def convert_to_dataframe(self, jsonl_file: Path) -> pd.DataFrame:
        """Convert JSONL file to DataFrame"""
        logger.info("📊 Converting JSONL to DataFrame...")

        records = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

        if not records:
            logger.warning("⚠️ No valid records found, creating empty DataFrame")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info(f"✅ DataFrame created with {len(df)} rows and {len(df.columns)} columns")
        
        if len(df) > 0:
            logger.info(f"📋 Column names: {list(df.columns)}")
            logger.info(f"📄 First record preview:")
            first_record = df.iloc[0].to_dict()
            for key, value in first_record.items():
                value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                logger.info(f"  {key}: {value_str}")
        
        return df

    def save_sample_data(self, jsonl_file: Path, sample_size: int = 100):
        """Save sample of the dataset"""
        logger.info(f"💾 Saving a sample of {sample_size} records for inspection...")

        sample_file = jsonl_file.parent / f"sample_{jsonl_file.stem}.json"
        samples = []

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                if line.strip():
                    try:
                        samples.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"📄 Sample data saved to: {sample_file}")
        logger.info(f"📊 Sample contains {len(samples)} records")

def main():
    """Main function"""

    CONFIG = {
        "model": "DeepSeek-R1",
        "output_dir": "general_math_data",
        "max_retries": 3,
        "save_sample": True,
        "sample_size": 50,
        "convert_to_csv": True
    }

    try:
        logger.info("🚀 Starting GR.inc General Math data download process...")

        downloader = GRIncDataDownloader()

        success = downloader.download_general_math_data(
            model=CONFIG["model"],
            output_dir=CONFIG["output_dir"],
            max_retries=CONFIG["max_retries"]
        )

        if success:
            data_dir = Path(CONFIG["output_dir"])
            jsonl_files = list(data_dir.glob("*.jsonl"))

            if jsonl_files:
                latest_file = max(jsonl_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"📁 Processing downloaded file: {latest_file}")

                if CONFIG["save_sample"]:
                    downloader.save_sample_data(latest_file, CONFIG["sample_size"])

                if CONFIG["convert_to_csv"]:
                    df = downloader.convert_to_dataframe(latest_file)
                    if not df.empty:
                        csv_file = latest_file.with_suffix('.csv')
                        df.to_csv(csv_file, index=False)
                        logger.info(f"📊 Data exported to CSV: {csv_file}")
                        logger.info(f"📊 DataFrame shape: {df.shape}")

                report_file = data_dir / "download_report.json"
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'config': CONFIG,
                        'stats': downloader.download_stats,
                        'timestamp': datetime.now().isoformat(),
                        'success': success
                    }, f, indent=2, default=str)

                logger.info(f"📄 Report saved to: {report_file}")
                logger.info("🎉 Data download and processing completed successfully!")
                
                logger.info("\n🎯 Next Steps:")
                logger.info("1. Check the downloaded data in the CSV file")
                logger.info("2. Examine the sample JSON for data structure")
                logger.info("3. Start implementing your beam search visualization!")

            else:
                logger.error("❌ No JSONL files were found in the output directory.")
        else:
            logger.error("❌ Data download was unsuccessful.")

    except Exception as e:
        logger.error(f"💥 An unrecoverable error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
