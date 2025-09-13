# -*- coding: utf-8 -*-
"""
QGpT: Improving Table Retrieval with Question Generation from Partial Tables
Query Evaluator with Multi-K Recall Metrics

This script evaluates query performance using test queries against built vector databases.
Now supports Recall@1, Recall@3, Recall@5, Recall@10 evaluation.

Usage:
    python query_evaluator.py "query text" --db database.db
    python query_evaluator.py --test-file test_queries.json --db database.db
    python query_evaluator.py --batch-eval  # 批次評估所有測試集

Author: QGpT Research Team
Repository: https://github.com/UDICatNCHU/QGpT
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel

from utils import (
    load_json_dataset,
    format_search_results,
    get_corpus_files,
    extract_corpus_name_from_path,
    generate_db_name,
    generate_collection_name
)
from embedding_models import EmbeddingModel, MODELS

# 固定的評估 K 值
EVAL_K_VALUES = [1, 3, 5, 10]


def normalize_filename(filepath: str) -> str:
    """標準化檔案路徑，只保留檔名"""
    if isinstance(filepath, str):
        return filepath.split('/')[-1].strip()
    return str(filepath)


def calculate_recall_at_k(retrieved_files: List[str], ground_truth: List[str], 
                         k_values: List[int] = EVAL_K_VALUES) -> Dict[str, float]:
    """
    計算多個 K 值的召回率指標
    
    Args:
        retrieved_files: 檢索到的檔案列表（已排序）
        ground_truth: 正確答案檔案列表
        k_values: 要計算的 K 值列表
        
    Returns:
        各個 K 值的召回率字典
    """
    if not ground_truth:
        return {f'recall_at_{k}': 0.0 for k in k_values}
    
    normalized_ground_truth = [normalize_filename(gt) for gt in ground_truth]
    normalized_retrieved = [normalize_filename(rf) for rf in retrieved_files]
    
    recall_metrics = {}
    for k in k_values:
        # 只考慮前 k 個結果
        top_k_retrieved = normalized_retrieved[:k]
        hits = sum(1 for gt in normalized_ground_truth if gt in top_k_retrieved)
        recall_metrics[f'recall_at_{k}'] = hits / len(ground_truth)
    
    return recall_metrics


class QGpTQueryEvaluator:
    """QGpT 查詢評估器"""
    
    def __init__(self, db_path: str, collection_name: str, model: str = "bge_m3_flag"):
        """
        初始化查詢評估器
        
        Args:
            db_path: 向量資料庫路徑
            collection_name: 向量集合名稱
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.embedding_fn = None
        self.initialize(model)
    
    def initialize(self, model):
        """初始化 Milvus 客戶端和嵌入函數"""
        try:
            if not Path(self.db_path).exists():
                raise FileNotFoundError(f"找不到資料庫檔案: {self.db_path}")
            
            self.client = MilvusClient(self.db_path)
            print("🔄 初始化 BGE-M3 模型...")
            self.embedding_fn = MODELS.get(model)()

            print("✅ BGE-M3 模型載入完成")
            
            # 檢查集合是否存在
            if not self.client.has_collection(collection_name=self.collection_name):
                raise ValueError(f"找不到集合 '{self.collection_name}' 在資料庫 '{self.db_path}'")
            
            print(f"✅ 成功連接到資料庫: {self.db_path}")
            print(f"✅ 使用集合: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ 初始化失敗: {e}")
            sys.exit(1)
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        執行搜索查詢
        
        Args:
            query: 搜索查詢字符串
            limit: 返回結果的數量
            
        Returns:
            搜索結果列表
        """
        try:
            # 將查詢轉換為向量 (使用 BGE-M3)
            query_vector = self.embedding_fn.encode([query])['dense_vecs'][0].astype('float32').tolist()
            
            # 執行搜索
            search_results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],  # 需要包裝成列表
                limit=limit,
                output_fields=["text", "filename", "sheet_name", "original_id"]
            )
            
            # 格式化結果
            formatted_results = []
            for result in search_results[0]:
                formatted_results.append({
                    'score': 1 - result['distance'],  # 轉換為相似度分數
                    'distance': result['distance'],
                    'filename': result['entity']['filename'],
                    'sheet_name': result['entity']['sheet_name'],
                    'original_id': result['entity']['original_id'],
                    'text': result['entity']['text']
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ 搜索錯誤: {e}")
            return []
    
    def evaluate_single_query(self, query: str, ground_truth: Optional[List[str]] = None) -> Dict:
        """
        評估單一查詢，計算多個 K 值的召回率
        
        Args:
            query: 查詢字符串
            ground_truth: 正確答案列表（檔案名或ID）
            
        Returns:
            評估結果，包含多個 K 值的召回率
        """
        # 搜尋 top-10 結果（足夠計算所有 K 值）
        results = self.search(query, max(EVAL_K_VALUES))
        
        evaluation = {
            'query': query,
            'results_count': len(results),
            'results': results
        }
        
        # 如果有正確答案，計算召回率指標
        if ground_truth:
            retrieved_files = [r['filename'] for r in results]
            recall_metrics = calculate_recall_at_k(retrieved_files, ground_truth, EVAL_K_VALUES)
            evaluation.update(recall_metrics)
        
        return evaluation


class BatchEvaluator:
    """批次評估器"""
    
    def __init__(self):
        self.test_files_dir = "Test_Query_and_GroundTruth_Table"
    
    def get_test_files(self) -> List[str]:
        """取得所有測試檔案"""
        test_dir = Path(self.test_files_dir)
        if not test_dir.exists():
            return []
        
        return [str(f) for f in test_dir.glob("*.json")]
    
    def match_corpus_to_test(self, test_file: str) -> Optional[str]:
        """
        將測試檔案匹配到對應的語料庫
        
        Args:
            test_file: 測試檔案路徑
            
        Returns:
            對應的語料庫資料庫路徑，如果找不到則返回 None
        """
        test_name = Path(test_file).stem
        
        # 定義測試檔案到語料庫的映射規則
        mapping_rules = {
            'MiMoTable-English': 'Table1_mimo_table_length_variation_mimo_en',
            'MiMoTable-Chinese': 'Table1_mimo_table_length_variation_mimo_ch',
            'E2E-WTQ': 'Table5_Single_Table_Retrieval_QGpT',
            'FetaQA': 'Table5_Single_Table_Retrieval_QGpT',
            'OTT-QA': 'Table7_OTTQA',
            'MMQA-2tables': 'Table6_Multi_Table_Retrieval_2_tables',
            'MMQA-3tables': 'Table6_Multi_Table_Retrieval_3_tables'
        }
        
        # 尋找匹配的語料庫
        for test_key, corpus_pattern in mapping_rules.items():
            if test_key in test_name:
                # 尋找對應的資料庫檔案
                corpus_files = get_corpus_files()
                for corpus_info in corpus_files:
                    if corpus_pattern in corpus_info['name']:
                        db_path = corpus_info['db_name']
                        if Path(db_path).exists():
                            return db_path
        
        return None
    
    def evaluate_test_file(self, test_file: str, db_path: str, model: str = "bge_m3_flag") -> Dict:
        """
        評估測試檔案，計算多個 K 值的平均召回率
        
        Args:
            test_file: 測試檔案路徑
            db_path: 資料庫路徑
            
        Returns:
            評估結果，包含各個 K 值的平均召回率
        """
        # 載入測試資料
        test_data = load_json_dataset(test_file)
        
        # 根據資料庫路徑生成集合名稱
        corpus_name = Path(db_path).stem.replace('qgpt_', '')
        collection_name = generate_collection_name(corpus_name)
        
        # 初始化評估器
        evaluator = QGpTQueryEvaluator(db_path, collection_name, model)
        
        results = []
        recall_sums = {f'recall_at_{k}': 0.0 for k in EVAL_K_VALUES}
        
        print(f"🔄 評估測試檔案: {Path(test_file).name}")
        print(f"   查詢數量: {len(test_data)}")
        
        for i, item in enumerate(test_data):
            query = item.get('question', '')
            
            # 提取正確答案（根據測試檔案結構調整）
            ground_truth = []
            if 'spreadsheet_list' in item:
                ground_truth = item['spreadsheet_list']
            elif 'answer' in item:
                # 某些測試檔案可能有不同的結構
                pass
            
            # 執行評估
            eval_result = evaluator.evaluate_single_query(query, ground_truth)
            results.append(eval_result)
            
            # 累計各個 K 值的召回率
            for k in EVAL_K_VALUES:
                recall_key = f'recall_at_{k}'
                if recall_key in eval_result:
                    recall_sums[recall_key] += eval_result[recall_key]
            
            # 顯示進度
            if (i + 1) % 10 == 0:
                print(f"   處理進度: {i + 1}/{len(test_data)}")
        
        # 計算平均指標
        avg_recalls = {}
        for k in EVAL_K_VALUES:
            recall_key = f'recall_at_{k}'
            avg_recalls[f'avg_{recall_key}'] = recall_sums[recall_key] / len(test_data) if test_data else 0.0
        
        # 儲存個別測試檔案結果
        experiment_dir = Path("experiment")
        experiment_dir.mkdir(exist_ok=True)
        
        # 生成個別結果檔案名稱：model_name_db_name.json
        db_name = Path(db_path).stem.replace('qgpt_', '')
        individual_result_file = experiment_dir / f"{model}_{db_name}.json"
        
        result_data = {
            'test_file': test_file,
            'db_path': db_path,
            'model': model,
            'total_queries': len(test_data),
            **avg_recalls,
            'detailed_results': results
        }
        
        with open(individual_result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"📄 個別結果已儲存到: {individual_result_file}")
        
        return result_data
    
    def run_batch_evaluation(self, save_results: bool = True, model: str = "bge_m3_flag") -> Dict:
        """
        執行批次評估，計算所有測試集的多 K 值召回率
        
        Args:
            save_results: 是否儲存詳細結果
            
        Returns:
            批次評估結果
        """
        test_files = self.get_test_files()
        if not test_files:
            print("❌ 沒有找到測試檔案")
            return {}
        
        batch_results = {}
        
        print(f"🔄 開始批次評估，找到 {len(test_files)} 個測試檔案")
        print(f"📊 評估指標: {', '.join(f'Recall@{k}' for k in EVAL_K_VALUES)}")
        
        for test_file in test_files:
            print(f"\n{'='*60}")
            
            # 尋找對應的語料庫資料庫
            db_path = self.match_corpus_to_test(test_file)
            if not db_path:
                print(f"⚠️  跳過 {Path(test_file).name}：找不到對應的語料庫資料庫")
                continue
            
            try:
                # 執行評估
                result = self.evaluate_test_file(test_file, db_path, model)
                batch_results[Path(test_file).stem] = result
                
                print(f"✅ 完成評估: {Path(test_file).name}")
                for k in EVAL_K_VALUES:
                    recall_value = result[f'avg_recall_at_{k}']
                    print(f"   Recall@{k}: {recall_value:.4f}")
                
            except Exception as e:
                print(f"❌ 評估失敗: {Path(test_file).name} - {e}")
        
        # 儲存結果
        if save_results and batch_results:
            # 創建 experiment 目錄（如果不存在）
            experiment_dir = Path("experiment")
            experiment_dir.mkdir(exist_ok=True)
            
            # 生成結果檔案名稱：model_name_overall_results.json
            results_file = experiment_dir / f"{model}_overall_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 評估結果已儲存到: {results_file}")
        
        return batch_results


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description='QGpT 查詢評估器 - 多 K 值召回率版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 單一查詢測試（會計算 Recall@1,3,5,10）
  python query_evaluator.py "財務報表" --db qgpt_Table1_mimo_ch.db
  
  # 使用測試檔案評估
  python query_evaluator.py --test-file Test_Query_and_GroundTruth_Table/MiMoTable-English_test.json --db qgpt_Table1_mimo_en.db
  
  # 批次評估所有測試集（計算多個 K 值的召回率）
  python query_evaluator.py --batch-eval
        """
    )
    
    parser.add_argument('query', nargs='?', help='搜索查詢字符串')
    parser.add_argument('--db', help='向量資料庫路徑')
    parser.add_argument('--collection', help='向量集合名稱（自動從資料庫名稱推導）')
    parser.add_argument('--test-file', help='測試查詢檔案路徑')
    parser.add_argument('--batch-eval', action='store_true', help='批次評估所有測試集')
    parser.add_argument('--limit', type=int, default=10, help='顯示結果數量 (預設: 10)')
    parser.add_argument('--format', choices=['detailed', 'simple', 'json'], 
                       default='detailed', help='輸出格式 (預設: detailed)')
    parser.add_argument('--save', action='store_true', help='儲存評估結果到檔案')
    parser.add_argument('--model', default="bge_m3_flag", help='jina_colbert_v2 or bge_m3_flag (預設: bge_m3_flag)')

    args = parser.parse_args()
    
    # 批次評估
    if args.batch_eval:
        evaluator = BatchEvaluator()
        results = evaluator.run_batch_evaluation(save_results=args.save, model=args.model)
        
        # 顯示總結
        if results:
            print(f"\n{'='*60}")
            print("批次評估總結:")
            for test_name, result in results.items():
                print(f"  {test_name}:")
                print(f"    查詢數量: {result['total_queries']}")
                for k in EVAL_K_VALUES:
                    recall_value = result[f'avg_recall_at_{k}']
                    print(f"    Recall@{k}: {recall_value:.4f}")
        
        return
    
    # 單一查詢或測試檔案評估
    if not args.db:
        print("❌ 請提供資料庫路徑 (--db)")
        sys.exit(1)
    
    # 自動生成集合名稱
    if not args.collection:
        corpus_name = Path(args.db).stem.replace('qgpt_', '')
        collection_name = generate_collection_name(corpus_name)
    else:
        collection_name = args.collection
    
    # 初始化評估器
    evaluator = QGpTQueryEvaluator(args.db, collection_name, args.model)
    
    # 測試檔案評估
    if args.test_file:
        if not Path(args.test_file).exists():
            print(f"❌ 找不到測試檔案: {args.test_file}")
            sys.exit(1)
        
        batch_evaluator = BatchEvaluator()
        result = batch_evaluator.evaluate_test_file(args.test_file, args.db, args.model)
        
        print(f"\n評估結果:")
        print(f"測試檔案: {result['test_file']}")
        print(f"查詢總數: {result['total_queries']}")
        for k in EVAL_K_VALUES:
            recall_value = result[f'avg_recall_at_{k}']
            print(f"Recall@{k}: {recall_value:.4f}")
        
        if args.save:
            # 創建 experiment 目錄
            experiment_dir = Path("experiment")
            experiment_dir.mkdir(exist_ok=True)
            
            # 生成結果檔案名稱：model_name_db_name.json
            db_name = Path(args.db).stem.replace('qgpt_', '')
            results_file = experiment_dir / f"{args.model}_{db_name}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"詳細結果已儲存到: {results_file}")
        
        return
    
    # 單一查詢
    if args.query:
        results = evaluator.search(args.query, args.limit)
        output = format_search_results(results, args.query, args.format)
        print(output)
        return
    
    # 如果沒有提供參數，顯示幫助
    parser.print_help()


if __name__ == "__main__":
    main()