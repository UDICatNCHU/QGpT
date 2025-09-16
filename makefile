PHONY: table1_embedding table3_embedding table5_embedding all_table_embedding

table1_embedding:
	python corpus_embedding_builder.py --scan-dir ./Corpora/Table1_mimo_table_length_variation --model bge_m3_flag

table3_embedding:
	python corpus_embedding_builder.py --scan-dir ./Corpora/Table3_mimo_en_table_representation --model bge_m3_flag

table5_embedding:
	python corpus_embedding_builder.py --scan-dir ./Corpora/Table5_Single_Table_Retrieval --model bge_m3_flag

all_table_embedding: table1_embedding table3_embedding table5_embedding


# Table Experiments
.PHONY: test_table1 test_table3 test_table5 all_table_test
test_table1:
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db qgpt_T1_MTLV_mimo_ch_1k_token.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db qgpt_T1_MTLV_mimo_ch_2k_token.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db qgpt_T1_MTLV_mimo_ch_5k_token.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db qgpt_T1_MTLV_mimo_ch_full_token.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db qgpt_T1_MTLV_mimo_ch_top10_rows.db --model bge_m3_flag 

	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T1_MTLV_mimo_en_1k_token.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T1_MTLV_mimo_en_2k_token.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T1_MTLV_mimo_en_5k_token.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T1_MTLV_mimo_en_full_token.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T1_MTLV_mimo_en_top10_rows.db --model bge_m3_flag

test_table3:
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T3_mimo_en_TR_desc_only.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T3_mimo_en_TR_header_only.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T3_mimo_en_TR_pT_and_desc.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T3_mimo_en_TR_pT_and_header.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T3_mimo_en_TR_pT.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T3_mimo_en_TR_QG_only.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T3_mimo_en_TR_QGpT.db --model bge_m3_flag 

test_table5:
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T5_STR_pT_MiMoT_pT.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/OTT-QA_test.json --db qgpt_T5_STR_pT_OTTQA_pT.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/FetaQA_test.json --db qgpt_T5_STR_pT_FetaQA_pT.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/E2E-WTQ_test.json --db qgpt_T5_STR_pT_E2EWTQ_pT.db --model bge_m3_flag 

	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db qgpt_T5_STR_QGpT_MiMoT_QGpT.db  --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/OTT-QA_test.json --db qgpt_T5_STR_QGpT_OTTQA_QGpT.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/FetaQA_test.json --db qgpt_T5_STR_QGpT_FetaQA_QGpT.db --model bge_m3_flag 
	python query_evaluator.py --test-file test_dataset/E2E-WTQ_test.json --db qgpt_T5_STR_QGpT_E2EWTQ_QGpT.db --model bge_m3_flag 

all_table_test: test_table1 test_table3 test_table5


.PHONY: research
research: all_table_embedding all_table_test
	
