TABLE1_FOLDER := Corpora/Table1_mimo_table_length_variation
TABLE3_FOLDER := Corpora/Table3_mimo_en_table_representation
TABLE5_FOLDER := Corpora/Table5_Single_Table_Retrieval

MODELS := bge_flag jina_colbert

.PHONY: table1_embedding table3_embedding table5_embedding all_table_embedding

table1_embedding:
	$(foreach model,$(MODELS),python corpus_embedding_builder.py --folder $(TABLE1_FOLDER) --model $(model);)

table3_embedding:
	$(foreach model,$(MODELS),python corpus_embedding_builder.py --folder $(TABLE3_FOLDER) --model $(model);)

table5_embedding:
	$(foreach model,$(MODELS),python corpus_embedding_builder.py --folder $(TABLE5_FOLDER) --model $(model);)

all_table_embedding: table1_embedding table3_embedding table5_embedding


# Table Experiments
.PHONY: test_table1

# Complete Table 1 experiment (all models, all variants)
test_table1:
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db db/bge_flag/qgpt_T1_MTLV_mimo_ch_1k_token.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db db/bge_flag/qgpt_T1_MTLV_mimo_ch_2k_token.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db db/bge_flag/qgpt_T1_MTLV_mimo_ch_5k_token.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db db/bge_flag/qgpt_T1_MTLV_mimo_ch_full_token.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db db/bge_flag/qgpt_T1_MTLV_mimo_ch_top10_rows.db --models bge_m3_flag

	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/bge_flag/qgpt_T1_MTLV_mimo_en_1k_token.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/bge_flag/qgpt_T1_MTLV_mimo_en_2k_token.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/bge_flag/qgpt_T1_MTLV_mimo_en_5k_token.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/bge_flag/qgpt_T1_MTLV_mimo_en_full_token.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/bge_flag/qgpt_T1_MTLV_mimo_en_top10_rows.db --models bge_m3_flag
	

	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db db/jina_colbert/qgpt_T1_MTLV_mimo_ch_1k_token.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db db/jina_colbert/qgpt_T1_MTLV_mimo_ch_2k_token.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db db/jina_colbert/qgpt_T1_MTLV_mimo_ch_5k_token.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db db/jina_colbert/qgpt_T1_MTLV_mimo_ch_full_token.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --db db/jina_colbert/qgpt_T1_MTLV_mimo_ch_top10_rows.db --models jina_colbert_v2

	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T1_MTLV_mimo_en_1k_token.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T1_MTLV_mimo_en_2k_token.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T1_MTLV_mimo_en_5k_token.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T1_MTLV_mimo_en_full_token.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T1_MTLV_mimo_en_top10_rows.db --models jina_colbert_v2
	

test_table3:
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_desc_only.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_header_only.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_pT_and_desc.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_pT_and_header.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_pT.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_QG_only.db --models bge_m3_flag
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_QGpT.db --models bge_m3_flag

	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_desc_only.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_header_only.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_pT_and_desc.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_pT_and_header.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_pT.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_QG_only.db --models jina_colbert_v2
	python query_evaluator.py --test-file test_dataset/MiMoTable-English_test.json --db db/jina_colbert/qgpt_T3_mimo_en_TR_QGpT.db --models jina_colbert_v2