.PHONY: table1_ch_embedding table1_en_embedding table3_en_embedding

table1_ch_embedding:
	python corpus_embedding_builder.py --folder Corpora/Table1_mimo_table_length_variation/mimo_ch --model bge_flag
	python corpus_embedding_builder.py --folder Corpora/Table1_mimo_table_length_variation/mimo_ch --model bge_milvus
	python corpus_embedding_builder.py --folder Corpora/Table1_mimo_table_length_variation/mimo_ch --model jina_colbert

table1_en_embedding:
	python corpus_embedding_builder.py --folder Corpora/Table1_mimo_table_length_variation/mimo_en --model bge_flag
	python corpus_embedding_builder.py --folder Corpora/Table1_mimo_table_length_variation/mimo_en --model bge_milvus
	python corpus_embedding_builder.py --folder Corpora/Table1_mimo_table_length_variation/mimo_en --model jina_colbert

all_table_embedding:
	python corpus_embedding_builder.py --all --model bge_flag
	python corpus_embedding_builder.py --all --model bge_milvus
	python corpus_embedding_builder.py --all --model jina_colbert

.PHONY: test_table1_ch
test_table1_ch:
	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --model jina_colbert --table-id T1_MTLV_mimo_ch
# 	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --model bge_flag --table-id T1_MTLV_mimo_ch
# 	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --model bge_milvus --table-id T1_MTLV_mimo_ch
# 	python query_evaluator.py --test-file test_dataset/MiMoTable-Chinese_test.json --model jina_colbert --table-id T1_MTLV_mimo_ch