.PHONY: table1_ch_embedding

table1_ch_embedding:
	python corpus_embedding_builder.py --folder Corpora/Table1_mimo_table_length_variation/mimo_ch --model bge_flag
	python corpus_embedding_builder.py --folder Corpora/Table1_mimo_table_length_variation/mimo_ch --model bge_milvus
	python corpus_embedding_builder.py --folder Corpora/Table1_mimo_table_length_variation/mimo_ch --model jina_colbert