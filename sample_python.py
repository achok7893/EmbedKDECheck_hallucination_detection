from src.modules.llm_bert import BertEmbeddingModel
import src.modules.embed2kde as embed2kde

bert_model = BertEmbeddingModel(model_name='./data/m_models/bert-base-uncased')
embed2kde.get_scores_from_input_output_texts("I am working the day", "I am working the day", bert_model)
