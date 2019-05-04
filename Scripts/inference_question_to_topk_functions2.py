from Scripts.predictor_functions import QAEmbed, FaissTopK, RetreiveQADoc

#pretrained_path = 'pubmed_pmc_470k/'
#ffn_weight_file = None
#bert_ffn_weight_file = 'models/bertffn_crossentropy/bertffn'
#embedding_file = 'qa_embeddings/bertffn_crossentropy.csv'

class createEmbeds:
    def __init__(self, pretrained_path = None, ffn_weight_file = None, \
        bert_ffn_weight_file = 'models/bertffn_crossentropy/bertffn',  embedding_file = 'qa_embeddings/bertffn_crossentropy.csv'):
            self.pretrained_path = pretrained_path
            self.ffn_weight_file = ffn_weight_file
            self.bert_ffn_weight_file = bert_ffn_weight_file
            self.embedding_file = embedding_file

    def get_qa_embed(self):
        qa_embed = QAEmbed(
            pretrained_path=self.pretrained_path,
            ffn_weight_file=self.ffn_weight_file,
            bert_ffn_weight_file=self.bert_ffn_weight_file
        )
        return qa_embed

    def get_faiss_topk(self):
        faiss_topk = FaissTopK(self.embedding_file)
        return faiss_topk

    doc = RetreiveQADoc(get_qa_embed(), get_faiss_topk())

#     print(doc.predict('i have a headache.',
#                       search_by='answer', topk=5, answer_only=True))
