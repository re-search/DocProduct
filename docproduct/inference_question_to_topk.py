from docproduct.predictor import RetreiveQADoc

pretrained_path = 'pubmed_pmc_470k/'
ffn_weight_file = None
bert_ffn_weight_file = 'models/bertffn_crossentropy/bertffn'
embedding_file = 'qa_embeddings/bertffn_crossentropy.pkl'

doc = RetreiveQADoc(pretrained_path=pretrained_path,
                    ffn_weight_file=None,
                    bert_ffn_weight_file=bert_ffn_weight_file,
                    embedding_file=embedding_file)

print(doc.predict('my eyes hurts and i have a headache.',
                  search_by='answer', topk=5, answer_only=True))
print(doc.predict('my eyes hurts and i have a headache.',
                  search_by='question', topk=5, answer_only=True))
