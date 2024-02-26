from bert_score import score
from sentence_transformers import SentenceTransformer, util
from sacrebleu.metrics import BLEU
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEvaluator:
    def bert_score_calc(cands, ref1):
        P1, R1, F1 = score(cands, ref1, lang="en", verbose=True)
        return F1

    def sentence_transformers_calc(cands, ref1):
        model = SentenceTransformer('stsb-roberta-large')
        # model = SentenceTransformer("all-MiniLM-L6-v2")
        cands_embeddings = model.encode(cands, convert_to_tensor=True)
        ref1_embeddings = model.encode(ref1, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(cands_embeddings, ref1_embeddings)
        # cosine_scores = util.cos_sim(cands_embeddings, ref1_embeddings)
        return cosine_scores

    def jaccard_similarity(cands, ref1):
        intersection_cardinality = len(
            set.intersection(*[set(cands), set(ref1)]))
        union_cardinality = len(set.union(*[set(cands), set(ref1)]))
        return intersection_cardinality/float(union_cardinality)

    def bleu_score_calc(cands, ref1):
        bleu_scorer = BLEU()
        score1 = bleu_scorer.sentence_score(
            hypothesis=cands[0],
            references=ref1,
        )
        return score1.score/100

    def rouge_score_calc(cands, ref1):
        rouge_scorer = Rouge()
        score1 = rouge_scorer.get_scores(
            hyps=cands[0],
            refs=ref1[0],
        )
        # return score1[0]['rouge-1']['f']
        return score1[0]['rouge-l']['f']

    def tfidf_similarity_calc(cands, ref1):
        vectorizer = TfidfVectorizer()
        # vectors = vectorizer.fit_transform(cands + ref1)
        # similarity = cosine_similarity(vectors)
        all_texts = cands + ref1

        tfidf_matrix = vectorizer.fit_transform(all_texts)
        tfidf_matrix_text1 = tfidf_matrix[:len(cands)]
        tfidf_matrix_text2 = tfidf_matrix[len(cands):len(cands)+len(ref1)]

        cosine_sim_tfidf1_2 = cosine_similarity(
            tfidf_matrix_text1, tfidf_matrix_text2)

        return cosine_sim_tfidf1_2[0][0]

    def evaluate_similarity(self, cands, ref1):
        results = {}
        results["BERT_SCORE"] = self.bert_score_calc(cands, ref1)
        results["SENTENCE_TRANSFORMERS"] = self.sentence_transformers_calc(
            cands, ref1)
        results["BLEU_SCORE"] = self.bleu_score_calc(cands, ref1)
        results["ROUGE_SCORE"] = self.rouge_score_calc(cands, ref1)
        results["TFIDF"] = self.tfidf_similarity_calc(cands, ref1)
        results["JACCARD_SIMILARITY"] = self.jaccard_similarity([text.lower().split(
            " ") for text in cands][0], [text.lower().split(" ") for text in ref1][0])
        return results

# if __name__ == "__main__":
#     SimilarityEvaluator = SimilarityEvaluator()
#     cands = ["The cat is on the mat"]
#     ref1 = ["There is a cat on the mat"]
#     # ref2 = ["There is a cat on the mat"]
#     print("BERT SCORE:", bert_score_calc(cands, ref1))
#     print("SENTENCE TRANSFORMERS:", sentence_transformers_calc(cands, ref1))
#     print("BLEU SCORE:", bleu_score_calc(cands, ref1))
#     print("ROGUE SCORE:", rouge_score_calc(cands, ref1))
#     print("TFIDF:", tfidf_similarity_calc(cands, ref1))
#     print(jaccard_similarity([text.lower().split(" ") for text in cands][0], [
#           text.lower().split(" ") for text in ref1][0]))
