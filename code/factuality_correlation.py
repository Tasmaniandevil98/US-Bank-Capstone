import pandas as pd
from factuality_evaluation import SimilarityEvaluator
from scipy.stats import pearsonr, spearmanr

# Load datasets
llm_openai = pd.read_csv("llm1_data.csv")
llm_llama = pd.read_csv("llm2_data.csv")
llm_gemma = pd.read_csv("llm3_data.csv")
llm_data4 = pd.read_csv("llm4_data.csv")

# Create an instance of SimilarityEvaluator
evaluator = SimilarityEvaluator()


def calculate_similarity_scores(llm_df, evaluator):
    similarity_scores_bert = []
    similarity_scores_transformer = []
    similarity_scores_bleu = []
    similarity_scores_rouge = []
    similarity_scores_tfidf = []
    similarity_scores_jaccard = []

    for index, row in llm_df.iterrows():
        results = evaluator.evaluate_similarity(
            row['Prompt_Output'], row['LLM_Output'])
        similarity_scores_bert.append(results["BERT_SCORE"])
        similarity_scores_transformer.append(results["SENTENCE_TRANSFORMERS"])
        similarity_scores_bleu.append(results["BLEU_SCORE"])
        similarity_scores_rouge.append(results["ROUGE_SCORE"])
        similarity_scores_tfidf.append(results["TFIDF"])
        similarity_scores_jaccard.append(results["JACCARD_SIMILARITY"])

    return (similarity_scores_bert, similarity_scores_transformer,
            similarity_scores_bleu, similarity_scores_rouge,
            similarity_scores_tfidf, similarity_scores_jaccard)


similarity_scores_llm_openai = calculate_similarity_scores(
    llm_openai, evaluator)
similarity_scores_llm_llama = calculate_similarity_scores(llm_llama, evaluator)
similarity_scores_llm_gemma = calculate_similarity_scores(llm_gemma, evaluator)
similarity_scores_llm_data4 = calculate_similarity_scores(llm_data4, evaluator)

# Create DataFrame to store the results
consolidated_df = pd.DataFrame({
    'Prompt_Output': llm_openai['Prompt_Output'],
    'LLM_OpenAI_Output': llm_openai['LLM_Output'],
    'LLM_LLAMA_Output': llm_llama['LLM_Output'],
    'LLM_GEMMA_Output': llm_gemma['LLM_Output'],
    'LLM_Data4_Output': llm_data4['LLM_Output'],
    'Human_Rank_LLM_OpenAI': llm_openai['Human_Rank'],
    'Human_Rank_LLM_LLAMA': llm_llama['Human_Rank'],
    'Human_Rank_LLM_GEMMA': llm_gemma['Human_Rank'],
    'Human_Rank_LLM_Data4': llm_data4['Human_Rank'],
    'BERT_SCORE_LLM_OpenAI': similarity_scores_llm_openai[0],
    'SENTENCE_TRANSFORMERS_LLM_OpenAI': similarity_scores_llm_openai[1],
    'BLEU_SCORE_LLM_OpenAI': similarity_scores_llm_openai[2],
    'ROUGE_SCORE_LLM_OpenAI': similarity_scores_llm_openai[3],
    'TFIDF_LLM_OpenAI': similarity_scores_llm_openai[4],
    'JACCARD_SIMILARITY_LLM_OpenAI': similarity_scores_llm_openai[5],
    'BERT_SCORE_LLM_LLAMA': similarity_scores_llm_llama[0],
    'SENTENCE_TRANSFORMERS_LLM_LLAMA': similarity_scores_llm_llama[1],
    'BLEU_SCORE_LLM_LLAMA': similarity_scores_llm_llama[2],
    'ROUGE_SCORE_LLM_LLAMA': similarity_scores_llm_llama[3],
    'TFIDF_LLM_LLAMA': similarity_scores_llm_llama[4],
    'JACCARD_SIMILARITY_LLM_LLAMA': similarity_scores_llm_llama[5],
    'BERT_SCORE_LLM_GEMMA': similarity_scores_llm_gemma[0],
    'SENTENCE_TRANSFORMERS_LLM_GEMMA': similarity_scores_llm_gemma[1],
    'BLEU_SCORE_LLM_GEMMA': similarity_scores_llm_gemma[2],
    'ROUGE_SCORE_LLM_GEMMA': similarity_scores_llm_gemma[3],
    'TFIDF_LLM_GEMMA': similarity_scores_llm_gemma[4],
    'JACCARD_SIMILARITY_LLM_GEMMA': similarity_scores_llm_gemma[5],
    'BERT_SCORE_LLM_Data4': similarity_scores_llm_data4[0],
    'SENTENCE_TRANSFORMERS_LLM_Data4': similarity_scores_llm_data4[1],
    'BLEU_SCORE_LLM_Data4': similarity_scores_llm_data4[2],
    'ROUGE_SCORE_LLM_Data4': similarity_scores_llm_data4[3],
    'TFIDF_LLM_Data4': similarity_scores_llm_data4[4],
    'JACCARD_SIMILARITY_LLM_Data4': similarity_scores_llm_data4[5]
})

# Save the results to a CSV file
consolidated_df.to_csv("results.csv", index=False)

prefixes = set(col.split('_')[0] for col in consolidated_df.columns[1:])
for prefix in prefixes:
    prefix_columns = [
        col for col in consolidated_df.columns if col.startswith(prefix)]
    ranks = consolidated_df[prefix_columns].rank(
        axis=1, method='min', ascending=False)
    rank_columns = [f'Rank_{col}' for col in prefix_columns]
    consolidated_df[rank_columns] = ranks.astype(int)

consolidated_df.to_csv("results_with_rank.csv", index=False)

# # Calculate correlation coefficients using Spearman correlation
# corr_algo1_openai = consolidated_df['Human_Rank_LLM_OpenAI'].corr(
#     consolidated_df['BERT_SCORE_LLM_OpenAI'], method='spearman')
# corr_algo2_llama = consolidated_df['Human_Rank_LLM_LLAMA'].corr(
#     consolidated_df['BERT_SCORE_LLM_LLAMA'], method='spearman')
# corr_algo3_gemma = consolidated_df['Human_Rank_LLM_GEMMA'].corr(
#     consolidated_df['BERT_SCORE_LLM_GEMMA'], method='spearman')
# corr_algo4_data4 = consolidated_df['Human_Rank_LLM_Data4'].corr(
#     consolidated_df['BERT_SCORE_LLM_Data4'], method='spearman')

# # Print correlation coefficients
# print("Spearman correlation coefficient for BERT Score and Human Rank in LLM OpenAI:", corr_algo1_openai)
# print("Spearman correlation coefficient for BERT Score and Human Rank in LLM LLAMA:", corr_algo2_llama)
# print("Spearman correlation coefficient for BERT Score and Human Rank in LLM GEMMA:", corr_algo3_gemma)
# print("Spearman correlation coefficient for BERT Score and Human Rank in LLM Data4:", corr_algo4_data4)


# # Choose the algorithm with the highest Pearson correlation coefficient
# best_algorithm = max((corr_algo1, 'Algorithm 1'),
#                      (corr_algo2, 'Algorithm 2'), (corr_algo3, 'Algorithm 3'))
# print("The best performing similarity algorithm is",
#       best_algorithm[1], "with a correlation coefficient of", best_algorithm[0])

llms = ['OpenAI', 'LLAMA', 'GEMMA', 'Data4']
metrics = ['BERT_SCORE', 'SENTENCE_TRANSFORMERS',
           'BLEU_SCORE', 'ROUGE_SCORE', 'JACCARD_SIMILARITY', 'TFIDF']

correlations = {}

for llm in llms:
    correlations[llm] = {}
    human_rank_column = f'Human_Rank_LLM_{llm}'
    for metric in metrics:
        rank_column = f'Rank_{metric}_LLM_{llm}'
        correlation = consolidated_df[[
            rank_column, human_rank_column]].corr().iloc[0, 1]
        correlations[llm][metric] = correlation
        # correlation = consolidated_df[[rank_column, human_rank_column]].corr(
        #     method='spearman').iloc[0, 1]


# Print the correlations
for llm, metric_correlations in correlations.items():
    print(f"LLM: {llm}")
    for metric, corr in metric_correlations.items():
        print(f"  {metric}: {corr}")
    print()
