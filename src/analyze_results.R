library(tidyverse)

feature_no_fixed <- fix_colnames("../results/results_features_only_no_fixed.csv")
feature_fixed <- fix_colnames("../results/results_features_only_fixed.csv")

phonetic_no_fixed <- fix_colnames("../results/results_phonetic_no_fixed.csv")
phonetic_fixed <- fix_colnames("../results/results_phonetic_fixed.csv")

full_no_fixed <- fix_colnames("../results/results_full_no_fixed.csv")
full_fixed <- fix_colnames("../results/results_full_fixed.csv")


# Fix bug in column names
fix_colnames <- function(filepath) {
  results <- read_csv(filepath)
  # base <- colnames(results)[1:5]
  # non_params <- colnames(results)[6:8]
  # params <- colnames(results)[9:length(colnames(results))]
  # colnames(results) <- c(base, params, non_params)
  # write_csv(results, filepath)
  return(results)
}

summarize_results <- function(results) {
  results %>% 
    group_by(beta) %>%
    summarize(mean_sse = mean(sse),
              best_sse = min(sse))
}

summarize_results(feature_no_fixed)
summarize_results(feature_fixed)
summarize_results(phonetic_no_fixed)
summarize_results(phonetic_fixed)
summarize_results(full_no_fixed)
summarize_results(full_fixed)

best_probs <- phonetic_no_fixed %>%
  filter(sse == min(sse)) %>%
  select(-name, -sse, -obj, -model_score, -item, -human_score) %>%
  pivot_longer(!c(beta, itr), names_to = "variable") %>%
  group_by(beta, itr, variable) %>%
  summarize(value=mean(value)) %>%
  ungroup() %>%
  select(variable, value)

best_no_fixed <- phonetic_no_fixed %>%
  filter(sse == min(sse))

best_fixed <- phonetic_fixed %>%
  filter(sse == min(sse))

ggplot(best_no_fixed, aes(x=human_score, y=model_score)) +
  geom_point()
ggsave('no_fixed_plot.png')

ggplot(best_fixed, aes(x=human_score, y=model_score)) +
  geom_point()
ggsave('fixed_plot.png')

phonetic_fixed %>% 
  group_by(item) %>%
  summarize(mean_score = mean(model_score)) %>%
  filter(mean_score == 0) %>% 
  select(item) %>% write_csv('sketchy_sentences.csv')
