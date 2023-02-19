library(tidyverse)

# df <- read_csv("results/prespecified.csv", col_names = FALSE)
# colnames(df) <- c("item", 'human_score', 'model_score', 'difference')


df <- read_csv("../results/agg/fixed_c_free_wh.csv")

# Visualize learned parameters
df %>%
  select(-itr, -sse, -obj, -item, -model_score, -human_score) %>%
  pivot_longer(c(-beta, -name), names_to="param") %>%
  ggplot(aes(x=param, y=value)) +
  geom_boxplot()
ggsave("../figs/agg/params_fixed_c_free_wh.png")

# Plot scores
plot_df <- df %>%
  separate(item, into=c("type1", "type2", "island", "matrix", "num"), sep="\\.") %>%
  unite('type', type1, type2, sep=".")

# Overall correlation
ggplot(plot_df, aes(x=human_score, y=model_score)) +
  geom_point()

# Score results
scores_df <- plot_df %>% 
  group_by(type, island, matrix) %>%
  summarize(score_mean = mean(model_score),
            score_sd = sd(model_score),
            human_mean = mean(human_score),
            human_sd = sd(human_score))

ggplot(scores_df, aes(x=matrix, y=score_mean, color=island, group=island)) +
  geom_line() +
  geom_point() + 
  facet_grid(~type)
ggsave("../figs/agg/model_scores_fixed_c_free_wh.png")

ggplot(scores_df, aes(x=matrix, y=human_mean, color=island, group=island)) +
  geom_line() +
  geom_point() + 
  facet_grid(~type)
ggsave("../figs/human_scores.png")
