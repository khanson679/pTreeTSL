library(tidyverse)

setwd("C:/Users/conno/git_repos/pTreeTSL")

# This script combines the Sprouse scores and the tree representation of the
# stimuli into a single csv file used to train the pTSL model

# Load our files
scores_file <- "data/response_key.csv"
scores <- read_csv(scores_file)

trees_file <- "data/ptreetsl_trees.csv"
trees <- read_csv(trees_file)

# Remove extra columns. Keep all the columns in scores in case we want
# to use them in later analysis of the model results
trees <- trees %>%
  select(id, sent, tree) %>%
  rename(item=id)

# Remove missing judgments
scores <- scores %>%
  filter(!is.na(judgment))

# Combine data
combined <- inner_join(scores, trees, by='item')

combined <- combined %>% 
  mutate(score = (zscores-min(zscores))/(max(zscores)-min(zscores)))

write_csv(combined, 'data/training_data.csv')
