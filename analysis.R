rm(list = ls())
folder = c('set1/', 'set2/', 'set3/', 'set4/', 'set5/')
prefix = './results/'
df_regular <- data.frame()
labels = c(rep('regular', 100), rep('grid-search', 100))
# Read cv result
for (i in 1:5){
  cv = read.table(paste(prefix,folder[i], 'cv_result.txt', sep = ''))$V1
  gs_cv = read.table(paste(prefix,folder[i],'optimized_CV_result.txt',sep = ''))$V1
  assign(paste('df_set',i,sep = ''), data.frame(labels = labels, cv = c(cv, gs_cv)))
}
# Read train result
for (i in 1:5){
  train = read.table(paste(prefix,folder[1], 'train_eval_result.txt', sep = ''))$V1
  gs_train = read.table(paste(prefix,folder[1], 'optimized_train_result.txt', sep = ''))$V1
  assign(paste('df_set',i,sep = ''), data.frame(get(paste('df_set',i,sep='')), train = c(train,gs_train)))
}

# Read test result
for (i in 1:5){
  train = read.table(paste(prefix,folder[1], 'test_eval_result.txt', sep = ''))$V1
  gs_train = read.table(paste(prefix,folder[1], 'optimized_test_result.txt', sep = ''))$V1
  assign(paste('df_set',i,sep = ''), data.frame(get(paste('df_set',i,sep='')), test = c(train,gs_train)))
}

df_everything = data.frame(df_set1, 
                           df_set2[,c('cv','train','test')], 
                           df_set3[,c('cv','train','test')], 
                           df_set4[,c('cv','train','test')], 
                           df_set5[,c('cv','train','test')])

df_mean = aggregate(df_everything[,-1], by = list(df_everything$labels), mean)
colnames(df_mean)[1] = 'labels'
names = colnames(df_mean)
# reshape mean data frame
bm25 = rep(c(rep('regular', 5), rep('grid-searched', 5)), 3)
evalset = c(rep('cv',10), rep('train', 10), rep('test', 10))
set = rep(rep(1:5, 2),3)
mean = c(t(df_mean[df_mean$labels == 'regular', grep('cv*', names)]), t(df_mean[df_mean$labels == 'grid-search', grep('cv*', names)]),
         t(df_mean[df_mean$labels == 'regular', grep('*train*', names)]), t(df_mean[df_mean$labels == 'grid-search', grep('*train*', names)]),
         t(df_mean[df_mean$labels == 'regular', grep('*test*', names)]), t(df_mean[df_mean$labels == 'grid-search', grep('*test*', names)]))

df_mean_reshaped = data.frame(mean = mean, bm25 = bm25, evalset = evalset, set = factor(set))


require(ggplot2)
# Compare mean NDCG
# Absolute NDCG for both GS and Regular BM25
ggplot(data = df_mean_reshaped, aes(x = set, y = mean, color = evalset, shape = evalset)) + 
  geom_point() + 
  facet_wrap('bm25') + 
  xlab('Set') + 
  ylab('Mean NDCG') +
  ggtitle('Mean NDCG w/ Different Evaluation Strategies') + 
  theme_bw()

# Analyze per query accuracy
# make df with one column specifying 
names = colnames(df_everything)
cv_NDCG = c()
bm25 = c()
sets = c()
qIDs = c()
rel_docs = c()
set = 1
for(i in grep('cv*', names)) {
  cv_NDCG = append(cv_NDCG, df_everything[,i])
  bm25 = append(bm25, as.character(df_everything[,c('labels')]))
  sets = append(sets, rep(set, 200))
  qIDs = append(qIDs, rep(1:100, 2))
  # Get num of relevant documents for each query
  rel_docs = append(rel_docs, rep(read.table(paste(prefix, folder[set], 'fold_avgnum_rel_docs.txt', sep = ''))$V1, 2))
  set = set + 1
}
set = 1
train_NDCG = c()
for(i in grep('train*', names)) {
  train_NDCG = append(train_NDCG, df_everything[,i])
  bm25 = append(bm25, as.character(df_everything[,c('labels')]))
  sets = append(sets, rep(set, 200))
  qIDs = append(qIDs, rep(1:100, 2))
  # Get num of relevant documents for each query
  rel_docs = append(rel_docs, rep(read.table(paste(prefix, folder[set], 'set_avgnum_rel_docs.txt', sep = ''))$V1, 2))
  set = set + 1
}
set = 1
test_NDCG = c()
for(i in grep('test*', names)) {
  test_NDCG = append(test_NDCG, df_everything[,i])
  bm25 = append(bm25, as.character(df_everything[,c('labels')]))
  sets = append(sets, rep(set, 200))
  qIDs = append(qIDs, rep(1:100, 2))
  # Get num of relevant documents for each query
  rel_docs = append(rel_docs, rep(read.table(paste(prefix, folder[set], 'test_avgnum_rel_docs.txt', sep = ''))$V1, 2))
  set = set + 1
}

# Special reference number of relevant docs from the test set
test_rel_docs = read.table(paste(prefix, folder[1], 'test_avgnum_rel_docs.txt', sep = ''))$V1

NDCG = c(cv_NDCG, train_NDCG, test_NDCG)
evalset = c(rep('cv', 1000), rep('train', 1000), rep('test', 1000))
df_everything_reshaped = data.frame(bm25 = bm25, set = sets, evalset = evalset, qID = qIDs, rel_docs = rel_docs, NDCG = NDCG)

divide <- function(x, y){
  result = c()
  for (i in 1:length(x)) {
    if (y[i] > 0) {
      result = append(result, x[i]/y[i])
    } else {
      result = append(result, 0)
    }
  }
  return(result)
}

# Analyze regular BM25 first
df_reg = df_everything_reshaped[df_everything_reshaped$bm25 == 'regular',]
NDCG_diff = c(df_reg[df_reg$evalset == 'test', c('NDCG')] - df_reg[df_reg$evalset == 'cv', c('NDCG')], 
              df_reg[df_reg$evalset == 'test', c('NDCG')] - df_reg[df_reg$evalset == 'train', c('NDCG')])
evalset = c(as.character(df_reg[df_reg$evalset == 'cv', c('evalset')]), as.character(df_reg[df_reg$evalset == 'train', c('evalset')]))
set = c(df_reg[df_reg$evalset == 'cv', c('set')], df_reg[df_reg$evalset == 'train', c('set')])
qID = c(df_reg[df_reg$evalset == 'cv', c('qID')], df_reg[df_reg$evalset == 'train', c('qID')])
rel_docs = c(df_reg[df_reg$evalset == 'cv', c('rel_docs')], df_reg[df_reg$evalset == 'train', c('rel_docs')])
t_rel_docs = rep(test_rel_docs, length(rel_docs)/length(test_rel_docs))
rel_docs_pro10 = sapply(rel_docs, FUN = {function(x) min(x, 10)/10})
rel_docs_pro = divide(rel_docs, t_rel_docs)
df_reg_diff = data.frame(evalset = evalset, set = set, qID = qID, NDCG_diff = NDCG_diff, 
                         rel_docs = rel_docs, test_rel_docs = t_rel_docs, rel_docs_pro10 = rel_docs_pro10, rel_docs_pro = rel_docs_pro)

# Regular BM25, Train with test set  rel number of docs > 0
ggplot(data = df_reg_diff[df_reg_diff$evalset == 'train' & df_reg_diff$test_rel_docs > 0,], 
       aes(x = qID, color = rel_docs_pro10, shape = evalset)) + 
  geom_point(aes(y = NDCG_diff)) + 
  facet_wrap('set') + 
  xlab('Query ID') + 
  ylab('NDCG Difference per Query') +
  ggtitle('Per Query NDCG Difference (Test - Train) using Regular BM25') + 
  scale_color_gradient(low = "black", high = "red") +
  theme_bw()

# Regular BM25, CV with test set rel number of docs > 0, colored using avg fold rel num of docs proportion out of 10
ggplot(data = df_reg_diff[df_reg_diff$evalset == 'cv' & df_reg_diff$test_rel_docs > 0,], 
       aes(x = qID, color = rel_docs_pro10, shape = evalset)) + 
  geom_point(aes(y = NDCG_diff)) + 
  facet_wrap('set') + 
  xlab('Query ID') + 
  ylab('NDCG Difference per Query') +
  ggtitle('Per Query NDCG Difference (Test - CV) using Regular BM25') + 
  scale_color_gradient(low = "black", high = "red") +
  theme_bw()

# Analyze GS BM25
df_gs = df_everything_reshaped[df_everything_reshaped$bm25 == 'grid-search',]
NDCG_diff = c(df_gs[df_gs$evalset == 'test', c('NDCG')] - df_gs[df_gs$evalset == 'cv', c('NDCG')], 
              df_gs[df_gs$evalset == 'test', c('NDCG')] - df_gs[df_gs$evalset == 'train', c('NDCG')])
evalset = c(as.character(df_gs[df_gs$evalset == 'cv', c('evalset')]), as.character(df_gs[df_gs$evalset == 'train', c('evalset')]))
set = c(df_gs[df_gs$evalset == 'cv', c('set')], df_gs[df_gs$evalset == 'train', c('set')])
qID = c(df_gs[df_gs$evalset == 'cv', c('qID')], df_gs[df_gs$evalset == 'train', c('qID')])
rel_docs = c(df_gs[df_gs$evalset == 'cv', c('rel_docs')], df_gs[df_gs$evalset == 'train', c('rel_docs')])
t_rel_docs = rep(test_rel_docs, length(rel_docs)/length(test_rel_docs))
rel_docs_pro10 = sapply(rel_docs, FUN = {function(x) min(x, 10)/10})
rel_docs_pro = divide(rel_docs, t_rel_docs)
df_gs_diff = data.frame(evalset = evalset, set = set, qID = qID, NDCG_diff = NDCG_diff, 
                         rel_docs = rel_docs, test_rel_docs = t_rel_docs, rel_docs_pro10 = rel_docs_pro10, rel_docs_pro = rel_docs_pro)

# GS BM25, Train with test set  rel number of docs > 0
ggplot(data = df_gs_diff[df_gs_diff$evalset == 'train' & df_gs_diff$test_rel_docs > 0,], 
       aes(x = qID, color = rel_docs_pro10, shape = evalset)) + 
  geom_point(aes(y = NDCG_diff)) + 
  facet_wrap('set') + 
  xlab('Query ID') + 
  ylab('NDCG Difference per Query') +
  ggtitle('Per Query NDCG Difference (Test - Train) using GS BM25') + 
  scale_color_gradient(low = "black", high = "red") +
  theme_bw()

# GS BM25, CV with test set rel number of docs > 0
ggplot(data = df_gs_diff[df_gs_diff$evalset == 'cv' & df_gs_diff$test_rel_docs > 0,], 
       aes(x = qID, color = rel_docs_pro10, shape = evalset)) + 
  geom_point(aes(y = NDCG_diff)) + 
  facet_wrap('set') + 
  xlab('Query ID') + 
  ylab('NDCG Difference per Query') +
  ggtitle('Per Query NDCG Difference (Test - Train) using GS BM25') + 
  scale_color_gradient(low = "black", high = "red") +
  theme_bw()
