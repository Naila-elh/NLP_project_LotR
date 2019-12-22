#########################################################

##############      0. ENVIRONNEMENT     ################

#########################################################

setwd("C:/Users/naila/Documents/ENSAE/Cours/3A/Apprentissage statistique appliqué/TP3")
file.create("lord-of-the-ring-data")

#Packages
library(devtools)
library(httr)
library(tm)
library(stopwords)
library(tsne)
library(rpart)
library(class)


#Proxy
set_config(use_proxy(url="proxy.bloomberg.com", port=80))
set_config( config( ssl_verifypeer = 0L ) )
#Install and charge word2vec
install_github("bmschmidt/wordVectors", INSTALL_opts = c('--no-lock'))
library(wordVectors)


#Word2vec function : train a word2vec from file.txt or charge an existing
#model from file.bin
word2vec <- function(fileName) {
  if (grepl('.txt', fileName, fixed=T)) {
    # Convert test.txt to test.bin.
    binaryFileName <- gsub('.txt', '.bin', fileName, fixed=T)
  }
  else {
    binaryFileName <- paste0(fileName, '.bin')
  }
  # Train word2vec model.
  if (!file.exists(binaryFileName)) {
    # Lowercase and setup ngrams.
    prepFileName <- 'temp.prep'
    prep_word2vec(origin=fileName, destination=prepFileName,
                  lowercase=T, bundle_ngrams=2)
    # Train word2vec model.
    model <- train_word2vec(prepFileName, binaryFileName,
                            vectors=200, threads=4, window=12, iter=5, negative_samples=0)
    # Cleanup.
    unlink(prepFileName)
  } else {
    model <- read.vectors(binaryFileName)
  }
  model
}


#########################################################

##############      1. DATA PROCESSING     ################

#########################################################

# Import .txt file
doc <- readChar('data_txt.txt', file.info('data_txt.txt')$size)

# Lowercase
doc <- tolower(doc)

# Drop stopwords
stopwords_regex <- paste(stopwords::stopwords(language='en', source='snowball'), collapse = '\\b|\\b')
stopwords_regex <- paste0('\\b', stopwords_regex, '\\b')
stopwords_regex2 <- paste(tm::stopwords('en'), collapse = '\\b|\\b')
stopwords_regex2 <- paste0('\\b', stopwords_regex2, '\\b')
other_stopwords <- paste0('\\b', paste(c("s", "ve", "le", "o", "ll", "d", "'s>", "m", "re"), collapse = '\\b|\\b'), '\\b')
doc <- stringr::str_replace_all(doc, stopwords_regex2, '')
doc <- stringr::str_replace_all(doc, stopwords_regex, '')
doc <- stringr::str_replace_all(doc, other_stopwords, '')
doc <- removePunctuation(doc)

# Save doc without stopwords in a .txt file
cat(doc, file="doc_no_stopwords.txt",sep="\n")


#########################################################

##############      2. NLP ALGORITHMS     ################

#########################################################
set.seed(10)
# Apply word_to_vec model
start_time <- Sys.time()
model <- word2vec("doc_no_stopwords.txt")
end_time <- Sys.time()
end_time - start_time

##########      EXPLORATION      ##########

# 100 most frequent words
rownames(model)
rownames(model[1:200,])

# Words similarity
    # characters
model %>% closest_to("gandalf")
model %>% closest_to("saruman")
model %>% closest_to("smeagol")
model %>% closest_to("gollum")
model %>% closest_to("frodo")
model %>% closest_to("aragorn")
model %>% closest_to("sauron")
model %>% closest_to("precious")

    # other words
model %>% closest_to("gondor")
model %>% closest_to("mordor")
model %>% closest_to("orc")
model %>% closest_to("orcs")
model %>% closest_to("hobbit")
model %>% closest_to("hobbits")
model %>% closest_to("war")



##########      PROJECTIONS      ##########

# Vectors of "smeagol" and "gollum"
smeagol <- model[[c("smeagol","gollum"),average=F]]
# Compute similarity with "smeagol" and "gollum" with 200 most frequent words
smeagol_and_gollum <- model[1:200,] %>% cosineSimilarity(smeagol)
head(smeagol_and_gollum)
# Filtre with 20 most similar words
smeagol_and_gollum <- smeagol_and_gollum[
  rank(-smeagol_and_gollum[,1])<20 |
    rank(-smeagol_and_gollum[,2])<20,
  ]
plot(smeagol_and_gollum,type='n')
text(smeagol_and_gollum,labels=rownames(smeagol_and_gollum),cex = 0.7)



##########     REDUCTION OF DIMENSION WITH T-SNE      ##########
#plot(model[1:100,], perplexity=10)
plot(model[1:200,], perplexity=20)


##########      CLUSTERING      ##########
set.seed(10)
centers = 6
clustering = kmeans(model[1:200,],centers=centers,iter.max = 40)
for (i in 1:6){
  print(paste("cluster", i))
  print(names(clustering$cluster[clustering$cluster==i]))
}






############################################################

#######    3. MACHINE LEARNING : CLASSIFICATION    #######

##########################################################

# Prediction of the clusters predicted before


#######     PROCESSING     #####

# Convert model into dataframe
df=data.frame(model)
df$words=rownames(df)
df=df[1:200,] #we only keep the 200 most frequent words
rownames(df) = NULL

# Convert clustering into dataframe
clusters=data.frame(clustering$cluster)
clusters$words=rownames(clusters)
rownames(clusters) = NULL

#merge dataset of features and clusters
df=merge(clusters, df, by.x="words", by.y="words")

######     TRAIN/TEST     ######
set.seed(101) # Set Seed so that same sample can be reproduced in future also
# Now Selecting 75% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(df), size = floor(.75*nrow(df)), replace = F)
train <- df[sample, ]
test  <- df[-sample, ]


#######     KNN     #####

# Choosing optimal cluster
sum_errors=0
for (i in 1:100) {
  fold = sample(rep(1:5,each=30)) # creation des groupes B_v
  cvpred = matrix(NA,nrow=nrow(train),ncol=ncol(train)) # initialisation de la matrice
  # des prédicteurs
  for (k in 1:10)
    for (v in 1:5)
    {
      sample1 = train[which(fold!=v),3:202]
      sample2 = train[which(fold==v), 3:202]
      class1 = train[which(fold!=v),2]
      cvpred[which(fold==v),k] = knn(sample1,sample2,class1,k=k)
    }
  class = as.numeric(train[,2])
  # display misclassification rates for k=1:10
  errors=apply(cvpred,2,function(x) sum(class!=x)) # calcule l'erreur de classif.
  sum_errors=sum_errors+errors
}
mean_errors=sum_errors/100
mean_errors


# Prediction
pred_1 = knn(train[,3:202], test[,3:202], train[,2], k = 8)
pred_2 = knn(train[,3:202], test[,3:202], train[,2], k = 9)

# Errors
# confusion matrices
cm1=as.matrix(table(Actual = test[,2], Predicted = pred_1))
cm2=as.matrix(table(Actual = test[,2], Predicted = pred_2))
cm1
cm2

# accuracy
sum(diag(cm1))/length(test[,2])
sum(diag(cm2))/length(test[,2])

# For which words the cluster was not well predicted?
test$words[test$clustering.cluster!=pred_1]
test$words[test$clustering.cluster!=pred_2]
test$clustering.cluster[test$clustering.cluster!=pred_2]



#######     DECISION TREES     #####

# FIRST MODEL : WITH ONE NODE (we only predict "b")
rt=rpart(clustering.cluster ~ ., data=train[,2:202], method="class") # problem : only one node

# prediction
pred_rt=predict(rt, test[,3:202], type="class")
# confusion matrix
cm_rt=as.matrix(table(Actual = test[,2], Predicted = pred_rt))
cm_rt
# accuracy
sum(diag(cm_rt))/length(test[,2])


# For which words the cluster was not well predicted?
test$words[test$clustering.cluster!=pred_rt]
test$clustering.cluster[test$clustering.cluster!=pred_rt]


#######     RANDOM FOREST     #####
library(randomForest)
# we convert the clusters into factors
train$clustering.cluster<- as.factor(train$clustering.cluster)
test$clustering.cluster<- as.factor(test$clustering.cluster)
rforest = randomForest(clustering.cluster ~ ., data=train[,2:202])

# prediction
pred_rf=predict(rforest, test[,3:202], type="class")
# confusion matrix
cm_rf=as.matrix(table(Actual = test[,2], Predicted = pred_rf))
cm_rf
# accuracy
sum(diag(cm_rf))/length(test[,2])

# For which words the cluster was not well predicted?
test$words[test$clustering.cluster!=pred_rf]
test$clustering.cluster[test$clustering.cluster!=pred_rf]
