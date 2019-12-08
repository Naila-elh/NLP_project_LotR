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


# Apply word_to_vec model
start_time <- Sys.time()
model <- word2vec("doc_no_stopwords.txt")
end_time <- Sys.time()
end_time - start_time

# Exploration: words similarity
model %>% closest_to("gollum", n=20)
model %>% closest_to("frodo")
model %>% closest_to("orc")
model %>% closest_to("orcs")
model %>% closest_to("war")




######     PROJECTIONS    ######

# Récuperer les vecteurs de "ring" et "precious"
ring <- model[[c("ring","sauron"),average=F]]
# Récupérer les 3000 mots les plus fréquents et calculer
# la similarité avec "computer" et "internet"
ring_and_precious <- model[1:500,] %>% cosineSimilarity(ring)
head(ring_and_precious)
# Filtrer en choisissant les 20 mots les plus similaires.
ring_and_precious <- ring_and_precious[
  rank(-ring_and_precious[,1])<20 |
    rank(-ring_and_precious[,2])<20,
  ]
plot(ring_and_precious,type='n')
text(ring_and_precious,labels=rownames(ring_and_precious),cex = 0.7)


#################  CLUSTERING  #################
set.seed(10)
centers = 5
clustering = kmeans(model[1:200,],centers=centers,iter.max = 40)
for (i in 1:5){
  print(paste("cluster", i))
  print(names(clustering$cluster[clustering$cluster==i]))
}



#################   REDUCTION DE DIMENSION ##############
plot(model[1:200,], perplexity=10)
plot(model[1:200,], perplexity=50)



############################################################

################    CLASSIFICATION    #####################

##########################################################

features=data.frame(model)
features$words=rownames(features)
rownames(features) = NULL
