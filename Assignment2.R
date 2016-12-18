# Text classification script for app store reviews
# Chang Hong Jie A0140154W
#load libraries
library("e1071")
library("RTextTools")
library("stringr")
library("tm")
library("plyr")
library("SnowballC")
library("maxent")
library("caret")

#stopwords some more
myStopwords <- unlist(read.table("http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a11-smart-stop-list/english.stop", header = FALSE))
all_class <- c("education","finance","social","game","finance")

#Helper functions
predictTest <- function(test_text, mat, classifier){
  train_mat = mat[1:2,] 
  train_mat[,1:ncol(train_mat)] = 0
  test_matrix = create_matrix(test_text, language="english", minWordLength=1, removeStopwords=F, removeNumbers=F, toLower=F, removePunctuation=F, stripWhitespace=T)
  test_mat <- as.matrix(test_matrix)
  for(col in colnames(test_mat)){ 
    if(col %in% colnames(train_mat)) {
      train_mat[2,col] = test_mat[1,col]; 
    }
  }
  #test_mat = as.matrix(t(test_mat))
  row.names(train_mat)[1] = ""
  row.names(train_mat)[2] = test_text
  p <- predict(classifier, train_mat[1:2,])
  rslt <- as.character(p[2])
}

combine.csv <- function(str){
  path <- paste("X:/Year 2 Sem 1/BT2101/Assignments/training_data",str,sep = '/')
  setwd(path)
  files <- dir(getwd())
  combined.csv <- do.call(rbind,lapply(files,read.csv))
}

clean.data <- function(data) {
  data <- data[!(is.na(data$review) | data$review==""), ]
  data$review <- tolower(data$review)
  data$review <- removeWords(data$review, myStopwords)
  punct <- '[]\\?!\"#$%&(){}+*/:;,._`|~\\[<=>@\\^-]'
  data$review <- removePunctuation(data$review)
  data$review <- gsub(punct, "", data$review)
  data$review <- removeNumbers(data$review)
  data$review <- stripWhitespace(data$review)
  data$review <- stemDocument(data$review)
  data$review <- iconv(data$review, "latin1", "ASCII", sub="")
  data$review <- removeWords(data$review, c("app", "love", "great", "good"))
  #data$review <- stripWhitespace(data$review)
  data <- data[!(is.na(data$review) | data$review=="" | data$review==" "), ]
  data<- as.matrix(data$review)
}

clean.app <- function(data) {
  data <- data[!(is.na(data$description) | data$description==""), ]
  data$description <- tolower(data$description)
  data$description <- removeWords(data$description, myStopwords)
  punct <- '[]\\?!\"#$%&(){}+*/:;,._`|~\\[<=>@\\^-]'
  data$description <- removePunctuation(data$description)
  data$description <- gsub(punct, "", data$description)
  data$description <- removeNumbers(data$description)
  data$description <- stripWhitespace(data$description)
  data$description <- stemDocument(data$description)
  data$description <- iconv(data$description, "latin1", "ASCII", sub="")
  data$description <- removeWords(data$description, c("app", "love", "great", "good"))
  #data$description <- stripWhitespace(data$description)
  data <- data[!(is.na(data$description) | data$description=="" | data$description==" "), ]
  data<- as.matrix(data$description)
}

generate.sample <- function(data, dataSize) {
  data.sample <- as.matrix(data[sample(nrow(data),dataSize),])
}

#read in csvs
education = combine.csv('education/')
finance = combine.csv('finance/')
game = combine.csv('game/')
social = combine.csv('social/')
weather = combine.csv('weather/')

#read in app descriptions
edu.app = read.csv("./app_description/education.csv", header = T)
fin.app = read.csv("./app_description/finance.csv", header = T)
game.app = read.csv("./app_description/game.csv", header = T)
soc.app = read.csv("./app_description/social.csv", header = T)
wea.app = read.csv("./app_description/weather.csv", header = T)

#preprocessing
#clean, trim and generate sample data to train model
education.clean <- as.matrix(rbind(clean.data(education),clean.app(edu.app)))
education.clean <- as.matrix(education.clean[nchar(education.clean)>150])
finance.clean <- as.matrix(rbind(clean.data(finance),clean.app(fin.app)))
finance.clean <- as.matrix(finance.clean[nchar(finance.clean)>150])
social.clean <- as.matrix(rbind(clean.data(social), clean.app(soc.app)))
social.clean <- as.matrix(social.clean[nchar(social.clean)>150])
game.clean <- as.matrix(rbind(clean.data(game), clean.app(game.app)))
game.clean <- as.matrix(game.clean[nchar(game.clean)>150])
weather.clean <- as.matrix(rbind(clean.data(weather), clean.app(wea.app)))
weather.clean <- as.matrix(weather.clean[nchar(weather.clean)>150])

#sampling
dataSize <- floor(0.8*nrow(weather.clean))
set.seed(1)

education.trim <- generate.sample(education.clean,dataSize)
finance.trim <- generate.sample(finance.clean,dataSize)
game.trim <- generate.sample(game.clean,dataSize)
social.trim <- generate.sample(social.clean,dataSize)
weather.trim <- generate.sample(weather.clean,dataSize)

#partition train and test data
train.size <- floor(0.75*dataSize)
train.inx <- sample(dataSize,size = train.size)

education.train <- education.trim[train.inx,]
education.test <- education.trim[-train.inx,]

finance.train <- finance.trim[train.inx,]
finance.test <- finance.trim[-train.inx,]

game.train <- game.trim[train.inx,]
game.test <- game.trim[-train.inx,]

weather.train <- weather.trim[train.inx,]
weather.test <- weather.trim[-train.inx,]

social.train <- social.trim[train.inx,]
social.test <- social.trim[-train.inx,]

train.data = data.frame(text = c(education.train, finance.train, game.train, social.train, weather.train), class = c(rep("education", length(education.train)), rep("finance", length(finance.train)), rep("game", length(game.train)), rep("social", length(social.train)), rep("weather", length(weather.train))))
test.data <- data.frame(text = c(education.test, finance.test, game.test, social.test, weather.test), class = c(rep("education", length(education.test)), rep("finance", length(finance.test)), rep("game", length(game.test)), rep("social", length(social.test)), rep("weather", length(weather.test))))
df <- rbind(train.data, test.data)

#train + evaluation using matrix
m <- create_matrix(df$text, removePunctuation = F, removeStopwords = F, stripWhitespace = F, toLower = F, ngramLength = 2, minWordLength = 5)

#find most prominent features to check for possible stopwords
freq <- as.matrix(findFreqTerms(m, 50))
common.features = data.frame(term = freq, value = tm_term_score(m, freq, FUN = slam::col_sums))

#create RTextTools container for training and main analysis
container.train <- create_container(m,
                              labels =  as.numeric(factor(df$class)),
                              trainSize = 1:nrow(train.data),
                              testSize = 11236:14980,
                              virgin = F)
#training and results
models <- train_models(container = container.train, algorithm =  c("MAXENT", "SVM", "GLMNET"))
results <- classify_models(container.train, models)
analytics <- create_analytics(container.train, results)
s <- create_precisionRecallSummary(container.train, results)
ensemble <- create_ensembleSummary(analytics@document_summary)

#k-fold CV
cv = cross_validate(container.train, 5)
