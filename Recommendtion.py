import pandas as pd
#import dataset

anime = pd.read_csv(r"C:\Users\anilk\OneDrive\Desktop\assigiment of data science\assignment-8\game.csv")
anime.shape
anime.columns
anime.game
 
from sklearn.feature_extraction.text import TfidfVectorizer
#creating Tfidf Vectorizerto remove all the stop words
Tfidf = TfidfVectorizer(stop_words=('english'))
## repacling the NAN values in overviewscolumn with empty string

anime['game'].isnull().sum()
anime['game'] = anime['game'].fillna('')

## preparing the Tfidf matrix byfittingand transforming
Tfidf_matrix = Tfidf.fit_transform(anime.game)
Tfidf_matrix.shape

## cosine(x,y) = (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel
# computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(Tfidf_matrix,Tfidf_matrix)

##creating a mapping of anime name to index number
anime_index = pd.Series(anime.index, index = anime['game']).drop_duplicates()
anime_id = anime_index['Metroid Prime']
anime_id 
def get_recommendations(game,topN):
    #topN = 10
    ## getting the game index using its title
    anime_id = anime_index[game]
    cosine_scores = list(enumerate((cosine_sim_matrix[anime_id])))
    ##Sorting the cosine_similartiy scores based on scores
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1],reverse=True)
    ## Getting the scores top N most similar games
    cosine_scores_N = cosine_scores[0:topN+1] 
    # Getting the geme index
    anime_idx = [i[0] for i in cosine_scores_N]
    anime_scores = [i[1] for i in cosine_scores_N]
    
    # similar movies and scores
    anime_similar_show = pd.DataFrame(columns=['game,Score'])
    anime_similar_show['game'] = anime.loc[anime_idx,'game']
    anime_similar_show['score'] = anime_scores
    anime_similar_show.reset_index(inplace=True)
    print(anime_similar_show)
    return(anime_similar_show)

## enter your entertainment and number to be recommended
get_recommendations('Super Mario Galaxy',topN = 10)
anime_index['Grand Theft Auto IV']


## ENTERTAINMENT ##
import pandas as pd

data = pd.read_csv(r"C:\Users\anilk\OneDrive\Desktop\assigiment of data science\assignment-8\Entertainment.csv")
data.shape
data.columns
data.Titles

from sklearn.feature_extraction.text import TfidfVectorizer

## cerating Tfidf vectorizer to remove all stop words
Tfidf = TfidfVectorizer(stop_words="english")

## Repalcing NAN values in overview columns with empty string
data["Titles"].isnull().sum()
data["Titles"] = data["Titles"].fillna('')

## preparing the Tfidf matrix by fitting transfroming
Tfidf_matrix = Tfidf.fit_transform(data.Titles)
Tfidf_matrix.shape

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

## Computing the cosine similarity Tfidf matrix

cosine_sim_matrix = linear_kernel(Tfidf_matrix, Tfidf_matrix)
## cerating mapping of data name to index number
data_index = pd.Series(data.index, index = data['Titles']).drop_duplicates()

data_id = data_index["Assassins (1995)"]
data_id

def get_recommendations(Titles, topN):
    
    # topN = 10
    # Getting the movie index using its title 
    data_id = data_index[Titles]
    
    # Getting the pair wise similarity score for all the Titles using the cosine based similarities
    cosine_scores = list(enumerate(cosine_sim_matrix[data_id]))
    cosine_scores = sorted(cosine_scores, key= lambda x:x[1], reverse= True)
    
    # We get the scores of top N most similar movies
    cosine_scores_N = cosine_scores[0:topN+1]
    
    # Getting the movie index 
    data_idx = [i[0] for i in cosine_scores_N]
    data_scores = [i[1] for i in cosine_scores_N]
    
    data_similar = pd.DataFrame(columns = ["Titles","Scores"])
    data_similar["Titles"] = data.loc[data_idx, "Titles"]
    data_similar["Scores"] = data_scores
    data_similar.reset_index(inplace = True)
    
    print(data_similar)
    
## Enter your Entertaniment and number to be recommened
get_recommendations('Toy Story (1995)', topN=10)
data_index['Jumanji (1995)']
                      



