###Game#########


import pandas as pd
 
game = pd.read_csv("C:\Users\Dell\Desktop\Assignment101\Recommendatio engine\game.csv", encoding = 'utf8')
game.shape # shape
game.columns
game.game # genre columns

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
game["game"].isnull().sum() 
game["game"] = game["genre"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(game.game)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294, 46



from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of anime name to index number 
game_index = pd.Series(game.index, index = game['game']).drop_duplicates()

game_id = game_index["Hitman"]
game_id

def get_recommendations(Name, topN):    
    game_id = game_index[Name]
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0: topN+1]
    game_idx =  [i[0] for i in cosine_scores_N]
    game_scores =  [i[1] for i in cosine_scores_N]
    

    game_similar_show = pd.DataFrame(columns=["game","Score"])
    game_similar_show["game"] = game.loc[game_idx,"game"]
    game_similar_show["Score"] = game_scores
    game_similar_show.reset_index(inplace = True)  
  
    print (game_similar_show)
    return (game_similar_show)

    

get_recommendations("ICO",topN = 10)
game_index["ICO"]





###Entertainment##############



import pandas as pd

ds = pd.read_csv("C:\Users\Dell\Desktop\Assignment101\Recommendatio engine\Entertainment.csv", encoding = 'utf8')
ds.shape # shape
ds.columns
ds.game # genre columns

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
ds["Titles"].isnull().sum() 

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(ds.Titles)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294, 46



from sklearn.metrics.pairwise import linear_kernel


cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


ds_index = pd.Series(ds.index, index = ds['Titles']).drop_duplicates()

ds_id = ds_index["Powder (1995)"]
ds_id

def get_recommendations(Name, topN):    
    ds_id = ds_index[Name]
    cosine_scores = list(enumerate(cosine_sim_matrix[ds_id]))
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0: topN+1]
    ds_idx =  [i[0] for i in cosine_scores_N]
    ds_scores =  [i[1] for i in cosine_scores_N]
    
    
    ds_similar_show = pd.DataFrame(columns=["Titles","Score"])
    ds_similar_show["Titles"] = ds.loc[ds_idx,"Titles"]
    ds_similar_show["Score"] = ds_scores
    ds_similar_show.reset_index(inplace = True)  
  
    print (ds_similar_show)
    return (ds_similar_show)

    
get_recommendations("Heat (1995)",topN = 10)
ds_index["Sabrina (1995)"]





















































