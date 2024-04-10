def process_text_column(column):

    return


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv('../../data/raw/global_space_launches.csv')
    df_new = pd.DataFrame(df[['Company Name','Location']])
    df_new['comma count'] = df_new['Location'].str.count(',')
    df_new['comma count'].value_counts()
    
    df_new[df_new['comma count']==1]
    
    
    comma_count = df_new['Location'].str.count(',')
    unique_count_array = comma_count.unique()
    test_array = []
    for _ , each_number in enumerate(unique_count_array): 
        test_array.append(df_new[df_new['comma count']==each_number])
    
    
    test_array_header = test_array(:,0)
    df_test = pd.DataFrame(test_array, columns = test_array[:,0])

