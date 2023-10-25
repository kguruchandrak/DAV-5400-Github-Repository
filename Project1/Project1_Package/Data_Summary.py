class data_summary():
    def __init__(self,df):
        self.df = df

    def data_sum(self):
        """
        Print a summary of the dataset.

        This method provides information about the dataset, including the data source,
        common attributes for graduates and non-graduates, data types of each attribute,
        and the summary of the DataFrame's data types.
        """
        print('''The data has been acquired from https://github.com/fivethirtyeight/data/tree/master/college-majors''')
        print("Data set provide 7 attributes in common for Graduates and Non-Graduates")
        print("Major_category and Major are of type String and rest of them are float and int type")
        print("----------------------------------------------------")
        print(self.df.dtypes)
        return 
    
    