import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EDA():
    def __init__(self,df):
        self.df = df

    def null_sum(self):
        """
        Calculate the sum of null values for each column in the DataFrame.

        Returns:
        --------
        pandas.Series:
            A Series containing the sum of null values for each column.
        """
        return self.df.isnull().sum()
    
    def describe(self):
        """
        Generate descriptive statistics of the dataset.

        Returns:
        --------
        pandas.DataFrame:
            A DataFrame containing summary statistics, including count, mean, std deviation,
            min, 25th percentile, median (50th percentile), 75th percentile, and max for each numeric column.
        """
        return self.df.describe()
        
    def describe_colname(self,colname):
        return self.df[colname].describe()
    
    def fill_na_with_median(self):
        """
        Fill NaN values in the dataframe with the median of respective columns.
        
        Args:
        - dataframe (pd.DataFrame): The input dataframe with NaN values.
        
        Returns:
        - pd.DataFrame: The dataframe with NaN values replaced by the median of respective columns.
        """
        return self.df.fillna(self.df.median(numeric_only=True))
    
    def bar_plot_Matplotlib(self,categories, values, title='Bar Plot', xlabel='', ylabel=''):
        """
        Create a bar plot using Matplotlib.

        Parameters:
        -----------
        categories : list
            List of categories for the x-axis.

        values : list
            List of corresponding values for the y-axis.

        title : str, optional
            Title of the bar plot.

        xlabel : str, optional
            Label for the x-axis.

        ylabel : str, optional
            Label for the y-axis.

        Returns:
        --------
        None
        """
        plt.bar(categories, values)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.xticks(rotation=90)
        plt.ylabel(ylabel)
        plt.show()

    def bar_plot_Seaborn(self, categories, values, title='Bar Plot', xlabel='', ylabel=''):
        """
        Create a bar plot using Seaborn.

        Parameters:
        -----------
        categories : list
            List of categories for the x-axis.

        values : list
            List of corresponding values for the y-axis.

        title : str, optional
            Title of the bar plot.

        xlabel : str, optional
            Label for the x-axis.

        ylabel : str, optional
            Label for the y-axis.

        Returns:
        --------
        None
        """
        sns.barplot(x=categories, y=values)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.xticks(rotation=90)
        plt.ylabel(ylabel)
        plt.show()
   


    def plot_student_data_seaborn(self):
        """
        Function to plot student data statistics.

        Returns:
        - None
        """
        # Set up the figure and axes
        fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))

        # Univariate Analysis for some selected numerical variables
        sns.histplot(self.df['Grad_median'], kde=True, ax=axs[0, 0])
        axs[0, 0].set_title('Distribution of Grad Median Salary')

        sns.histplot(self.df['Nongrad_median'], kde=True, ax=axs[0, 1])
        axs[0, 1].set_title('Distribution of Non-Grad Median Salary')

        sns.histplot(self.df['Grad_unemployment_rate'], kde=True, ax=axs[1, 0])
        axs[1, 0].set_title('Distribution of Grad Unemployment Rate')

        sns.histplot(self.df['Nongrad_unemployment_rate'], kde=True, ax=axs[1, 1])
        axs[1, 1].set_title('Distribution of Non-Grad Unemployment Rate')

        sns.histplot(self.df['Grad_share'], kde=True, ax=axs[2, 0])
        axs[2, 0].set_title('Distribution of Grad Share')

        sns.histplot(self.df['Grad_premium'], kde=True, ax=axs[2, 1])
        axs[2, 1].set_title('Distribution of Grad Premium')

        sns.countplot(y=self.df['Major_category'], ax=axs[3, 0])
        axs[3, 0].set_title('Count of Each Major Category')
        axs[3, 0].set_xlabel('Count')

        # Remove the unused subplot
        axs[3, 1].axis('off')

        plt.tight_layout()
        plt.show()


    def plot_student_data_matplotlib(self):
        """
        Function to plot student data statistics using only matplotlib.

        Returns:
        - None
        """
        fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
        
        # Histograms with density plots
        axs[0, 0].hist(self.df['Grad_median'], bins=30, density=True, alpha=0.7, color='blue', label='Histogram', edgecolor='black')
        axs[0, 0].set_title('Distribution of Grad Median Salary')
        axs[0, 0].set_ylabel('Density')
        
        axs[0, 1].hist(self.df['Nongrad_median'], bins=30, density=True, alpha=0.7, color='green', label='Histogram', edgecolor='black')
        axs[0, 1].set_title('Distribution of Non-Grad Median Salary')
        axs[0, 1].set_ylabel('Density')
        
        axs[1, 0].hist(self.df['Grad_unemployment_rate'], bins=30, density=True, alpha=0.7, color='skyblue', label='Histogram', edgecolor='black')
        axs[1, 0].set_title('Distribution of Grad Unemployment Rate')
        axs[1, 0].set_ylabel('Density')
        
        axs[1, 1].hist(self.df['Nongrad_unemployment_rate'], bins=30, density=True, alpha=0.7, color='red', label='Histogram', edgecolor='black')
        axs[1, 1].set_title('Distribution of Non-Grad Unemployment Rate')
        axs[1, 1].set_ylabel('Density')
        
        axs[2, 0].hist(self.df['Grad_share'], bins=30, density=True, alpha=0.7, color='blue', label='Histogram', edgecolor='black')
        axs[2, 0].set_title('Distribution of Grad Share')
        axs[2, 0].set_ylabel('Density')
        
        axs[2, 1].hist(self.df['Grad_premium'], bins=30, density=True, alpha=0.7, color='grey', label='Histogram', edgecolor='black')
        axs[2, 1].set_title('Distribution of Grad Premium')
        axs[2, 1].set_ylabel('Density')
        
        # Bar plot for Major Categories
        major_categories, counts = np.unique(self.df['Major_category'], return_counts=True)
        axs[3, 0].barh(major_categories, counts, color='blue')
        axs[3, 0].set_title('Count of Each Major Category')
        axs[3, 0].set_xlabel('Count')
        
        # Remove unused subplot
        axs[3, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix_seaborn(self):
        """
        Function to plot a correlation matrix for the numerical variables in the dataframe.
        """
        # Keep only the numeric columns for the correlation matrix
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Correlation matrix for numerical variables
        correlation_matrix = numeric_df.corr()

        # Set up the matplotlib figure
        plt.figure(figsize=(15, 10))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap
        sns.heatmap(correlation_matrix, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})

        plt.title('Correlation Matrix')
        plt.show()

    def plot_correlation_matrix_matplotlib(self):
        """
        Function to plot a correlation matrix for the numerical variables in the dataframe.
        """
        # Keep only the numeric columns for the correlation matrix
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Correlation matrix for numerical variables
        correlation_matrix = numeric_df.corr()

        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(15, 10))

        # Choose a color map, I'll use the colormap that most resembles the seaborn one
        cmap = plt.get_cmap('coolwarm')

        # Create heatmap using imshow
        cax = ax.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1)

        # Create colorbar
        cbar = fig.colorbar(cax, ticks=[-1, -0.5, 0, 0.5, 1])
        cbar.ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])  # vertically oriented colorbar

        # Set ticks with labels
        ax.set_xticks(np.arange(len(correlation_matrix.columns)))
        ax.set_yticks(np.arange(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns)
        ax.set_yticklabels(correlation_matrix.columns)

        # Display values in the heatmap
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, np.around(correlation_matrix.iloc[i, j], decimals=2),
                               ha="center", va="center", color="black")

        plt.title('Correlation Matrix')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


    def scatter_box_plot_matplotlib(self):
        """
        Function to plot student data statistics.
        """
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

        # Scatter plot for Grad Median Salary vs. Non-Grad Median Salary
        axs[0, 0].scatter(self.df['Grad_median'], self.df['Nongrad_median'], color = 'black')
        axs[0, 0].set_title('Grad Median Salary vs. Non-Grad Median Salary')
        axs[0, 0].set_xlabel('Grad Median')
        axs[0, 0].set_ylabel('Non-Grad Median')

        # Scatter plot for Grad Unemployment Rate vs. Non-Grad Unemployment Rate
        axs[0, 1].scatter(self.df['Grad_unemployment_rate'], self.df['Nongrad_unemployment_rate'], color = 'red')
        axs[0, 1].set_title('Grad Unemployment Rate vs. Non-Grad Unemployment Rate')
        axs[0, 1].set_xlabel('Grad Unemployment Rate')
        axs[0, 1].set_ylabel('Non-Grad Unemployment Rate')

        # Scatter plot for Grad Share vs. Grad Premium
        axs[1, 0].scatter(self.df['Grad_share'], self.df['Grad_premium'], color = 'red')
        axs[1, 0].set_title('Grad Share vs. Grad Premium')
        axs[1, 0].set_xlabel('Grad Share')
        axs[1, 0].set_ylabel('Grad Premium')

        # Boxplot for Major Category vs. Grad Median Salary
        categories = self.df['Major_category'].unique()
        data_by_category = [self.df['Grad_median'][self.df['Major_category'] == cat].values for cat in categories]
        axs[1, 1].boxplot(data_by_category, vert=False, patch_artist=True)
        axs[1, 1].set_yticklabels(categories)
        axs[1, 1].set_title('Major Category vs. Grad Median Salary')
        axs[1, 1].set_xlabel('Grad Median')
        axs[1, 1].set_ylabel('Major Category')

        plt.tight_layout()
        plt.show()


    def scatter_box_plot_seaborn(self):
        """
        Function to plot student data statistics.
        """
        # Set up the figure and axes
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

        # Scatter plot for Grad Median Salary vs. Non-Grad Median Salary
        sns.scatterplot(data=self.df, x='Grad_median', y='Nongrad_median', ax=axs[0, 0], color = 'green')
        axs[0, 0].set_title('Grad Median Salary vs. Non-Grad Median Salary')

        # Scatter plot for Grad Unemployment Rate vs. Non-Grad Unemployment Rate
        sns.scatterplot(data=self.df, x='Grad_unemployment_rate', y='Nongrad_unemployment_rate', ax=axs[0, 1], color = 'brown')
        axs[0, 1].set_title('Grad Unemployment Rate vs. Non-Grad Unemployment Rate')

        # Scatter plot for Grad Share vs. Grad Premium
        sns.scatterplot(data=self.df, x='Grad_share', y='Grad_premium', ax=axs[1, 0])
        axs[1, 0].set_title('Grad Share vs. Grad Premium')

        # Boxplot for Major Category vs. Grad Median Salary
        sns.boxplot(data=self.df, x='Grad_median', y='Major_category', ax=axs[1, 1], showfliers=False)
        axs[1, 1].set_title('Major Category vs. Grad Median Salary')

        plt.tight_layout()
        plt.show()


    



    




