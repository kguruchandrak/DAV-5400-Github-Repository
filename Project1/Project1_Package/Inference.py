import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class inference:
    def __init__(self,df) :
        self.df = df

    def question1_matplotlib(self):
        """
        Plots the distribution of median salaries for graduates across all majors.
        
        Returns:
        - None. Displays a plot.
        """
        plt.figure(figsize=(14, 10))
        sorted_majors = self.df.sort_values(by='Grad_median', ascending=False)
        sns.barplot(data=sorted_majors.head(55), y='Major', x='Grad_median', palette='viridis')
        plt.title('Distribution of Median Salaries for Graduates Across All Majors')
        plt.xlabel('Median Salary ($)', fontsize = 14)
        plt.yticks(fontsize = 10)
        plt.ylabel('Major', fontsize = 14)
        plt.tight_layout()
        plt.show()

    def question1_seaborn(self):
        """
        Plots the distribution of median salaries for graduates across all majors using Seaborn.
        
        Returns:
        - None. Displays a plot.
        """
        plt.figure(figsize=(12, 8))
        sorted_majors = self.df.sort_values(by='Grad_median', ascending=False)
        sns.barplot(data=sorted_majors.head(55), y='Major', x='Grad_median', palette='viridis')
        plt.title('Distribution of Median Salaries for Graduates Across All Majors')
        plt.xlabel('Median Salary ($)')
        plt.ylabel('Major')
        plt.tight_layout()
        plt.show()


    def question2_matplotlib(self):
        """
        Uses Matplotlib to plot the relationship between the proportion of graduates in a major 
        and their earnings premium over non-graduates.
        """
        plt.figure(figsize=(12, 8))
        
        # Create a color map based on the unique values in the 'Major_category' column
        colors = plt.cm.tab20(range(len(self.df['Major_category'].unique())))
        color_dict = dict(zip(self.df['Major_category'].unique(), colors))
        
        for category in self.df['Major_category'].unique():
            subset = self.df[self.df['Major_category'] == category]
            plt.scatter(subset['Grad_share'], subset['Grad_premium'], label=category, color=color_dict[category])
        
        plt.title('Grad Share vs. Grad Premium by Major Category (Matplotlib)')
        plt.xlabel('Proportion of Graduates')
        plt.ylabel('Earnings Premium of Graduates')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()


    def question2_seaborn(self):
        """
        Plots the relationship between the proportion of graduates in a major 
        and their earnings premium over non-graduates.
        
        Parameters:
            None.
        
        Returns:
            None. Displays a plot.
        """
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=self.df, x='Grad_share', y='Grad_premium', hue='Major_category', palette='tab20')
        plt.title('Grad Share vs. Grad Premium by Major Category')
        plt.xlabel('Proportion of Graduates')
        plt.ylabel('Earnings Premium of Graduates')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()


    def question3_matplotlib(self):
        """
        Uses Matplotlib to plot the relationship between the number of students in a major 
        and the employment rate for graduates.
        """
        plt.figure(figsize=(12, 8))
        
        # Create a color map based on the unique values in the 'Major_category' column
        colors = plt.cm.tab20(range(len(self.df['Major_category'].unique())))
        color_dict = dict(zip(self.df['Major_category'].unique(), colors))
        
        for category in self.df['Major_category'].unique():
            subset = self.df[self.df['Major_category'] == category]
            plt.scatter(subset['Grad_total'], subset['Grad_unemployment_rate'], label=category, color=color_dict[category])
        
        plt.title('Total Students vs. Graduate Unemployment Rate by Major Category (Matplotlib)')
        plt.xlabel('Total Students')
        plt.ylabel('Graduate Unemployment Rate')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    def question3_seaborn(self):
        """
        Uses Seaborn to plot the relationship between the number of students in a major 
        and the unemployment rate for graduates.
        """
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=self.df, x='Grad_total', y='Grad_unemployment_rate', hue='Major_category', palette='tab20')
        plt.title('Total Students vs. Graduate Unemployment Rate by Major Category (Seaborn)')
        plt.xlabel('Total Students')
        plt.ylabel('Graduate Unemployment Rate')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()


    

    
