import seaborn as sns
import matplotlib.pyplot as plt

def plot_countplots_y(df):
    # Get the column name of the DataFrame
    column_name = df.columns[0]
    
    # Create a countplot for the column as the y-axis
    ax = sns.countplot(y=column_name, data=df, hue=column_name, palette='Set2')
    ax.set_title(f'Countplot Launch Number Distribution of {column_name}')
    ax.set_xlabel('Launch Number')
    ax.set_ylabel(column_name)
    return ax

def plot_countplots_x(df):
    # Get the column name of the DataFrame
    column_name = df.columns[0]
    
    # Create a countplot for the column as the x-axis
    ax = sns.countplot(x=column_name, data=df, hue=column_name, palette='Set2')
    ax.set_title(f'Countplot Launch Number Distribution of {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Launch Number')
    return ax

def plot_countplots_y_ordered(df):
    # Get the column name of the DataFrame
    column_name = df.columns[0]
    
    # Create a countplot for the column as the y-axis
    ax = sns.countplot(y=column_name, data=df, order=df[column_name].value_counts().index, hue=column_name, palette='Set2')
    ax.set_title(f'Countplot Launch Number Distribution of {column_name}')
    ax.set_xlabel('Launch Number')
    ax.set_ylabel(column_name)
    return ax

def plot_countplots_x_ordered(df):
    # Get the column name of the DataFrame
    column_name = df.columns[0]
    
    # Create a countplot for the column as the x-axis
    ax = sns.countplot(x=column_name, data=df, order=df[column_name].value_counts().index, hue=column_name, palette='Set2')
    ax.set_title(f'Countplot Launch Number Distribution of {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Launch Number')
    return ax

def plot_bar(success_percent, failure_percent, column_name):
    # Create a figure and an axes object
    _, ax = plt.subplots()  

    # Create barplot for success and failure percentage
    sns.barplot(x=success_percent.index, y=success_percent.values, label='Success', alpha=0.6, ax=ax)
    sns.barplot(x=failure_percent.index, y=failure_percent.values, label='Failure', alpha=0.6, ax=ax)
    ax.set_title(f'Barplot Success and Failure Percentages of {column_name}')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel(column_name)
    
    # Set x-ticks and x-tick labels
    ax.set_xticks(range(len(success_percent.index)))  # Ensure the number of ticks match the number of labels
    ax.set_xticklabels(success_percent.index, rotation=90)  # Rotate x-tick labels for clarity
    ax.legend()  
    return ax 