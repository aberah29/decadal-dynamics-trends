import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#import stats 
from scipy import stats

def read_data(file_paths, selected_countries, start_year, end_year):
    
    
    """
    Read and preprocess data from CSV files.

    Parameters:
    - file_paths (list of str): List of file paths for multiple datasets.
    - selected_countries (list of str): List of countries to include in the analysis.
    - start_year (int): Start year for data extraction.
    - end_year (int): End year for data extraction.

    Returns:
    - dataframes_dict (dict): Dictionary of DataFrames with keys as file names.
    - dataframes_dict_transpose (dict): Dictionary of transposed DataFrames.
    """
     
    # Dictionary to store original DataFrames
    dataframes_dict = {}
    # Dictionary to store transposed DataFrames
    dataframes_dict_transpose = {}
    
    # Columns to exclude
    exclude_columns = ['Country Code', 'Indicator Name', 'Indicator Code']
    
    
    # Iterate over each file path
    for path in file_paths:
        
        # Extract file name without extension
        file_name = path.split('.')[0].replace(' ', '_')
        
        # Load the dataset, skipping the first 4 rows
        df = pd.read_csv(path, skiprows=4, usecols=lambda x: x.strip() != "Unnamed: 67" if x not in ['Indicator Name', 'Indicator Code', 'Country Code'] else True)
        
        # Exclude specified columns
        df = df.drop(columns=exclude_columns, errors='ignore')
        
        # Set 'Country Name' as the index
        df.set_index("Country Name", inplace=True)
        
        # Filter data for selected countries and the specified year range
        df = df.loc[selected_countries, "2012":"2022"]
        
        # Calculate the mean of each row
        df['mean'] = df.mean(axis=1)
        
        # Fill null values in each row with the mean of that row
        df = df.apply(lambda row: row.fillna(row['mean']), axis=1)
        
        # Remove the 'mean' column from the dictionary
        df = df.drop(columns=['mean'])
        
        # Transpose the DataFrame
        df_trans = df.transpose()
        
        df_trans.dropna(axis=0, how="all", inplace=True)
        
        # Reset index to make years a column
        df_trans.reset_index(inplace=True)
        df_trans = df_trans.rename(columns={'index': "Year"})
        
        # Convert 'Year' column to integers
        df_trans['Year'] = df_trans['Year'].astype(int)
        
        # Store DataFrames in dictionaries
        dataframes_dict[file_name] = df
        dataframes_dict_transpose[file_name] = df_trans
        
        
    return dataframes_dict, dataframes_dict_transpose


def create_line_plot(data_frame, title):
    
    
    """
    Create and display a line plot for the countries.

    """
    
    # Set the figure size
    plt.figure(figsize=(12, 6))
    
    
    # Iterate over each country column (excluding the 'Year' column)
    for country in data_frame.columns[1:]:
        plt.plot(data_frame['Year'], data_frame[country], label=country)
        
    
    # Set labels and title
    plt.xlabel('Year')
    plt.ylabel('Values')
    plt.title(f"{title} Over Years")
    
    # Add a grid for better readability
    plt.grid(True)
    
    # Add a legend with the country names
    plt.legend(title='Country', bbox_to_anchor=(1, 1))
    
    # Adjust layout for better visualization
    plt.tight_layout()
    
    # Display the plot
    plt.show()


def create_bar_chart(data_frame, y_label, title):
    
    
    """
    Create and display a bar chart.

    Args:
        data_frame (pd.DataFrame): Data for plotting.
        y_label (str): Label for the y-axis.
        title (str): Title of the chart.
    """
    
    # Set 'Year' as the index
    data_frame.set_index("Year", inplace=True)

    # Plotting bar chart
    data_frame.plot(kind="bar", figsize=(10, 6))
    
    # Set labels and title
    plt.xlabel("Year")
    plt.ylabel(y_label)
    plt.title(title)
    
    # Add legend with DataFrame names
    plt.legend(title="Country", bbox_to_anchor=(1, 1))
    
    # Display the plot
    plt.show()
    
    
def generate_correlation_heatmaps(dataframes_dict_transpose, countries):
    
    
    """
    Generate correlation heatmaps for selected countries.

    Args:
        dataframes_dict_transpose (dict): Dictionary of transposed DataFrames.
        countries (list): List of countries for which to generate heatmaps.
    """
    
    # Concatenate transposed DataFrames into a pivot table
    pivot_table = pd.concat(dataframes_dict_transpose, axis=1)
    
    
    for country in countries:
        
        # Extract data for the current country
        selected_data = pivot_table.loc[:, (slice(None), country)]
        
        # Extract category names
        categories = selected_data.columns.get_level_values(0).unique()
        categories_name = ['Urban population','GDP Growth','CO2 emissions','Energy use']

        # Calculate the correlation matrix
        correlation_matrix = selected_data.corr()
        print(correlation_matrix)
        # Define custom colors for the heatmap
        my_colors = ['#75C8AE', '#FC9871', '#98A9D0', '#E995C9', '#AFDB65','#FFDC44','#E7C99E','#BABABA']
        my_cmap = ListedColormap(my_colors)
        
        # Create a heatmap
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap=my_cmap, linewidths=.5, xticklabels=categories_name, yticklabels= categories_name)
        
        # Set title
        plt.title(f'Correlation Heatmap for {country}')
        
        # Set x-axis and y-axis labels
        plt.xlabel('Indicators')
        plt.ylabel('Indicators')
        
        # Rotate x-axis tick labels
        plt.xticks(rotation=90)
        # Rotate y-axis tick labels
        plt.yticks(rotation=0)
        
        # Customize the color bar
        cbar = heatmap.collections[0].colorbar
        cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        cbar.set_ticklabels(['-1', '-0.75', '-0.5', '-0.25', '0', '0.25', '0.5', '0.75', '1'])
        
        # Display the heatmap
        plt.show()
        
        
def method_stats(dist):
    
    
    """ Prints moments of distribution dist. Uses numpy and scipy.stats"""

    print("skewness:       ", stats.skew(dist))
    print("kurtosis:       ", stats.kurtosis(dist))

    return



def cen_norm(dist):
    """ Centralises and normalises distribution dist. Uses numpy"""
    
    aver = np.average(dist)
    stdev = np.std(dist)
    
    dist = (dist-aver) / stdev
    
    return dist





if __name__ == "__main__":
    file_paths = ['Urban population.csv','GDP Growth.csv','CO2 emissions.csv','Energy use.csv']
    selected_countries = ['Canada', 'China', 'United Kingdom', 'India', 'United States']
    start_year = 2012
    end_year = 2022
    # Call the read function
    dataframe, dataframes_dict_transpose = read_data(file_paths, selected_countries, start_year, end_year)
    #print(dataframes_dict_transpose["Urban_population"])
    
    # Line plots of all countries which are already selected in dataframes
    create_line_plot(dataframes_dict_transpose["Urban_population"], title = "Urban Population")
    create_line_plot(dataframes_dict_transpose["GDP_Growth"], title ="GDP Growth")
    create_line_plot(dataframes_dict_transpose["CO2_emissions"], title ="CO2 emissions")
    create_line_plot(dataframes_dict_transpose["Energy_use"], title ="Energy use")
    
   
    #create_bar_chart(dataframes_dict_transpose, df_names, selected_countries)
    df_names = ['Urban_population','GDP_Growth','CO2_emissions','Energy_use',]
    dataframes = {
        "Urban_population" : dataframes_dict_transpose["Urban_population"],
        "GDP_Growth" : dataframes_dict_transpose["GDP_Growth"],
        "CO2_emissions" : dataframes_dict_transpose["CO2_emissions"],
        "Energy_use" : dataframes_dict_transpose["Energy_use"]
        
    }
    
    
    for df_name, df in dataframes.items():
        y_label = df_name.replace("_", " ").title()  # Create a y-axis label from DataFrame name
        title = f"{y_label} Over Years"
        create_bar_chart(df, y_label, title)
    
    #Heatmap
    generate_correlation_heatmaps(dataframes_dict_transpose, selected_countries)

    
    #for statistical approach
    # Assuming dataframes_dict is your dictionary of DataFrames
    for key, df in dataframes_dict_transpose.items():
        
        # Exclude the first column (assuming it's at index 0)
        df_without_first_column = df[selected_countries]
        print(f"Summary statistics for DataFrame '{key}':")
        print(df[selected_countries].describe().round(2))
        print("\n")
    
    
    #other Statistical methods
    for key, df in dataframes_dict_transpose.items():
        
        # Exclude the first column (assuming it's at index 0)
        df_without_first_column = df[selected_countries]
        print(f"other statistics for DataFrame '{key}':")
        print(method_stats(df[selected_countries]))
        print("\n")
        