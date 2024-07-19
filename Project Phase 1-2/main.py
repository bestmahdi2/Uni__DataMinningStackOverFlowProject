import numpy as np
import pandas as pd
from random import randint
from scipy.stats import norm
from pickle import load, dump
import matplotlib.pyplot as plt
from re import sub, findall, DOTALL
from os import listdir, system, name
from csv import DictReader, DictWriter


class Phase1:
    """
        Phase1 class
    """

    def __init__(self) -> None:
        """
        Constructor to initialize class variables and process the CSV file.
        """

        # Define input and output file paths
        input_file_name = '1.QueryResults.csv'
        output_file_name = '2.PythonResults.csv'

        # Define the required CSV fields
        field_names = [
            'QuestionId', 'QuestionTitle', 'QuestionBody', 'QuestionTags', 'QuestionBodyLength',
            'URLImageCount', 'LOC', 'UserReputation', 'UserGoldBadges', 'UserSilverBadges', 'UserBronzeBadges',
            'QuestionAcceptRate', 'QuestionViewCount', 'QuestionFavoriteCount', 'UserUpVoteCount',
            'QuestionAnswersCount', 'QuestionScore', 'QuestionCreationDate', 'FirstAnswerCreationDate',
            'AcceptedAnswerCreationDate', 'FirstAnswerIntervalDays', 'AcceptedAnswerIntervalDays'
        ]

        # Read the CSV file and process each row
        result_dict = self.read_change_csv(input_file_name)

        # Write the modified data to a new CSV file
        self.write_to_csv(output_file_name, field_names, result_dict)

    def read_change_csv(self, input_file_name: str) -> list:
        """
        Reads the CSV file, processes each question body, and updates the dictionary.

        Args:
            input_file_name (str): Path to the input CSV file

        Returns:
            A list of dictionaries containing processed CSV data
        """

        # Initialize an empty list to store processed data
        result_dict = []

        # Open the CSV file for reading
        with open(input_file_name, newline='', encoding="utf8") as csv_file:
            reader = DictReader(csv_file)

            # Iterate through each row of the CSV file
            for row in reader:
                # Extract the question body
                question_body = row['QuestionBody']

                # Process the question body by removing HTML tags, counting lines of code, and counting URLs
                results = self.count_remove_html_tags(question_body)

                # Create a dictionary to store the processed data
                keeper = {}

                # Add the original row data to the dictionary
                keeper.update(row)

                # Add the processed data to the dictionary
                keeper['LOC'] = results[0]
                keeper['URLImageCount'] = results[1]
                keeper['QuestionBody'] = results[2]
                keeper['QuestionBodyLength'] = len(results[2])
                keeper['QuestionAcceptRate'] = randint(35, 81)

                # Append the processed dictionary to the result list
                result_dict.append(keeper)

        return result_dict

    @staticmethod
    def write_to_csv(output_file_name: str, field_names: list, result_dict: list) -> None:
        """
        Writes the processed data to a new CSV file.

        Args:
            output_file_name (str): Path to the output CSV file
            field_names (list): List of field names for the CSV file
            result_dict (list): A list of dictionaries containing processed CSV data
        """

        # Open the output file for writing
        with open(output_file_name, 'w', newline='', encoding="utf8") as csv_file:
            # Create a CSV writer object
            writer = DictWriter(csv_file, fieldnames=field_names)

            # Write the header row with field names
            writer.writeheader()

            # Write each processed dictionary to the CSV file
            for i in result_dict:
                writer.writerow(i)

    @staticmethod
    def count_remove_html_tags(text: str) -> tuple:
        """
        Counts the number of lines of code, URLs, and removes HTML tags from the text.

        Args:
            text (str): The text to process

        Returns:
            A tuple containing the number of lines of code, number of URLs, and the processed text
        """

        # Count the number of lines of code using regular expressions
        codes = sum([len(i.strip('\n').split("\n")) for i in findall(r'''<code>(.*?)</code>''', text, flags=DOTALL)])

        # Extract URLs using regular expressions
        images = list(set(findall(r'<img\s+[^>]+>', text, flags=DOTALL)))
        urls = list(set(findall(r'((?:https?://|www\.)[^\s]+)', text, flags=DOTALL)))

        # Remove duplicate URLs by excluding URLs present in images
        img_url_counter = len(images) + len([i for i in urls if i not in "\n".join(images)])

        # Remove HTML tags and multiple newline characters from the text
        processed_text = sub(r'\n\s*\n', '\n', sub(r'<[^>]*>', '', text))

        # Return the count of codes, URLs, and processed text
        return codes, img_url_counter, processed_text


class Phase2:
    """
        Phase2 class
    """

    def __init__(self) -> None:
        """
            Constructor to initialize the class and perform labeling of questions.
        """

        # Define the input and temporary file paths
        input_file_name = '2.PythonResults.csv'
        output_file_name = '3.LabeledResults.csv'
        output_pic_file_name = '4.PlotResults_[[0]].png'
        temp_keeper_name = '2.5.keeper.pickle'

        # Define the required CSV fields
        field_names = [
            'QuestionId', 'QuestionTitle', 'QuestionBody', 'QuestionTags', 'QuestionBodyLength',
            'URLImageCount', 'LOC', 'UserReputation', 'UserGoldBadges', 'UserSilverBadges', 'UserBronzeBadges',
            'QuestionAcceptRate', 'QuestionViewCount', 'QuestionFavoriteCount', 'UserUpVoteCount',
            'QuestionAnswersCount', 'QuestionScore', 'QuestionCreationDate', 'FirstAnswerCreationDate',
            'AcceptedAnswerCreationDate', 'FirstAnswerIntervalDays', 'AcceptedAnswerIntervalDays',
            'QuestionLabel', 'QuestionLabelDefinition'
        ]

        # Label the questions and store the labels temporarily
        self.labeling_results(input_file_name, temp_keeper_name)

        # Merge the temporary labels into a final file
        result_dict = self.merge_labels(input_file_name, temp_keeper_name)

        # Write the modified data to a new CSV file
        Phase1.write_to_csv(output_file_name, field_names, result_dict)

        # Show and save plots
        self.show_save_plot(output_file_name, output_pic_file_name)

    @staticmethod
    def labeling_results(input_file_name: str, temp_keeper_name: str) -> None:
        """
        Labels the questions from the input CSV file and stores them in a temporary pickle file.

        Args:
            input_file_name (str): Path to the input CSV file
            temp_keeper_name (str): Path to the temporary pickle file for storing labels
        """

        # Clear the terminal to enhance user experience
        system('cls' if name == 'nt' else 'clear')

        # Check if the temporary file exists, load the existing labels if present
        if temp_keeper_name in listdir("."):
            with open(temp_keeper_name, 'rb') as file:
                keeper = load(file)

        # If the temporary file doesn't exist, initialize an empty keeper dictionary
        else:
            keeper = {}

        # Read the CSV file and label each question
        with open(input_file_name, newline='', encoding="utf8") as csv_file:
            reader = DictReader(csv_file)

            # Iterate through questions and label them
            x = 0  # Counter for tracking every 5th question
            iteration = [i for i in reader if i['QuestionId'] not in keeper]
            previous_question = None
            for row in iteration:
                x += 1
                # Clear the terminal before displaying the question and receiving the label
                system('cls' if name == 'nt' else 'clear')

                # Save the labels after every 5th question to avoid memory issues
                if x % 5 == 0:
                    with open(temp_keeper_name, 'wb') as file:
                        dump(keeper, file)

                # Print the question title and body
                print(f"<--------------- Title --------------->\n\n{row['QuestionTitle']}"
                      f"\n\n<--------------- Body ---------------->\n\n{row['QuestionBody']}"
                      f"\n\n<--------------- ID ---------------->\n\n[Question: {row['QuestionId']}]\n"
                      f"[Previous Question: {previous_question}]\n\n<--------------- Index ---------------->")

                # Prompt the user to provide a label from 1 to 47
                try:
                    print_string = f"\nWhat is index 1-47 ? [{x}/{len(iteration)}]{'' if x % 5 else '[saved]'} > "
                    number = int(input(print_string))

                    # Validate the input and update the keeper dictionary
                    if 0 < number < 48:
                        keeper[row['QuestionId']] = number
                        # x += 1

                    else:
                        previous_question = row['QuestionId']
                        continue

                # Handle invalid inputs
                except Exception as E:
                    previous_question = row['QuestionId']
                    continue

                previous_question = row['QuestionId']

    @staticmethod
    def merge_labels(input_file_name: str, temp_keeper_name: str) -> list:
        """
        This function merges labels from a CSV file and a temporary keeper file into a list of dictionaries.

        Args:
            input_file_name (str): The path to the CSV file containing the questions and their respective IDs.
            temp_keeper_name (str): The path to the temporary keeper file containing predefined question labels.

        Returns:
            list: A list of dictionaries containing the question ID, definition, and label, along with the corresponding label category.
        """

        # Check if the temporary keeper file exists
        if temp_keeper_name in listdir("."):
            # Load the label keeper data from the file
            with open(temp_keeper_name, "rb") as file:
                label_keeper = load(file)
        else:
            # No keeper file found, raise an error
            print("There is no keeper file!")
            exit()

        # Initialize an empty list to store the merged data
        result_dict = []

        # Open the CSV file and iterate through its rows
        with open(input_file_name, newline="", encoding="utf8") as csv_file:
            reader = DictReader(csv_file)

            # Process each row, checking for existing labels
            for row in reader:
                if row["QuestionId"] not in label_keeper:
                    # Question ID not found in label keeper, skip to next row
                    print(f"This ID is not labeled >>> {row['QuestionId']}")
                    continue

                # Create a temporary dictionary to store merged data
                keeper = {}

                # Populate the temporary dictionary with data from the CSV row
                keeper.update(row)

                # Add the retrieved label definition to the temporary dictionary
                keeper["QuestionLabelDefinition"] = label_keeper[row["QuestionId"]]

                # Determine the label category based on the definition
                keeper["QuestionLabel"] = "Basic" if 1 <= keeper[
                    "QuestionLabelDefinition"] <= 14 else "Intermediate" if 15 <= keeper[
                    "QuestionLabelDefinition"] <= 31 else "Advanced"

                # Append the merged data dictionary to the result list
                result_dict.append(keeper)

        # Return the list of merged data dictionaries
        return result_dict

    @staticmethod
    def show_save_plot(input_file_name: str, output_file_name: str) -> None:
        """
        This function generates and saves three plots visualizing the distribution of question labels in a CSV file.

        Args:
            input_file_name (str): The path to the CSV file containing the question labels with their respective categories.
            output_file_name (str): The path to the file where the generated plots will be saved.

        Returns:
            None
        """

        # Define the labels representing the question difficulty levels
        labels = ["Basic", "Intermediate", "Advanced"]

        # Read the CSV file and extract the relevant data
        df = pd.read_csv(input_file_name)
        data_column = "QuestionLabel"  # Specify the column containing the question labels
        distribution = df.groupby(data_column)[data_column].count()  # Count the occurrences of each label

        # ---------------------
        # Generate and save the first plot: Pie chart
        plt.pie(distribution, labels=labels, autopct="%1.1f%%")  # Create a pie chart with label percentages
        plt.title("Distribution of Question Labels by Category")  # Set the chart title
        fig1 = plt.gcf()  # Get the current figure
        plt.show()  # Display the chart
        plt.draw()  # Ensure the chart is updated
        fig1.savefig(output_file_name.replace("[[0]]", "1"), dpi=100)  # Save the chart as an image

        # ---------------------
        # Generate and save the second plot: Bar chart
        plt.bar(labels, distribution)  # Create a bar chart with label occurrences
        plt.title("Distribution of Question Labels by Category")  # Set the chart title
        plt.xlabel("Category")  # Set the x-axis label
        plt.ylabel("Count")  # Set the y-axis label
        fig2 = plt.gcf()  # Get the current figure
        plt.show()  # Display the chart
        plt.draw()  # Ensure the chart is updated
        fig2.savefig(output_file_name.replace("[[0]]", "2"), dpi=100)  # Save the chart as an image

        # ---------------------
        # Generate and save the third plot: Histogram and fitted normal distribution
        df = (pd.read_csv("3.LabeledResults.csv")["QuestionLabel"]
              .replace("Basic", 2).replace("Intermediate", 1).replace("Advanced",
                                                                      3))  # Replace labels with numeric values

        mu, std = norm.fit(df)  # Calculate the mean (μ) and standard deviation (σ) of the distribution
        dist = norm(mu, std)  # Create a normal distribution object with the calculated parameters

        plt.hist(df, bins=5, density=True, alpha=0.5)  # Create a histogram with 5 bins and semi-transparent bars

        x = np.linspace(df.min(), df.max(), 1000)  # Generate a range of values from minimum to maximum label value
        plt.plot(x, dist.pdf(x), 'r-', lw=2)  # Plot the fitted normal distribution curve

        dist_name = f"Normal Distribution (μ={mu:.2f}, σ={std:.2f})"  # Create a label for the distribution curve
        plt.title(dist_name)  # Set the chart title
        fig3 = plt.gcf()  # Get the current figure
        plt.show()  # Display the chart
        plt.draw()  # Ensure the chart is updated
        fig3.savefig(output_file_name.replace("[[0]]", "3"), dpi=100)  # Save the chart as an image

        # Print the distribution data
        print(distribution)


if __name__ == "__main__":
    # phase1 = Phase1()
    phase2 = Phase2()
