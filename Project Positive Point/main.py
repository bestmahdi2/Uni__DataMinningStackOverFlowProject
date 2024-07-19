import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from tqdm import tqdm
from nltk import download
from sklearn.svm import SVC
from os import listdir, path
from textblob import TextBlob
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from gensim.models import Doc2Vec
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from gensim.models.doc2vec import TaggedDocument
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class Main:
    """
    Main class for data processing and model training.
    """

    # Define input and output file paths
    input_file_name_1 = 'train.csv'
    split_csv_file_name_1 = 'train'
    split_csv_folder_name_1 = './SplitFiles1/'
    output_text_process_file_name_1 = 'TextProcessedTrainData.csv'
    input_file_name_2 = 'valid.csv'
    split_csv_file_name_2 = 'valid'
    split_csv_folder_name_2 = './SplitFiles2/'
    output_text_process_file_name_2 = 'TextProcessedTestData.csv'
    train_file_name = 'TextProcessedTrainData.csv'
    test_file_name = 'TextProcessedTestData.csv'
    confusion_matrix_name = 'ConfusionMatrixPlot.png'
    comparison_plot_name = 'ComparisonResultPlot.png'
    optimized_comparison_plot_name_1 = 'OptimizedComparisonHstackingResultPlot.png'
    keep_results_file_1 = 'OptimizedComparisonHstackingResult.results'
    optimized_comparison_plot_name_2 = 'OptimizedComparisonHyperResultPlot.png'
    keep_results_file_2 = 'OptimizedComparisonHyperResult.results'

    csv_split_count = 20

    def __init__(self) -> None:
        """
        Constructor for the Main class. Initializes necessary parameters and performs initial setup.
        """
        tqdm.pandas()
        download('stopwords')
        print()

        print("--------------- Part 1 / Train ---------------\n")

        # Step 1: Splitting the CSV files for training
        print("Start splitting the CSV!")
        self.split_csv(self.input_file_name_1, self.split_csv_file_name_1, self.split_csv_folder_name_1,
                       self.csv_split_count)
        print("Finish splitting the CSV!\n")

        # Step 2: Preprocessing each split file
        print("Start preprocessing on each file >>")
        x, count = 0, len(listdir(self.split_csv_folder_name_1))
        for file in listdir(self.split_csv_folder_name_1):
            x += 1
            print(f'({x}/{count}) {file}:')
            self.preprocessing(f'{self.split_csv_folder_name_1}{file}', self.split_csv_folder_name_1 + "output_" + file)
            print()
        print("Finish preprocessing on texts!\n")

        # Step 3: Merging the CSV files
        print("Start Merging the CSV!")
        self.merge_csv(self.split_csv_folder_name_1, self.output_text_process_file_name_1)
        print("Finish Merging the CSV!\n")

        print("--------------- Part 2 / Test ---------------\n")

        # Similar steps for testing data
        # ...

        print("--------------- Part 3 ---------------\n")

        # Step 4: Training models and creating plots
        self.train_models_create_plots(self.train_file_name, self.test_file_name, self.comparison_plot_name,
                                       self.confusion_matrix_name)
        print("Created Results Plot!\n")

        print("--------------- Part 4 ---------------\n")

        # Step 5: Optimizing with Hstacking Text
        print("Start Hstacking Text optimizing!")
        self.optimize_hstacking_text(self.train_file_name, self.test_file_name, self.optimized_comparison_plot_name_1,
                                     self.keep_results_file_1)
        print("Finish Hstacking Text optimizing!\n")

        # Step 6: Optimizing with Set Hyper Parameters
        print("Start Set Hyper Parameters optimizing!")
        self.optimize_set_hyper_parameter(self.train_file_name, self.test_file_name,
                                          self.optimized_comparison_plot_name_2, self.keep_results_file_2)
        print("Finish Set Hyper Parameters optimizing!")

    @staticmethod
    def split_csv(input_file: str, output_prefix: str, split_folder: str, num_files: int) -> None:
        """
        Splits a CSV file into multiple smaller CSV files.

        Parameters:
        - input_file (str): Path to the input CSV file.
        - output_prefix (str): Prefix to be used for the names of the output files.
        - split_folder (str): Folder where the split CSV files will be saved.
        - num_files (int): Number of files to split the input into.

        Returns:
        None
        """

        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Calculate the number of rows per output file
        rows_per_file = len(df) // num_files

        # Split the DataFrame into chunks and save each chunk to a separate CSV file
        for i in range(num_files):
            start_idx = i * rows_per_file
            end_idx = (i + 1) * rows_per_file if i < num_files - 1 else len(df)

            # Extract chunk of data
            chunk = df.iloc[start_idx:end_idx]

            # Define the output file path
            output_file = f"{split_folder}{output_prefix}_{i + 1}.csv"

            # Save the chunk to a CSV file without index
            chunk.to_csv(output_file, index=False)

    def preprocessing(self, input_file: str, output_file: str) -> None:
        """
        Preprocesses a CSV file containing text data and saves the processed data to a new CSV file.

        Parameters:
        - input_file (str): Path to the input CSV file.
        - output_file (str): Path to the output CSV file.

        Returns:
        None
        """

        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Merge QuestionTags, QuestionTitle and QuestionBody fields into a new feature 'MergedText'
        df['MergedText'] = (df['Tags'].astype(str) + ', ' + df['Title'].astype(str) + ', ' +
                            df['Body'].astype(str))

        # Create stop words and porter stemmer
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()

        # Apply text preprocessing to the 'MergedText' column
        df['ProcessedText'] = tqdm(
            df['MergedText'].progress_apply(self.preprocess_given_text, stop_words=stop_words, ps=ps),
            total=len(df))

        # Save the processed DataFrame to a new CSV file
        df.to_csv(output_file, index=False)

    @staticmethod
    def preprocess_given_text(text: str, stop_words: set, ps) -> str:
        """
        Preprocesses a given text by removing stop words, performing stemming, and correcting spellings.

        Parameters:
        - text (str): The input text to be preprocessed.
        - stop_words (set): Set of stop words to be removed.
        - ps: PorterStemmer object for stemming.

        Returns:
        str: The preprocessed text.
        """

        # Remove stop words
        words = [word for word in text.lower().split() if word.lower() not in stop_words]

        # Perform stemming
        stemmed_words = [ps.stem(word) for word in words]

        # Correct spellings using TextBlob
        corrected_text = ' '.join([str(TextBlob(word).correct()) for word in stemmed_words])

        return corrected_text

    @staticmethod
    def merge_csv(split_folder: str, output_file: str) -> None:
        """
        Merges multiple CSV files into a single CSV file.

        Parameters:
        - split_folder (str): Path to the folder containing the split CSV files.
        - output_file (str): Path to the output CSV file.

        Returns:
        None
        """

        # Get a list of all CSV files in the split folder
        csv_files = [f for f in listdir(split_folder) if f.startswith("output_")]

        # Initialize an empty DataFrame to store the merged data
        merged_df = pd.DataFrame()

        # Read each split file and concatenate into the merged DataFrame
        for csv_file in csv_files:
            file_path = path.join(split_folder, csv_file)
            chunk_df = pd.read_csv(file_path)
            merged_df = pd.concat([merged_df, chunk_df], ignore_index=True)

        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv(output_file, index=False)

    @staticmethod
    def vectorize(input_file: str, output_file: str) -> None:
        """
        Performs vectorization on the text data and adds features to the original DataFrame.

        Parameters:
        - input_file (str): Path to the input CSV file.
        - output_file (str): Path to the output CSV file.

        Returns:
        None
        """

        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Vectorization using CountVectorizer
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        x_count = vectorizer.fit_transform(df['ProcessedText'])

        # Vectorization using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        x_tfidf = tfidf_vectorizer.fit_transform(df['ProcessedText'])

        # Add n-gram and TF-IDF features to the DataFrame
        df_count = pd.DataFrame(x_count.toarray(), columns=vectorizer.get_feature_names_out())
        df_tfidf = pd.DataFrame(x_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        # Concatenate the original DataFrame with the n-gram and TF-IDF features
        df_with_features = pd.concat([df, df_count, df_tfidf], axis=1)

        # Save the DataFrame with features to a new CSV file
        df_with_features.to_csv(output_file, index=False)

    @staticmethod
    def train_test_split(input_file: str, train_file: str, test_file: str) -> None:
        """
        Splits a dataset into training and testing sets and saves them into separate CSV files.

        Parameters:
        - input_file (str): Path to the input CSV file containing the dataset.
        - train_file (str): Path to the output CSV file for the training set.
        - test_file (str): Path to the output CSV file for the testing set.

        Returns:
        None
        """

        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file, low_memory=False)

        # Assume 'Y' is the column you want to predict, and 'test_size' is the proportion of the dataset to
        # include in the test split
        x_train, x_test, y_train, y_test = train_test_split(df.drop('Y', axis=1),
                                                            df['Y'], test_size=0.2, random_state=42)

        # Combine x_train and y_train into one DataFrame
        train_data = pd.concat([x_train, y_train], axis=1)

        # Combine x_test and y_test into one DataFrame
        test_data = pd.concat([x_test, y_test], axis=1)

        # Save the combined DataFrames to new CSV files
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)

    @staticmethod
    def train_models_create_plots(train_file_name: str, test_file_name: str, comparison_plot_name: str,
                                  confusion_matrix_name: str) -> None:
        """
        Trains multiple machine learning models, evaluates their performance, and creates confusion matrices.

        Parameters:
        - train_file_name (str): Path to the CSV file containing the training dataset.
        - test_file_name (str): Path to the CSV file containing the testing dataset.
        - comparison_plot_name (str): Name of comparison plot to be saved.
        - confusion_matrix_name (str): Name of confusion matrix to be saved.

        Returns:
        None
        """

        # Load the training dataset
        train_df = pd.read_csv(train_file_name, low_memory=False)
        print("Train imported !")

        # Load the testing dataset
        test_df = pd.read_csv(test_file_name, low_memory=False)
        print("Test imported !")

        # Drop rows with missing values in the 'ProcessedText' column
        train_df = train_df.dropna(subset=['ProcessedText'])
        test_df = test_df.dropna(subset=['ProcessedText'])

        # Drop rows with missing values in the target variable 'Y'
        train_df = train_df.dropna(subset=['Y'])

        # Separate features and labels
        x_train, y_train = train_df['ProcessedText'], train_df['Y']
        x_test, y_test = test_df['ProcessedText'], test_df['Y']

        # Vectorize the text data using TF-IDF
        vectorizer = TfidfVectorizer()
        x_train_tfidf = vectorizer.fit_transform(x_train)
        x_test_tfidf = vectorizer.transform(x_test)

        # Initialize the classifiers
        models = {
            'Naive Bayes': MultinomialNB(),
            'SVC': SVC(),
            'CART': DecisionTreeClassifier(),
            'Logistic Regression': LogisticRegression(),
            'MLP': MLPClassifier(max_iter=500),
            'XGBoost': xgb.XGBClassifier()
        }

        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()

        # Encode the target variable 'y_train'
        y_train_encoded = label_encoder.fit_transform(y_train)

        # Train and evaluate each model
        results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

        for model_name, model in models.items():
            print(f'Start {model_name}!')

            # Train the model
            model.fit(x_train_tfidf, y_train_encoded)

            # Make predictions on the test set
            y_pred_encoded = model.predict(x_test_tfidf)

            # Inverse transform the predictions to get original class labels
            y_pred = label_encoder.inverse_transform(y_pred_encoded)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Store the results
            results['Model'].append(model_name)
            results['Accuracy'].append(accuracy)
            results['Precision'].append(precision)
            results['Recall'].append(recall)
            results['F1 Score'].append(f1)

            print(f'Finish {model_name}!')

        # Display the results
        results_df = pd.DataFrame(results)

        print(results_df)

        # Plotting confusion matrix
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

        for ax, (model_name, model) in zip(axes.flatten(), models.items()):
            # Train the model
            model.fit(x_train_tfidf, y_train_encoded)

            # Make predictions on the test set
            y_pred_encoded = model.predict(x_test_tfidf)

            # Inverse transform the predictions to get original class labels
            y_pred = label_encoder.inverse_transform(y_pred_encoded)

            # Confusion matrix with specified labels parameter
            cm = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)

            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_encoder.classes_,
                        yticklabels=label_encoder.classes_, ax=ax)
            ax.set_title(f'Confusion Matrix - {model_name}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')

        # Ensure tight layout for better visualization
        plt.tight_layout()

        # Save the generated comparison plot to a file
        plt.savefig(comparison_plot_name)

        # Display the comparison plot
        plt.show()

        # Plotting comparing methods
        plt.figure(figsize=(12, 8))

        # Bar plot for Accuracy
        plt.subplot(2, 2, 1)
        sns.barplot(x='Model', y='Accuracy', data=results_df)
        plt.title('Accuracy Comparison')

        # Bar plot for Precision
        plt.subplot(2, 2, 2)
        sns.barplot(x='Model', y='Precision', data=results_df)
        plt.title('Precision Comparison')

        # Bar plot for Recall
        plt.subplot(2, 2, 3)
        sns.barplot(x='Model', y='Recall', data=results_df)
        plt.title('Recall Comparison')

        # Bar plot for F1 Score
        plt.subplot(2, 2, 4)
        sns.barplot(x='Model', y='F1 Score', data=results_df)
        plt.title('F1 Score Comparison')

        # Ensure tight layout for better visualization
        plt.tight_layout()

        # Save the generated comparison plot to a file
        plt.savefig(comparison_plot_name)

        # Display the comparison plot
        plt.show()

    @staticmethod
    def optimize_hstacking_text(train_file_name: str, test_file_name: str, comparison_plot_name: str,
                                keep_results_file: str):
        """
        Optimizes a Stacking Classifier using Decision Trees as base classifiers (CART) and additional classifiers
        (Logistic Regression, KNN, MLP, Naive Bayes, SVC, and XGBoost). The function performs text data vectorization
        with various approaches (TF-IDF, n-grams, Doc2Vec, and Latent Dirichlet Allocation) and combines the features
        to train the Stacking Classifier with Decision Trees.

        Parameters:
        - train_file_name (str): File path for the training dataset in CSV format.
        - test_file_name (str): File path for the testing dataset in CSV format.
        - comparison_plot_name (str): File path for saving image of the plot.
        - keep_results_file (str): File path for saving results of the methods.

        Returns:
        None (prints the results to the console).
        """

        methods = ["CART", "LogisticRegression", "KNN", "MLP", "SVC", "XGBoost"]  # "Naive Bayes",

        # Customize stop words as needed
        stop_words = ['sql', 'mysql', 'database']

        # Train and evaluate each model
        results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

        # Load the training dataset
        train_df = pd.read_csv(train_file_name, low_memory=False)
        print("Train imported !")

        # Load the testing dataset
        test_df = pd.read_csv(test_file_name, low_memory=False)
        print("Test imported !")

        # Separate features and labels
        x_train, y_train = train_df['ProcessedText'], train_df['Y']
        x_test, y_test = test_df['ProcessedText'], test_df['Y']

        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()

        # Encode the target variable 'y_train'
        y_train = label_encoder.fit_transform(y_train)

        # Encode the target variable 'y_test'
        y_test = label_encoder.transform(y_test)

        # Define vectorizers
        tfidf_vectorizer = TfidfVectorizer()
        bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))

        # Approach 1: Vectorize the text data using TF-IDF
        x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
        x_test_tfidf = tfidf_vectorizer.transform(x_test)

        # Approach 2: Vectorize the text data using n-grams (bigrams)
        x_train_bigram = bigram_vectorizer.fit_transform(x_train)
        x_test_bigram = bigram_vectorizer.transform(x_test)

        # Approach 3: Vectorize the text data using Doc2Vec
        tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(x_train)]
        doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)
        doc2vec_model.build_vocab(tagged_data)
        doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=10)

        x_train_doc2vec = np.array([doc2vec_model.infer_vector(doc.split()) for doc in x_train])
        x_test_doc2vec = np.array([doc2vec_model.infer_vector(doc.split()) for doc in x_test])

        # Approach 4: Apply Topic Modeling (Latent Dirichlet Allocation)
        vectorizer_lda = CountVectorizer(stop_words=stop_words)
        x_train_lda = vectorizer_lda.fit_transform(x_train)
        x_test_lda = vectorizer_lda.transform(x_test)

        lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        x_train_topics = lda_model.fit_transform(x_train_lda)
        x_test_topics = lda_model.transform(x_test_lda)

        for method in methods:
            print(f"Start {method}!")

            if method == "CART":
                # Define base classifiers
                base_classifiers = [
                    ('tree_tfidf', DecisionTreeClassifier()),
                    ('tree_bigram', DecisionTreeClassifier()),
                    ('tree_doc2vec', DecisionTreeClassifier()),
                    ('tree_lda', DecisionTreeClassifier())
                ]

                # Concatenate features along the second axis
                x_train_combined = np.concatenate([
                    x_train_tfidf.toarray(),
                    x_train_bigram.toarray(),
                    x_train_doc2vec,
                    x_train_topics
                ], axis=1)

                x_test_combined = np.concatenate([
                    x_test_tfidf.toarray(),
                    x_test_bigram.toarray(),
                    x_test_doc2vec,
                    x_test_topics
                ], axis=1)

                # Initialize the Stacking Classifier with Decision Trees as base estimators
                stacking_classifier = StackingClassifier(
                    estimators=base_classifiers,
                    final_estimator=DecisionTreeClassifier(),  # You can change the final estimator
                    stack_method='predict_proba',  # Use 'predict_proba' for probabilities
                    passthrough=True,
                    cv=3  # Number of cross-validation folds
                )

            elif method == "LogisticRegression":
                # Define base classifiers
                base_classifiers = [
                    ('knn_tfidf', KNeighborsClassifier()),
                    ('knn_bigram', KNeighborsClassifier()),
                    ('knn_doc2vec', KNeighborsClassifier()),
                    ('knn_lda', KNeighborsClassifier())
                ]

                # Concatenate features along the second axis
                x_train_combined = hstack([
                    x_train_tfidf,
                    x_train_bigram,
                    x_train_doc2vec,
                    x_train_topics
                ])

                x_test_combined = hstack([
                    x_test_tfidf,
                    x_test_bigram,
                    x_test_doc2vec,
                    x_test_topics
                ])

                # Define the stacking classifier with Logistic Regression as the final estimator
                stacking_classifier = StackingClassifier(
                    estimators=base_classifiers,
                    final_estimator=LogisticRegression(),  # You can change the final estimator
                    stack_method='predict_proba',  # Use 'predict_proba' for probabilities
                    passthrough=True,
                    cv=3  # Number of cross-validation folds
                )

            elif method == "KNN":
                # Define base classifiers
                base_classifiers = [
                    ('knn_tfidf', KNeighborsClassifier()),
                    ('knn_bigram', KNeighborsClassifier()),
                    ('knn_doc2vec', KNeighborsClassifier()),
                    ('knn_lda', KNeighborsClassifier())
                ]

                # Concatenate features along the second axis
                x_train_combined = hstack([
                    x_train_tfidf,
                    x_train_bigram,
                    x_train_doc2vec,
                    x_train_topics
                ])

                x_test_combined = hstack([
                    x_test_tfidf,
                    x_test_bigram,
                    x_test_doc2vec,
                    x_test_topics
                ])

                # Define the stacking classifier
                stacking_classifier = StackingClassifier(
                    estimators=base_classifiers,
                    final_estimator=RandomForestClassifier(),  # You can change the final estimator
                    stack_method='predict_proba',  # Use 'predict_proba' for probabilities
                    passthrough=True,
                    cv=3  # Number of cross-validation folds
                )

            elif method == "MLP":

                # Define base classifiers with MLP as base estimator
                base_classifiers = [
                    ('mlp_tfidf', MLPClassifier()),  # You can customize MLP parameters
                    ('mlp_bigram', MLPClassifier()),  # You can customize MLP parameters
                    ('mlp_doc2vec', MLPClassifier()),  # You can customize MLP parameters
                    ('mlp_lda', MLPClassifier())  # You can customize MLP parameters
                ]

                # Concatenate features along the second axis
                x_train_combined = hstack([
                    x_train_tfidf,
                    x_train_bigram,
                    x_train_doc2vec,
                    x_train_topics
                ])

                x_test_combined = hstack([
                    x_test_tfidf,
                    x_test_bigram,
                    x_test_doc2vec,
                    x_test_topics
                ])

                # Define the stacking classifier with MLP as the final estimator
                stacking_classifier = StackingClassifier(
                    estimators=base_classifiers,
                    final_estimator=MLPClassifier(),  # You can customize MLP parameters
                    stack_method='predict_proba',  # Use 'predict_proba' for probabilities
                    passthrough=True,
                    cv=3  # Number of cross-validation folds
                )

            elif method == "Naive Bayes":

                # Define base classifiers
                base_classifiers = [
                    ('nb_tfidf', MultinomialNB()),
                    ('nb_bigram', MultinomialNB()),
                    ('nb_doc2vec', MultinomialNB()),
                    ('nb_lda', MultinomialNB())
                ]

                # Concatenate features along the second axis
                x_train_combined = hstack([
                    x_train_tfidf,
                    x_train_bigram,
                    x_train_doc2vec,
                    x_train_topics
                ])

                x_test_combined = hstack([
                    x_test_tfidf,
                    x_test_bigram,
                    x_test_doc2vec,
                    x_test_topics
                ])

                # Define the stacking classifier with Naive Bayes as the final estimator
                stacking_classifier = StackingClassifier(
                    estimators=base_classifiers,
                    final_estimator=MultinomialNB(),  # You can customize Naive Bayes parameters
                    stack_method='predict_proba',  # Use 'predict_proba' for probabilities
                    passthrough=True,
                    cv=3  # Number of cross-validation folds
                )

            elif method == "SVC":

                # Define base classifiers
                base_classifiers = [
                    ('knn_tfidf', KNeighborsClassifier()),
                    ('knn_bigram', KNeighborsClassifier()),
                    ('knn_doc2vec', KNeighborsClassifier()),
                    ('knn_lda', KNeighborsClassifier())
                ]

                # Concatenate features along the second axis
                x_train_combined = hstack([
                    x_train_tfidf,
                    x_train_bigram,
                    x_train_doc2vec,
                    x_train_topics
                ])

                x_test_combined = hstack([
                    x_test_tfidf,
                    x_test_bigram,
                    x_test_doc2vec,
                    x_test_topics
                ])

                # Define the stacking classifier
                stacking_classifier = StackingClassifier(
                    estimators=base_classifiers,
                    final_estimator=SVC(probability=True),  # Use Support Vector Classifier as the final estimator
                    stack_method='predict_proba',  # Use 'predict_proba' for probabilities
                    passthrough=True,
                    cv=3  # Number of cross-validation folds
                )

            else:  # if "XGBoost":

                # Define base classifiers
                base_classifiers = [
                    ('knn_tfidf', KNeighborsClassifier()),
                    ('knn_bigram', KNeighborsClassifier()),
                    ('knn_doc2vec', KNeighborsClassifier()),
                    ('knn_lda', KNeighborsClassifier())
                ]

                # Concatenate features along the second axis
                x_train_combined = hstack([
                    x_train_tfidf,
                    x_train_bigram,
                    x_train_doc2vec,
                    x_train_topics
                ])

                x_test_combined = hstack([
                    x_test_tfidf,
                    x_test_bigram,
                    x_test_doc2vec,
                    x_test_topics
                ])

                # Define the stacking classifier
                stacking_classifier = StackingClassifier(
                    estimators=base_classifiers,
                    final_estimator=XGBClassifier(),  # Use XGBoost as the final estimator
                    stack_method='predict_proba',  # Use 'predict_proba' for probabilities
                    passthrough=True,
                    cv=3  # Number of cross-validation folds
                )

            # Fit the stacking classifier
            stacking_classifier.fit(x_train_combined, y_train)

            # Make predictions on the test set
            predictions_stacking = stacking_classifier.predict(x_test_combined)

            # Evaluate the stacking classifier
            accuracy_stacking = accuracy_score(y_test, predictions_stacking)
            precision_stacking = precision_score(y_test, predictions_stacking, average='weighted', zero_division=0)
            recall_stacking = recall_score(y_test, predictions_stacking, average='weighted', zero_division=0)
            f1_stacking = f1_score(y_test, predictions_stacking, average='weighted')

            # Store the results
            results['Model'].append(method)
            results['Accuracy'].append(accuracy_stacking)
            results['Precision'].append(precision_stacking)
            results['Recall'].append(recall_stacking)
            results['F1 Score'].append(f1_stacking)

            print(f"Done {method}!")

        # Display the results
        results_df = pd.DataFrame(results)

        print(results_df)
        with open(keep_results_file, "wb") as f:
            pickle.dump(results, f)

        # Plotting comparing methods
        plt.figure(figsize=(12, 8))

        # Bar plot for Accuracy
        plt.subplot(2, 2, 1)
        sns.barplot(x='Model', y='Accuracy', data=results_df)
        plt.title('Accuracy Comparison')

        # Bar plot for Precision
        plt.subplot(2, 2, 2)
        sns.barplot(x='Model', y='Precision', data=results_df)
        plt.title('Precision Comparison')

        # Bar plot for Recall
        plt.subplot(2, 2, 3)
        sns.barplot(x='Model', y='Recall', data=results_df)
        plt.title('Recall Comparison')

        # Bar plot for F1 Score
        plt.subplot(2, 2, 4)
        sns.barplot(x='Model', y='F1 Score', data=results_df)
        plt.title('F1 Score Comparison')

        # Ensure tight layout for better visualization
        plt.tight_layout()

        # Save the generated comparison plot to a file
        plt.savefig(comparison_plot_name)

        # Display the comparison plot
        plt.show()

    @staticmethod
    def optimize_set_hyper_parameter(train_file_name: str, test_file_name: str, comparison_plot_name: str,
                                     keep_results_file: str):
        """
        Optimize hyperparameters for different machine learning models using various text vectorization approaches.

        Parameters:
        - train_file_name (str): File name for the training dataset.
        - test_file_name (str): File name for the testing dataset.
        - comparison_plot_name (str): File path for saving image of the plot.
        - keep_results_file (str): File path for saving results of the methods.

        Returns:
        None

        """
        # List of machine learning models
        methods = ["CART", "LogisticRegression", "KNN", "MLP", "SVC", "XGBoost"]  # "Naive Bayes",

        # Dictionary to store accuracy results for each method and vectorization approach
        # results = {k: {"TF-IDF": 0.0, "Bigrams": 0.0, "Doc2Vec": 0.0, "Topic Modeling": 0.0} for k in methods}
        results = {'Model': [], 'TF-IDF': [], 'Bigrams': [], 'Doc2Vec': [], 'Topic Modeling': []}

        # List of stop words to be excluded during vectorization
        stop_words = ['sql', 'mysql', 'database']

        # Load the training dataset
        train_df = pd.read_csv(train_file_name, low_memory=False)
        print("Train imported !")

        # Load the testing dataset
        test_df = pd.read_csv(test_file_name, low_memory=False)
        print("Test imported !")

        # Separate features and labels
        x_train, y_train = train_df['ProcessedText'], train_df['Y']
        x_test, y_test = test_df['ProcessedText'], test_df['Y']

        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()

        # Encode the target variable 'y_train'
        y_train = label_encoder.fit_transform(y_train)

        # Encode the target variable 'y_test'
        y_test = label_encoder.transform(y_test)

        # Approach 1: Vectorize the text data using TF-IDF and use Logistic Regression
        tfidf_vectorizer = TfidfVectorizer()
        x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
        x_test_tfidf = tfidf_vectorizer.transform(x_test)

        for method in methods:
            print(f"Start {method}!")

            if method == "LogisticRegression":
                # Train a Logistic Regression classifier using TF-IDF
                lr_classifier_tfidf = LogisticRegression(C=1.0, max_iter=100, random_state=42)
                lr_classifier_tfidf.fit(x_train_tfidf, y_train)

                # Make predictions on the TF-IDF test set
                predictions_tfidf = lr_classifier_tfidf.predict(x_test_tfidf)

                # Evaluate the accuracy for TF-IDF
                accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
                # results[method]["TF-IDF"] = accuracy_tfidf

                # Approach 2: Vectorize the text data using n-grams (bigrams) and use Logistic Regression
                bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
                x_train_bigram = bigram_vectorizer.fit_transform(x_train)
                x_test_bigram = bigram_vectorizer.transform(x_test)

                # Train a Logistic Regression classifier using bigrams
                lr_classifier_bigram = LogisticRegression(C=1.0, max_iter=100, random_state=42)
                lr_classifier_bigram.fit(x_train_bigram, y_train)

                # Make predictions on the bigram test set
                predictions_bigram = lr_classifier_bigram.predict(x_test_bigram)

                # Evaluate the accuracy for bigrams
                accuracy_bigram = accuracy_score(y_test, predictions_bigram)
                # results[method]["Bigrams"] = accuracy_bigram

                # Approach 3: Vectorize the text data using Doc2Vec and use Logistic Regression
                tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(x_train)]
                doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)
                doc2vec_model.build_vocab(tagged_data)
                doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=10)

                x_train_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_train]
                x_test_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_test]

                # Train a Logistic Regression classifier using Doc2Vec
                lr_classifier_doc2vec = LogisticRegression(C=1.0, max_iter=100, random_state=42)
                lr_classifier_doc2vec.fit(x_train_doc2vec, y_train)

                # Make predictions on the Doc2Vec test set
                predictions_doc2vec = lr_classifier_doc2vec.predict(x_test_doc2vec)

                # Evaluate the accuracy for Doc2Vec
                accuracy_doc2vec = accuracy_score(y_test, predictions_doc2vec)
                # results[method]["Doc2Vec"] = accuracy_doc2vec

                # Approach 4: Apply Topic Modeling (Latent Dirichlet Allocation) and use Logistic Regression
                vectorizer_lda = CountVectorizer(stop_words=stop_words)
                x_train_lda = vectorizer_lda.fit_transform(x_train)
                x_test_lda = vectorizer_lda.transform(x_test)

                lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
                x_train_topics = lda_model.fit_transform(x_train_lda)
                x_test_topics = lda_model.transform(x_test_lda)

                # Train a Logistic Regression classifier using Topic Modeling features
                lr_classifier_lda = LogisticRegression(C=1.0, max_iter=100, random_state=42)
                lr_classifier_lda.fit(x_train_topics, y_train)

                # Make predictions on the Topic Modeling test set
                predictions_lda = lr_classifier_lda.predict(x_test_topics)

                # Evaluate the accuracy for Topic Modeling
                accuracy_lda = accuracy_score(y_test, predictions_lda)
                # results[method]["Topic Modeling"] = accuracy_lda

            elif method == "CART":
                # Train a Decision Tree classifier using TF-IDF
                dt_classifier_tfidf = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=42)
                dt_classifier_tfidf.fit(x_train_tfidf, y_train)

                # Make predictions on the TF-IDF test set
                predictions_tfidf = dt_classifier_tfidf.predict(x_test_tfidf)

                # Evaluate the accuracy for TF-IDF
                accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
                # results[method]["TF-IDF"] = accuracy_tfidf

                # Approach 2: Vectorize the text data using n-grams (bigrams) and use Decision Tree
                bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
                x_train_bigram = bigram_vectorizer.fit_transform(x_train)
                x_test_bigram = bigram_vectorizer.transform(x_test)

                # Train a Decision Tree classifier using bigrams
                dt_classifier_bigram = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=42)

                dt_classifier_bigram.fit(x_train_bigram, y_train)

                # Make predictions on the bigram test set
                predictions_bigram = dt_classifier_bigram.predict(x_test_bigram)

                # Evaluate the accuracy for bigrams
                accuracy_bigram = accuracy_score(y_test, predictions_bigram)
                # results[method]["Bigrams"] = accuracy_bigram

                # Approach 3: Vectorize the text data using Doc2Vec and use Decision Tree
                tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(x_train)]
                doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)
                doc2vec_model.build_vocab(tagged_data)
                doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=10)
                x_train_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_train]
                x_test_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_test]

                # Train a Decision Tree classifier using Doc2Vec
                dt_classifier_doc2vec = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=42)

                dt_classifier_doc2vec.fit(x_train_doc2vec, y_train)

                # Make predictions on the Doc2Vec test set
                predictions_doc2vec = dt_classifier_doc2vec.predict(x_test_doc2vec)

                # Evaluate the accuracy for Doc2Vec
                accuracy_doc2vec = accuracy_score(y_test, predictions_doc2vec)
                # results[method]["Doc2Vec"] = accuracy_doc2vec

                # Approach 4: Apply Topic Modeling (Latent Dirichlet Allocation) and use Decision Tree
                vectorizer_lda = CountVectorizer(stop_words=stop_words)
                x_train_lda = vectorizer_lda.fit_transform(x_train)
                x_test_lda = vectorizer_lda.transform(x_test)
                lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
                x_train_topics = lda_model.fit_transform(x_train_lda)
                x_test_topics = lda_model.transform(x_test_lda)

                # Train a Decision Tree classifier using Topic Modeling features
                dt_classifier_lda = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=42)
                dt_classifier_lda.fit(x_train_topics, y_train)

                # Make predictions on the Topic Modeling test set
                predictions_lda = dt_classifier_lda.predict(x_test_topics)

                # Evaluate the accuracy for Topic Modeling
                accuracy_lda = accuracy_score(y_test, predictions_lda)
                # results[method]["Topic Modeling"] = accuracy_lda

            elif method == "KNN":
                # Train a KNN classifier using TF-IDF
                knn_classifier_tfidf = KNeighborsClassifier(n_neighbors=5)  # Set KNN hyperparameter (e.g., n_neighbors)
                knn_classifier_tfidf.fit(x_train_tfidf, y_train)

                # Make predictions on the TF-IDF test set
                predictions_tfidf = knn_classifier_tfidf.predict(x_test_tfidf)

                # Evaluate the accuracy for TF-IDF
                accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
                # results[method]["TF-IDF"] = accuracy_tfidf

                # Approach 2: Vectorize the text data using n-grams (bigrams) and use KNN
                bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
                x_train_bigram = bigram_vectorizer.fit_transform(x_train)
                x_test_bigram = bigram_vectorizer.transform(x_test)

                # Train a KNN classifier using bigrams
                knn_classifier_bigram = KNeighborsClassifier(
                    n_neighbors=5)  # Set KNN hyperparameter (e.g., n_neighbors)
                knn_classifier_bigram.fit(x_train_bigram, y_train)

                # Make predictions on the bigram test set
                predictions_bigram = knn_classifier_bigram.predict(x_test_bigram)

                # Evaluate the accuracy for bigrams
                accuracy_bigram = accuracy_score(y_test, predictions_bigram)
                # results[method]["Bigrams"] = accuracy_bigram

                # Approach 3: Vectorize the text data using Doc2Vec and use KNN
                tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(x_train)]
                doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)
                doc2vec_model.build_vocab(tagged_data)
                doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=10)
                x_train_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_train]
                x_test_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_test]

                # Train a KNN classifier using Doc2Vec
                knn_classifier_doc2vec = KNeighborsClassifier(
                    n_neighbors=5)  # Set KNN hyperparameter (e.g., n_neighbors)
                knn_classifier_doc2vec.fit(x_train_doc2vec, y_train)

                # Make predictions on the Doc2Vec test set
                predictions_doc2vec = knn_classifier_doc2vec.predict(x_test_doc2vec)

                # Evaluate the accuracy for Doc2Vec
                accuracy_doc2vec = accuracy_score(y_test, predictions_doc2vec)
                # results[method]["Doc2Vec"] = accuracy_doc2vec

                # Approach 4: Apply Topic Modeling (Latent Dirichlet Allocation) and use KNN
                vectorizer_lda = CountVectorizer(stop_words=stop_words)
                x_train_lda = vectorizer_lda.fit_transform(x_train)
                x_test_lda = vectorizer_lda.transform(x_test)
                lda_model = LatentDirichletAllocation(n_components=10,
                                                      random_state=42)  # Set LDA hyperparameters (e.g., n_components)
                x_train_topics = lda_model.fit_transform(x_train_lda)
                x_test_topics = lda_model.transform(x_test_lda)

                # Train a KNN classifier using Topic Modeling features
                knn_classifier_lda = KNeighborsClassifier(n_neighbors=5)  # Set KNN hyperparameter (e.g., n_neighbors)
                knn_classifier_lda.fit(x_train_topics, y_train)

                # Make predictions on the Topic Modeling test set
                predictions_lda = knn_classifier_lda.predict(x_test_topics)

                # Evaluate the accuracy for Topic Modeling
                accuracy_lda = accuracy_score(y_test, predictions_lda)
                # results[method]["Topic Modeling"] = accuracy_lda

            elif method == "MLP":
                # Train an MLP classifier using TF-IDF
                mlp_classifier_tfidf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42,
                                                     activation='relu', solver='adam')  # Set MLP hyperparameters
                mlp_classifier_tfidf.fit(x_train_tfidf, y_train)

                # Make predictions on the TF-IDF test set
                predictions_tfidf = mlp_classifier_tfidf.predict(x_test_tfidf)

                # Evaluate the accuracy for TF-IDF
                accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
                # results[method]["TF-IDF"] = accuracy_tfidf

                # Approach 2: Vectorize the text data using n-grams (bigrams) and use MLP
                bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
                x_train_bigram = bigram_vectorizer.fit_transform(x_train)
                x_test_bigram = bigram_vectorizer.transform(x_test)

                # Train an MLP classifier using bigrams
                mlp_classifier_bigram = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42,
                                                      activation='relu', solver='adam')  # Set MLP hyperparameters
                mlp_classifier_bigram.fit(x_train_bigram, y_train)

                # Make predictions on the bigram test set
                predictions_bigram = mlp_classifier_bigram.predict(x_test_bigram)

                # Evaluate the accuracy for bigrams
                accuracy_bigram = accuracy_score(y_test, predictions_bigram)
                # results[method]["Bigrams"] = accuracy_bigram

                # Approach 3: Vectorize the text data using Doc2Vec and use MLP
                tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(x_train)]
                doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)
                doc2vec_model.build_vocab(tagged_data)
                doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=10)
                x_train_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_train]
                x_test_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_test]

                # Train an MLP classifier using Doc2Vec
                mlp_classifier_doc2vec = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42,
                                                       activation='relu', solver='adam')  # Set MLP hyperparameters
                mlp_classifier_doc2vec.fit(x_train_doc2vec, y_train)

                # Make predictions on the Doc2Vec test set
                predictions_doc2vec = mlp_classifier_doc2vec.predict(x_test_doc2vec)

                # Evaluate the accuracy for Doc2Vec
                accuracy_doc2vec = accuracy_score(y_test, predictions_doc2vec)
                # results[method]["Doc2Vec"] = accuracy_doc2vec

                # Approach 4: Apply Topic Modeling (Latent Dirichlet Allocation) and use MLP
                vectorizer_lda = CountVectorizer(stop_words=stop_words)
                x_train_lda = vectorizer_lda.fit_transform(x_train)
                x_test_lda = vectorizer_lda.transform(x_test)
                lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
                x_train_topics = lda_model.fit_transform(x_train_lda)
                x_test_topics = lda_model.transform(x_test_lda)

                # Train an MLP classifier using Topic Modeling features
                mlp_classifier_lda = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42,
                                                   activation='relu', solver='adam')  # Set MLP hyperparameters
                mlp_classifier_lda.fit(x_train_topics, y_train)

                # Make predictions on the Topic Modeling test set
                predictions_lda = mlp_classifier_lda.predict(x_test_topics)

                # Evaluate the accuracy for Topic Modeling
                accuracy_lda = accuracy_score(y_test, predictions_lda)
                # results[method]["Topic Modeling"] = accuracy_lda

            elif method == "Naive Bayes":

                # Train a Naive Bayes classifier using TF-IDF
                nb_classifier_tfidf = MultinomialNB(alpha=1.0)  # Set Naive Bayes hyperparameters
                nb_classifier_tfidf.fit(x_train_tfidf, y_train)

                # Make predictions on the TF-IDF test set
                predictions_tfidf = nb_classifier_tfidf.predict(x_test_tfidf)

                # Evaluate the accuracy for TF-IDF
                accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
                # results[method]["TF-IDF"] = accuracy_tfidf

                # Approach 2: Vectorize the text data using n-grams (bigrams) and use Naive Bayes
                bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
                x_train_bigram = bigram_vectorizer.fit_transform(x_train)
                x_test_bigram = bigram_vectorizer.transform(x_test)

                # Train a Naive Bayes classifier using bigrams
                nb_classifier_bigram = MultinomialNB(alpha=1.0)  # Set Naive Bayes hyperparameters
                nb_classifier_bigram.fit(x_train_bigram, y_train)

                # Make predictions on the bigram test set
                predictions_bigram = nb_classifier_bigram.predict(x_test_bigram)

                # Evaluate the accuracy for bigrams
                accuracy_bigram = accuracy_score(y_test, predictions_bigram)
                # results[method]["Bigrams"] = accuracy_bigram

                # Approach 3: Vectorize the text data using Doc2Vec and use Naive Bayes
                # Doc2Vec is not suitable for Naive Bayes since it requires non-negative input data

                # Approach 4: Apply Topic Modeling (Latent Dirichlet Allocation) and use Naive Bayes
                vectorizer_lda = CountVectorizer(stop_words=stop_words)
                x_train_lda = vectorizer_lda.fit_transform(x_train)
                x_test_lda = vectorizer_lda.transform(x_test)
                lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
                x_train_topics = lda_model.fit_transform(x_train_lda)
                x_test_topics = lda_model.transform(x_test_lda)

                # Train a Naive Bayes classifier using Topic Modeling features
                nb_classifier_lda = MultinomialNB(alpha=1.0)  # Set Naive Bayes hyperparameters
                nb_classifier_lda.fit(x_train_topics, y_train)

                # Make predictions on the Topic Modeling test set
                predictions_lda = nb_classifier_lda.predict(x_test_topics)

                # Evaluate the accuracy for Topic Modeling
                accuracy_lda = accuracy_score(y_test, predictions_lda)
                # results[method]["Topic Modeling"] = accuracy_lda

            elif method == "SVC":
                # Train an SVC classifier using TF-IDF
                svc_classifier_tfidf = SVC(C=1.0, kernel='rbf', gamma='scale',
                                           random_state=42)  # Set SVC hyperparameters
                svc_classifier_tfidf.fit(x_train_tfidf, y_train)

                # Make predictions on the TF-IDF test set
                predictions_tfidf = svc_classifier_tfidf.predict(x_test_tfidf)

                # Evaluate the accuracy for TF-IDF
                accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
                # results[method]["TF-IDF"] = accuracy_tfidf

                # Approach 2: Vectorize the text data using n-grams (bigrams) and use SVC
                bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
                x_train_bigram = bigram_vectorizer.fit_transform(x_train)
                x_test_bigram = bigram_vectorizer.transform(x_test)

                # Train an SVC classifier using bigrams
                svc_classifier_bigram = SVC(C=1.0, kernel='rbf', gamma='scale',
                                            random_state=42)  # Set SVC hyperparameters
                svc_classifier_bigram.fit(x_train_bigram, y_train)

                # Make predictions on the bigram test set
                predictions_bigram = svc_classifier_bigram.predict(x_test_bigram)

                # Evaluate the accuracy for bigrams
                accuracy_bigram = accuracy_score(y_test, predictions_bigram)
                # results[method]["Bigrams"] = accuracy_bigram

                # Approach 3: Vectorize the text data using Doc2Vec and use SVC
                tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(x_train)]
                doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)
                doc2vec_model.build_vocab(tagged_data)
                doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=10)
                x_train_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_train]
                x_test_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_test]

                # Train an SVC classifier using Doc2Vec
                svc_classifier_doc2vec = SVC(C=1.0, kernel='rbf', gamma='scale',
                                             random_state=42)  # Set SVC hyperparameters
                svc_classifier_doc2vec.fit(x_train_doc2vec, y_train)

                # Make predictions on the Doc2Vec test set
                predictions_doc2vec = svc_classifier_doc2vec.predict(x_test_doc2vec)

                # Evaluate the accuracy for Doc2Vec
                accuracy_doc2vec = accuracy_score(y_test, predictions_doc2vec)
                # results[method]["Doc2Vec"] = accuracy_doc2vec

                # Approach 4: Apply Topic Modeling (Latent Dirichlet Allocation) and use SVC
                vectorizer_lda = CountVectorizer(stop_words=stop_words)
                x_train_lda = vectorizer_lda.fit_transform(x_train)
                x_test_lda = vectorizer_lda.transform(x_test)
                lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
                x_train_topics = lda_model.fit_transform(x_train_lda)
                x_test_topics = lda_model.transform(x_test_lda)

                # Train an SVC classifier using Topic Modeling features
                svc_classifier_lda = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)  # Set SVC hyperparameters
                svc_classifier_lda.fit(x_train_topics, y_train)

                # Make predictions on the Topic Modeling test set
                predictions_lda = svc_classifier_lda.predict(x_test_topics)

                # Evaluate the accuracy for Topic Modeling
                accuracy_lda = accuracy_score(y_test, predictions_lda)
                # results[method]["Topic Modeling"] = accuracy_lda

            else:  # method == "XGBoost":
                # Train an XGBoost classifier using TF-IDF
                xgb_classifier_tfidf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                                     random_state=42)  # Set XGBoost hyperparameters
                xgb_classifier_tfidf.fit(x_train_tfidf, y_train)

                # Make predictions on the TF-IDF test set
                predictions_tfidf = xgb_classifier_tfidf.predict(x_test_tfidf)

                # Evaluate the accuracy for TF-IDF
                accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
                # results[method]["TF-IDF"] = accuracy_tfidf

                # Approach 2: Vectorize the text data using n-grams (bigrams) and use XGBoost
                bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
                x_train_bigram = bigram_vectorizer.fit_transform(x_train)
                x_test_bigram = bigram_vectorizer.transform(x_test)

                # Train an XGBoost classifier using bigrams
                xgb_classifier_bigram = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                                      random_state=42)  # Set XGBoost hyperparameters
                xgb_classifier_bigram.fit(x_train_bigram, y_train)

                # Make predictions on the bigram test set
                predictions_bigram = xgb_classifier_bigram.predict(x_test_bigram)

                # Evaluate the accuracy for bigrams
                accuracy_bigram = accuracy_score(y_test, predictions_bigram)
                # results[method]["Bigrams"] = accuracy_bigram

                # Approach 3: Vectorize the text data using Doc2Vec and use XGBoost
                tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(x_train)]
                doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)
                doc2vec_model.build_vocab(tagged_data)
                doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=10)

                x_train_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_train]
                x_test_doc2vec = [doc2vec_model.infer_vector(doc.split()) for doc in x_test]

                # Train an XGBoost classifier using Doc2Vec
                xgb_classifier_doc2vec = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                                       random_state=42)  # Set XGBoost hyperparameters
                xgb_classifier_doc2vec.fit(x_train_doc2vec, y_train)

                # Make predictions on the Doc2Vec test set
                predictions_doc2vec = xgb_classifier_doc2vec.predict(x_test_doc2vec)

                # Evaluate the accuracy for Doc2Vec
                accuracy_doc2vec = accuracy_score(y_test, predictions_doc2vec)
                # results[method]["Doc2Vec"] = accuracy_doc2vec

                # Approach 4: Apply Topic Modeling (Latent Dirichlet Allocation) and use XGBoost
                vectorizer_lda = CountVectorizer(stop_words=stop_words)
                x_train_lda = vectorizer_lda.fit_transform(x_train)
                x_test_lda = vectorizer_lda.transform(x_test)

                lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
                x_train_topics = lda_model.fit_transform(x_train_lda)
                x_test_topics = lda_model.transform(x_test_lda)

                # Train an XGBoost classifier using Topic Modeling features
                xgb_classifier_lda = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                                   random_state=42)  # Set XGBoost hyperparameters
                xgb_classifier_lda.fit(x_train_topics, y_train)

                # Make predictions on the Topic Modeling test set
                predictions_lda = xgb_classifier_lda.predict(x_test_topics)

                # Evaluate the accuracy for Topic Modeling
                accuracy_lda = accuracy_score(y_test, predictions_lda)
                # results[method]["Topic Modeling"] = accuracy_lda

            # Store the results
            results['Model'].append(method)
            results['TF-IDF'].append(accuracy_tfidf)
            results['Bigrams'].append(accuracy_bigram)
            results['Doc2Vec'].append(accuracy_doc2vec)
            results['Topic Modeling'].append(accuracy_lda)

            print(f"Done {method}!")

        # Display the results
        results_df = pd.DataFrame(results)

        print(results_df)
        with open(keep_results_file, "wb") as f:
            pickle.dump(results, f)

        # Plotting comparing methods
        plt.figure(figsize=(12, 8))

        # Bar plot for Accuracy
        plt.subplot(2, 2, 1)
        sns.barplot(x='Model', y='TF-IDF', data=results_df)
        plt.title('Accuracy TF-IDF Comparison')

        # Bar plot for Precision
        plt.subplot(2, 2, 2)
        sns.barplot(x='Model', y='Bigrams', data=results_df)
        plt.title('Accuracy Bigrams Comparison')

        # Bar plot for Recall
        plt.subplot(2, 2, 3)
        sns.barplot(x='Model', y='Doc2Vec', data=results_df)
        plt.title('Accuracy Doc2Vec Comparison')

        # Bar plot for F1 Score
        plt.subplot(2, 2, 4)
        sns.barplot(x='Model', y='Topic Modeling', data=results_df)
        plt.title('Accuracy Topic Modeling Comparison')

        # Ensure tight layout for better visualization
        plt.tight_layout()

        # Save the generated comparison plot to a file
        plt.savefig(comparison_plot_name)

        # Display the comparison plot
        plt.show()


if __name__ == "__main__":
    main = Main()
