import logging
import pandas as pd
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

logging.basicConfig(
    filename='process_log.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

sensible = []
privilege = []
unprivilege = []
df_original = None
methode = []

def takeDataframe(name, encoding):
    logging.info(f"Attempting to read the dataset: {name}")
    try:
        # Reading the CSV file into dataframe
        folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Dataset'))
        path = os.path.join(folder, name)
        dataframe = pd.read_csv(path, sep=';', encoding=encoding)
        if name == 'bank-full.csv':
            month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            dataframe['month_numeric'] = dataframe['month'].map(month_mapping)
            dataframe.drop(columns=['month'], inplace=True)
        logging.info(f"Dataset {name} loaded successfully.")
        return dataframe
    except Exception as e:
        logging.error(f"Error while reading dataset {name}: {e}")
        raise

def numerique(dataframe) :
    # Numerise student-por
    logging.info("Starting to convert categorical variables into numeric codes.")
    try:
        dataframeN = dataframe
        school_mapping = {"GP":0, "MS":1}
        dataframeN = dataframeN.assign(schoolN = dataframe['school'].map(school_mapping))
        sex_mapping = {"F":0, "M":1}
        dataframeN = dataframeN.assign(sexN = dataframe['sex'].map(sex_mapping))
        address_mapping = {"U":0,"R":1}
        dataframeN = dataframeN.assign(addressN = dataframe['address'].map(address_mapping))
        famsize_mapping = {"LE3":0, "GT3":1}
        dataframeN = dataframeN.assign(famsizeN = dataframe['famsize'].map(famsize_mapping))
        Pstatus_mapping = {"T":0, "A":1}
        dataframeN = dataframeN.assign(PstatusN = dataframe['Pstatus'].map(Pstatus_mapping))
        job_mapping = {"teacher":0, "health":1, "services":2, "at_home":3, "other":4}
        dataframeN = dataframeN.assign(MjobN = dataframe['Mjob'].map(job_mapping))
        dataframeN = dataframeN.assign(FjobN = dataframe['Fjob'].map(job_mapping))
        reason_mapping = {"home":0, "reputation":1, "course":2, "other":3}
        dataframeN = dataframeN.assign(reasonN = dataframe['reason'].map(reason_mapping))
        guard_mapping = {"mother":0, "father":1, "other":2}
        dataframeN = dataframeN.assign(guardianN = dataframe['guardian'].map(guard_mapping))
        yesno_mapping = {"no":0, "yes":1}
        dataframeN = dataframeN.assign(schoolsupN = dataframe['schoolsup'].map(yesno_mapping))
        dataframeN = dataframeN.assign(famsupN = dataframe['famsup'].map(yesno_mapping))
        dataframeN = dataframeN.assign(paidN = dataframe['paid'].map(yesno_mapping))
        dataframeN = dataframeN.assign(activitiesN = dataframe['activities'].map(yesno_mapping))
        dataframeN = dataframeN.assign(nurseryN = dataframe['nursery'].map(yesno_mapping))
        dataframeN = dataframeN.assign(higherN = dataframe['higher'].map(yesno_mapping))
        dataframeN = dataframeN.assign(internetN = dataframe['internet'].map(yesno_mapping))
        dataframeN = dataframeN.assign(romanticN = dataframe['romantic'].map(yesno_mapping))
        value_mapping = {str(i): i for i in range(21)}
        #dataframeN = dataframeN.assign(G1N = dataframe['G1'].map(value_mapping))
        #dataframeN = dataframeN.assign(G2N = dataframe['G2'].map(value_mapping))
        dataframeN = dataframeN.assign(yN=dataframe['G3'].apply(lambda x: 0 if x <= 9 else 1))


        dataframeN.drop(columns=[
            'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
            'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
            'higher', 'internet', 'romantic', 'G3'
        ], inplace=True)

        return dataframeN
    except Exception as e:
        logging.error(f"Error during conversion of categorical variables: {e}")
        raise


def numeric(dataframe):
    logging.info("Starting 'numeric' function to convert categorical data to numeric.")
    
    try:
        dataframeN = dataframe.copy()  # Avoid modifying original dataframe
        logging.info("Created a copy of the original dataframe.")

        # Job mapping
        job_mapping = {'student':0,'unemployed':1, 'unknown':2, 'housemaid':3, 'blue-collar':4, 'services':5, 'retired':6, 'self-employed':7, 'technician':8 ,'management':9,'entrepreneur':10,'admin.':11 }
        logging.info("Mapping 'job' column to numeric values.")
        dataframeN = dataframeN.assign(jobN = dataframe['job'].map(job_mapping))

        # Marital status mapping
        marital_mapping = {'divorced':0,'single':1,'married':2}
        logging.info("Mapping 'marital' column to numeric values.")
        dataframeN = dataframeN.assign(maritalN = dataframe['marital'].map(marital_mapping))

        # Education mapping
        education_mapping = {'primary':0,'unknown':1,'secondary':2,'tertiary':3}
        logging.info("Mapping 'education' column to numeric values.")
        dataframeN = dataframeN.assign(educationN = dataframe['education'].map(education_mapping))

        # Yes/No mapping for binary columns
        yesno_mapping = {'yes': 1,'no':0}
        logging.info("Mapping binary columns ('default', 'housing', 'loan', 'y') to numeric values.")
        dataframeN = dataframeN.assign(defaultN = dataframe['default'].map(yesno_mapping))
        dataframeN = dataframeN.assign(housingN = dataframe['housing'].map(yesno_mapping))
        dataframeN = dataframeN.assign(loanN = dataframe['loan'].map(yesno_mapping))
        dataframeN = dataframeN.assign(yN = dataframe['y'].map(yesno_mapping))

        # Contact mapping
        contact_mapping = {'unknown':0,'telephone':1,'cellular':2}
        logging.info("Mapping 'contact' column to numeric values.")
        dataframeN = dataframeN.assign(contactN = dataframe['contact'].map(contact_mapping))

        # Drop original columns after transformation
        logging.info("Dropping original categorical columns.")
        dataframeN.drop(columns=['job','marital','education','default','housing','loan','contact','poutcome','y'], inplace=True)

        logging.info("Data conversion complete, returning transformed dataframe.")
        return dataframeN
    
    except Exception as e:
        logging.error(f"Error during 'numeric' function: {e}")
        raise

def metric_privilege(dataframe):
    logging.info("Starting 'metric_privilege' function.")
    
    try:
        metric = []
        # Loop through sensitive attributes
        for x in range(len(sensible)):
            logging.info(f"Processing sensitive attribute: {sensible[x]}")
            
            # Creating BinaryLabelDataset objects for current sensitive attribute
            dataset = BinaryLabelDataset(df=dataframe, label_names=['yN'], protected_attribute_names=[sensible[x]])
            dataset_original = BinaryLabelDataset(df=df_original, label_names=['yN'], protected_attribute_names=[sensible[x]])

            # Log group creation
            unp_group = [{sensible[x]: v} for v in unprivilege[x]]
            p_group = [{sensible[x]: v} for v in privilege[x]]
            logging.info(f"Unprivileged group: {unp_group}, Privileged group: {p_group}")
            
            # Create the ClassificationMetric object
            metric_object = ClassificationMetric(dataset_original, dataset, unprivileged_groups=unp_group, privileged_groups=p_group)
            metric.append(metric_object)
        
        logging.info("Finished processing all sensitive attributes and created metrics.")
        return metric

    except Exception as e:
        logging.error(f"Error in 'metric_privilege' function: {e}")
        raise

def test(x_train, x_test, y_train, y_test) :
    logging.info("Starting model testing.")
    try:
        models = {
            "Logistic Regression": LogisticRegression(),
            "Naive Bayes": MultinomialNB(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier(),
            "MLP": MLPClassifier()
        }
        
        results = {}
        for model_name, model in models.items():
            logging.info(f"Training and testing {model_name}")
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            results[model_name] = accuracy
            logging.info(f"{model_name} accuracy: {accuracy}")
        
        logging.info("Model testing completed.")
        return results
    except Exception as e:
        logging.error(f"Error during model testing: {e}")
        raise

def multiple(dataframe, n):
    logging.info(f"Starting 'multiple' function with {n} iterations.")
    
    try:
        y = dataframe['yN']
        x = dataframe.drop(columns=['yN'])

        # Convert categorical columns to numeric codes
        for column in x.columns:
            x[column] = x[column].astype('category').cat.codes
        
        logging.info(f"Dataframe preprocessing completed. Columns: {x.columns}.")
        
        memoire = [0] * n
        for m in range(n):
            logging.info(f"Running test iteration {m+1} out of {n}.")
            tmp = []
            random_state = np.random.randint(0, 1000)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=random_state)
            tmp = (test(x_train, x_test, y_train, y_test))  # Assuming 'test' is defined elsewhere
            memoire[m] = tmp
        
        logging.info("Completed all test iterations.")
        return memoire

    except Exception as e:
        logging.error(f"Error in 'multiple' function: {e}")
        raise


def moyenne(liste):
    logging.info("Starting 'moyenne' function to calculate statistics and averages.")
    
    try:
        taille = len(liste)
        resultat = [0]*len(liste[0])
        somme_carre = [0]*len(liste[0])
        n = 0
        logging.info(f"Initializing result and sum of squares arrays of size {len(liste[0])}.")
        
        # Initialize the result and sum of squares
        for x in liste[0]:
            resultat[n] = [0]*len(x)
            somme_carre[n] = [0]*len(x)
            for y in range(len(sensible)):
                resultat[n][y] = [0]*4
                somme_carre[n][y] = [0]*4
            n = n+1

        # Process the data
        logging.info("Processing data in 'liste'.")
        for x in liste:
            m = 0
            for y in x:
                o = 0
                for z in y:
                    # Accumulate the statistics
                    resultat[m][o][0] += z.disparate_impact() / taille
                    resultat[m][o][1] += z.equal_opportunity_difference() / taille
                    resultat[m][o][2] += z.accuracy() / taille
                    resultat[m][o][3] += z.error_rate_difference() / taille

                    somme_carre[m][o][0] += (z.disparate_impact() ** 2) / taille
                    somme_carre[m][o][1] += (z.equal_opportunity_difference() ** 2) / taille
                    somme_carre[m][o][2] += (z.accuracy() ** 2) / taille
                    somme_carre[m][o][3] += (z.error_rate_difference() ** 2) / taille
                    o += 1
                m += 1

        # Calculate standard deviation
        ecart_type = [0]*len(liste[0])
        logging.info("Calculating standard deviation for each statistic.")
        for n in range(len(resultat)):
            ecart_type[n] = [0]*len(resultat[n])
            for y in range(len(resultat[n])):
                ecart_type[n][y] = [0]*4
                for stat in range(4):
                    variance = somme_carre[n][y][stat] - (resultat[n][y][stat] ** 2)
                    ecart_type[n][y][stat] = math.sqrt(variance)

        logging.info("Standard deviation calculation completed successfully.")
        return resultat, ecart_type
    
    except Exception as e:
        logging.error(f"Error in 'moyenne' function: {e}")
        raise

def plot_bar(test_name, x, y, stdev, bidule):
    logging.info(f"Starting plot_bar function for {test_name} with {bidule}.")
    
    try:
        # Create the plot
        logging.info(f"Creating bar chart for {test_name}.")
        plt.figure()
        plt.bar(x, y, yerr=stdev, width=0.4)
        plt.xlabel("Methode")
        plt.ylabel(test_name)
        plt.title(f"Methode {test_name}")

        # Save the plot to file
        save_path = f"../Graph/{bidule}_{test_name}.png"
        plt.savefig(save_path)
        logging.info(f"Bar chart saved to {save_path}.")

        plt.close()  # Close the plot to avoid memory overload
        logging.info("Plot closed.")
    
    except Exception as e:
        logging.error(f"Error in plot_bar function: {e}")
        raise

def main ():
    logging.info("Main function started.")
    try:
        # Load and prepare data
        df_original = takeDataframe("bank-full.csv", encoding='utf-8')
        df_processed = numerique(df_original)
        
        # Data preprocessing
        df_scaled = preprocessing(df_processed)

        # Split data into train and test
        X = df_scaled.drop('y', axis=1)
        y = df_scaled['y']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Test models
        results = test(x_train, x_test, y_train, y_test)
        logging.info(f"Test results: {results}")

        logging.info("Main function completed successfully.")
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

def maindeux ():
    logging.info("Starting maindeux function.")
    try:
        # Modification et Creation de donnée pour base de donnée student-por
        logging.info("Loading 'student-por.csv' dataset.")
        df_student = takeDataframe('student-por.csv', 'utf-8')
        df_studentN = numerique(df_student)
        logging.info("'student-por.csv' dataset loaded and converted successfully.")

        # Initialize global variables for sensitive, privileged, and unprivileged
        global sensible, privilege, unprivilege, methode
        sensible = ['sexN', 'age']
        privilege = [[1], [15, 16, 17, 18]]
        unprivilege = [[0], [19, 20, 21, 22]]
        logging.info("Sensitive attributes, privileged, and unprivileged groups initialized.")

        # Test on student-por dataset
        logging.info("Running multiple function on student-por data.")
        test = multiple(df_studentN, 10)
        result, stdev = moyenne(test)
        logging.info("Test completed. Results calculated.")

        # Method names for the models
        methode = ["naive_bayes", "logistic_regression", "k_neighbors", "random_forest", "neural_network", "support_vector_machine"]
        columns = ["Methode_sensible", 'Disparate Impact', 'Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference']
        logging.info("Model methods and columns defined.")

        # Prepare the final results table
        logging.info("Preparing results table for output.")
        res_final = []
        res_final.append(columns)
        for s in range(len(sensible)):
            for m in range(len(methode)):
                rows = []
                row_name = f"{methode[m]}_{sensible[s]}"
                rows.append(row_name)
                rows.extend(result[m][s])
                res_final.append(rows)
        logging.info("Results table prepared.")

        # Print the results
        logging.info("Printing results table.")
        for x in range(len(res_final)):
            logging.info(res_final[x])  # Log each row of the final results

        # Plot bar charts for each sensitive attribute
        logging.info("Generating bar charts for each sensitive attribute.")
        for s in range(len(sensible)):
            for m in range(len(columns)-1):
                y = [result[i][s][m] for i in range(len(methode))]
                standev = [stdev[i][s][m] for i in range(len(methode))]
                plot_bar(columns[m+1], methode, y, standev, sensible[s])
        logging.info("Bar charts generated.")

        logging.info("maindeux function completed successfully.")
    except Exception as e:
        logging.error(f"Error in maindeux function: {e}")
        raise

maindeux()
