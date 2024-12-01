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

sensible = []
privilege = []
unprivilege = []
df_original = None
methode = []

def takeDataframe(name, encoding):
    # Recuperation des fichier csv en dataframe
    dossier = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Dataset'))
    chemin = os.path.join(dossier, name)
    dataframe = pd.read_csv(chemin,sep=';', encoding=encoding)
    if( name == 'bank-full.csv'):
        month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        dataframe['month_numeric'] = dataframe['month'].map(month_mapping)
        dataframe.drop(columns=['month'],inplace= True)
    return dataframe

def numerique(dataframe) :
    # Numerise student-por
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
def numeric(dataframe) :
    # Numerise bank-full
    dataframeN = dataframe
    job_mapping = {'student':0,'unemployed':1, 'unknown':2, 'housemaid':3, 'blue-collar':4, 'services':5, 'retired':6, 'self-employed':7, 'technician':8 ,'management':9,'entrepreneur':10,'admin.':11 }
    dataframeN = dataframeN.assign(jobN = dataframe['job'].map(job_mapping))
    marital_mapping = {'divorced':0,'single':1,'married':2}
    dataframeN = dataframeN.assign(maritalN = dataframe['marital'].map(marital_mapping))
    education_mapping = {'primary':0,'unknown':1,'secondary':2,'tertiary':3}
    dataframeN = dataframeN.assign(educationN = dataframe['education'].map(education_mapping))
    yesno_mapping = {'yes': 1,'no':0}
    dataframeN = dataframeN.assign(defaultN = dataframe['default'].map(yesno_mapping))
    dataframeN = dataframeN.assign(housingN = dataframe['housing'].map(yesno_mapping))
    dataframeN = dataframeN.assign(loanN = dataframe['loan'].map(yesno_mapping))
    dataframeN = dataframeN.assign(yN = dataframe['y'].map(yesno_mapping))
    contact_mapping = {'unknown':0,'telephone':1,'cellular':2}
    dataframeN = dataframeN.assign(contactN = dataframe['contact'].map(contact_mapping))
    #poutcome_mapping = {'unknown':0,'failure':1,'other':2,'success':3}
    #dataframeN = dataframeN.assign(poutcomeN = dataframe['job'].map(poutcome_mapping))

    dataframeN.drop(columns=['job','marital','education','default','housing','loan','contact','poutcome','y'],inplace= True)

    return dataframeN

def metric_privilege(dataframe):
    #Creation des objects nous permettant de recuper les metrics voulues
    metric = []
    for x in range(len(sensible)) :
        dataset = BinaryLabelDataset(df=dataframe, label_names=['yN'], protected_attribute_names=[sensible[x]])
        dataset_original = BinaryLabelDataset(df=df_original, label_names=['yN'], protected_attribute_names=[sensible[x]])
        unp_group = [{sensible[x]: v} for v in unprivilege[x]]
        p_group = [{sensible[x]: v} for v in privilege[x]]
        metric.append(ClassificationMetric(dataset_original, dataset,unprivileged_groups=unp_group,privileged_groups=p_group))
    return metric

def test(x_train, x_test, y_train, y_test) :
    global df_original
    df_original = pd.DataFrame(x_test)
    df_original ['yN'] = y_test

    # Normalise data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(x_train, y_train)
    y_pred = nb_model.predict(x_test)
    df_Multinnomina_bayes = pd.DataFrame(x_test)
    df_Multinnomina_bayes['yN'] = y_pred
    metric_bayes = metric_privilege(df_Multinnomina_bayes)

    # Logistic Regression
    regression_model = LogisticRegression()
    regression_model.fit(x_train_scaled, y_train)
    y_pred_regression = regression_model.predict(x_test_scaled)
    df_Reg = pd.DataFrame(x_test)
    df_Reg['yN'] = y_pred_regression
    metric_reg = metric_privilege(df_Reg)

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=20)
    knn_model.fit(x_train_scaled, y_train)
    y_pred_knn = knn_model.predict(x_test_scaled)
    df_knn = pd.DataFrame(x_test)
    df_knn['yN'] = y_pred_knn
    metric_knn = metric_privilege(df_knn)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(x_train_scaled, y_train)
    y_pred_rf=rf_model.predict(x_test_scaled)
    df_rf = pd.DataFrame(x_test)
    df_rf['yN'] = y_pred_rf
    metric_rf = metric_privilege(df_rf)

    # Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=1)
    nn_model.fit(x_train_scaled, y_train)
    y_pred_nn = nn_model.predict(x_test_scaled)
    df_nn = pd.DataFrame(x_test)
    df_nn['yN'] = y_pred_nn
    metric_nn = metric_privilege(df_nn)

    # SVM
    svm_model = SVC()
    svm_model.fit(x_train, y_train)
    y_pred_svm = svm_model.predict(x_test)
    df_svm = pd.DataFrame(x_test)
    df_svm['yN'] = y_pred_svm
    metric_svm = metric_privilege(df_svm)

    resultat = [metric_bayes,metric_reg,metric_knn,metric_rf,metric_nn,metric_svm]
    return resultat

def multiple(dataframe, n):
    # lance les test n fois
    y = dataframe['yN']
    x = dataframe.drop(columns=['yN'])

    for column in x.columns:
        x[column] = x[column].astype('category').cat.codes
    memoire = [0] * n
    for m in range(n):
        tmp = []
        random_state = np.random.randint(0, 1000)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=random_state)
        tmp = (test(x_train, x_test, y_train, y_test))
        memoire[m] = tmp
    return memoire

def moyenne(liste):
    # Recherche des statisques + moyenne
    taille = len(liste)
    resultat = [0]*len(liste[0])
    somme_carre = [0]*len(liste[0])
    n=0
    for x in liste[0] :
        resultat[n] = [0]*len(x)
        somme_carre[n] = [0]*len(x)
        for y in range(len(sensible)):
            resultat[n][y] = [0]*4
            somme_carre[n][y] = [0]*4
        n = n+1
    for x in liste:
        m = 0
        for y in x :
            o = 0
            for z in y :
                resultat[m][o][0] = resultat[m][o][0] + (z.disparate_impact()/taille)
                resultat[m][o][1] = resultat[m][o][1] + (z.equal_opportunity_difference()/taille)
                resultat[m][o][2] = resultat[m][o][2] + (z.accuracy()/taille)
                resultat[m][o][3] = resultat[m][o][3] + (z.error_rate_difference()/taille)

                somme_carre[m][o][0] += (z.disparate_impact() ** 2) / taille
                somme_carre[m][o][1] += (z.equal_opportunity_difference() ** 2) / taille
                somme_carre[m][o][2] += (z.accuracy() ** 2) / taille
                somme_carre[m][o][3] += (z.error_rate_difference() ** 2) / taille
                o = o+1
            m = m+1

    ecart_type = [0]*len(liste[0])
    for n in range(len(resultat)):
        ecart_type[n] = [0]*len(resultat[n])
        for y in range(len(resultat[n])):
            ecart_type[n][y] = [0]*4
            for stat in range(4):
                variance = somme_carre[n][y][stat] - (resultat[n][y][stat] ** 2)
                ecart_type[n][y][stat] = math.sqrt(variance)

    return resultat, ecart_type

def plot_bar(test_name, x, y, stdev, bidule):
    plt.figure()
    plt.bar(x, y, yerr=stdev, width=0.4)
    plt.xlabel("Methode")
    plt.ylabel(test_name)
    plt.title("Methode " + test_name)
    plt.savefig(f"../Graph/{bidule}_{test_name}.png")
    plt.close()

def main ():
    # Modification et Creation de donnée pour base de donnée bank-full
    df_bank = takeDataframe('bank-full.csv', 'utf-8')
    df_bankN = numeric(df_bank)
    sensible.extend(['maritalN','educationN'])
    privilege.extend([[2],[3]])
    unprivilege.extend([[0, 1], [0, 1, 2]])

    # Test sur bank-full
    test = multiple(df_bankN,10)
    result = moyenne(test)

    # Tableau avec les resultats
    methode.extend(["naive_bayes", "logistic_regression", "k_neighbors", "random_forest", "neural_network", "support_vector_machine"])
    columns = ["Methode_sensible",'Statistical Parity Difference', 'Disparate Impact', 'Generalized Equalized Odds Difference',
               'Equal Opportunity Difference', 'False Positive Rate', 'Accuracy', 'Error Rate Difference']
    res_final=[]
    res_final.append(columns)
    for s in range(len(sensible)):
        for m in range(len(methode)):
            rows = []
            row_name = f"{methode[m]}_{sensible[s]}"
            rows.append(row_name)
            rows.extend(result[m][s])
            res_final.append(rows)

    for x in range(len(res_final)):
        print(res_final[x])

def maindeux ():
    # Modification et Creation de donnée pour base de donnée student-por
    df_student = takeDataframe('student-por.csv', 'utf-8')
    df_studentN = numerique(df_student)
    global sensible, privilege, unprivilege, methode
    sensible = (['sexN','age'])
    privilege = ([[1],[15,16,17,18]])
    unprivilege = ([[0],[19,20,21,22]])

    # Test sur student-por
    test = multiple(df_studentN,10)
    result, stdev = moyenne(test)

    # Tableau avec les resultats
    methode = (["naive_bayes", "logistic_regression", "k_neighbors", "random_forest", "neural_network", "support_vector_machine"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference']

    print("Moyenne :")
    res_final=[]
    res_final.append(columns)
    for s in range(len(sensible)):
        for m in range(len(methode)):
            rows = []
            row_name = f"{methode[m]}_{sensible[s]}"
            rows.append(row_name)
            rows.extend(result[m][s])
            res_final.append(rows)

    for x in range(len(res_final)):
        print(res_final[x])

    #print("Stdev :")
    #stdev_final=[]
    #stdev_final.append(columns)
    #for s in range(len(sensible)):
    #    for m in range(len(methode)):
    #        rows = []
    #        row_name = f"{methode[m]}_{sensible[s]}"
    #        rows.append(row_name)
    #        rows.extend([(stdev[m][s][i] / result[m][s][i]) * 100 if result[m][s][i] != 0 else 0 for i in range(len(result[m][s]))])
    #        stdev_final.append(rows)


    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result[i][s][m] for i in range(len(methode))]
            standev = [stdev[i][s][m] for i in range(len(methode))]
            plot_bar(columns[m+1],methode,y,standev,sensible[s])

    #for x in range(len(stdev_final)):
    #    print(stdev_final[x])

#main()
maindeux()
