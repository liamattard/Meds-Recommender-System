import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def start(dataset):
    gender_medicine_stacked_chart(dataset)

def gender_medicine_stacked_chart(dataset):
    gender= []
    medicine = []
    for patient in dataset.data[0][0]:
        for med in patient[2]:
            if med < 100:
                gender.append(dataset.voc[0]["gender_voc"].idx2word[patient[5]])
                medicine.append(med)
    
    data = {"Gender": gender, "Medication ID": medicine}

    df = pd.DataFrame(data)

    # Count the number of occurrences of each medication ID for each gender
    medication_counts = df.groupby(['Medication ID', 'Gender']).size().reset_index(name='Counts')

    # Pivot the data to create a stacked bar chart
    medication_counts_pivot = medication_counts.pivot(index='Medication ID', columns='Gender', values='Counts')

    # Plot the stacked bar chart
    medication_counts_pivot.plot(kind='bar', stacked=True)

    # Add a title to the plot
    plt.title("Distribution of Medication ID by Gender")

    # Add a label for the x axis
    plt.xlabel("Medication ID")

    # Add a label for the y axis
    plt.ylabel("Count")

    # Show the plot
    plt.show()

def gender_medicine_bar_chart(dataset):
    gender= []
    medicine = []
    for patient in dataset.data[0][0]:
        for med in patient[2]:
            if med < 100:
                gender.append(dataset.voc[0]["gender_voc"].idx2word[patient[5]])
                medicine.append(med)
    
    data = {"Gender": gender, "Medication ID": medicine}
    df = pd.DataFrame(data)
    medication_counts = df.groupby(['Gender', 'Medication ID']).size().reset_index(name='Counts')

    male_counts = medication_counts[medication_counts['Gender'] == 'M']['Counts']
    female_counts = medication_counts[medication_counts['Gender'] == 'F']['Counts']

    breakpoint()
    fig, ax = plt.subplots()

    # medicine_ids = list(dataset.voc[0]['med_voc'].idx2word.keys())
    medicine_ids = list(range(100))

    rects1 = ax.bar(medicine_ids, male_counts, label='Male')
    rects2 = ax.bar(medicine_ids, female_counts, label='Female')

    # Add a title to the plot
    ax.set_title("Distribution of Medication ID by Gender")

    # Add a label for the x axis
    ax.set_xlabel("Medication ID")

    # Add a label for the y axis
    ax.set_ylabel("Count")

    # Add a legend to the plot
    ax.legend()

    # Show the plot
    plt.show()


    # breakpoint()
    # sns.violinplot(x="Medications", y="Gender", data=df)
    # table = pd.crosstab(gender, medicine)
    # chi2, p, dof, expected = chi2_contingency(table)
    # print("The chi-squared test statistic is {}".format(chi2))
    # print("The p-value is {}".format(p))

    # # Add labels and title to the plot
    # plt.xlabel("Medicine")
    # plt.ylabel("Gender")
    # plt.title("Medication Count by Gender")

    # Show the plot
    plt.show()



def insurance_medicine(dataset):
    insurance = []
    medicine = []
    for patient in dataset.data[0][0]:
        for med in patient[2]:
            insurance.append(dataset.voc[0]["insurance_voc"].idx2word[patient[6]])
            medicine.append(med)

    table = pd.crosstab(insurance, medicine)
    chi2, p, dof, expected = chi2_contingency(table)
    print("The chi-squared test statistic is {}".format(chi2))
    print("The p-value is {}".format(p))


    # Plot the heatmap using seaborn
    # sns.countplot(x="Insurance", hue="Medication", data=df)
    sns.heatmap(table, annot=False, fmt='d', cmap="YlOrRd", cbar=False, vmax=100)

    # Add labels and title to the plot
    plt.xlabel("Medicine")
    plt.ylabel("Count")
    plt.title("Medication Count by Insurance Plan")

    # Show the plot
    plt.show()


    # sns.heatmap(table, annot=True, cmap='Blues')

    # plt.xlabel('Medication')
    # plt.ylabel('Insurance')
    # plt.title('Heatmap of Insurance and Medication')
    # plt.show()


def age_medicine_count(dataset):

    age_med_count_map = dict()

    for patient in dataset.data[0][0]:
        if patient[3] not in age_med_count_map:
            age_med_count_map[patient[3]] = []

        age_med_count_map[patient[3]].append(len(patient[2]))
    
    for age in age_med_count_map:
        age_med_count_map[age] = sum(age_med_count_map[age])/len(age_med_count_map[age])



    age = [i for i in age_med_count_map]
    medicine_count  = [age_med_count_map[i] for i in age_med_count_map]
    data = {"Age": age, "Medication_Count": medicine_count}
    df = pd.DataFrame(data)

    sns.scatterplot(x='Age', y='Medication_Count', data=df)
    sns.regplot(x='Age', y='Medication_Count', data=df)

    corr, p_value = pearsonr(df['Age'], df['Medication_Count'])
    print(f'The Pearson correlation coefficient between Age and Medications is {corr:.3f} with a p-value of {p_value:.3f}')



    rho, p_value = stats.spearmanr(age, medicine_count)

    # Print the results
    print("The Spearman's rank correlation coefficient is", rho)
    print("The p-value is", p_value)

    # plt.show()


def age_dist_histogram(dataset):
    age = []
    for patient in dataset.data[0][0]:
        true_age = dataset.voc[0]["age_voc"].idx2word[patient[3]]
        if true_age > 90:
            true_age = 90
        age.append(true_age)


    data = {"Age": age}
    df = pd.DataFrame(data)

    plt.hist(df['Age'], bins=20, edgecolor='black')
    plt.title("Distribution of Age")
    plt.xlabel("Age")
    plt.ylabel("Frequency")


    plt.show()


def age_category_medicine_violin(dataset):
    age = []
    medicine = []
    for patient in dataset.data[0][0]:
        true_age = dataset.voc[0]["age_voc"].idx2word[patient[3]]
        if true_age < 14:
            age_category = "Child (0-14)"
        elif true_age < 24:
            age_category = "Youth (15-24)"
        elif true_age < 64:
            age_category = "Adult (25-64)"
        if true_age > 90:
            age_category = "Seniors 65+"
        for med in patient[2]:
            if med < 100:
                age.append(age_category)
                medicine.append(med)

    data = {"Age": age, "Medications": medicine}
    table = pd.crosstab(age, medicine)
    breakpoint()
    sns.heatmap(table, annot=False, fmt='d', cmap="YlOrRd", cbar=False, vmax=1000)

    # Add labels and title to the plot
    plt.xlabel("Medicine")
    plt.ylabel("Age")
    plt.title("Medication Count by Age Category")

    # Show the plot
    plt.show()



def age_medicine(dataset):
    age = []
    medicine = []
    for patient in dataset.data[0][0]:
        for med in patient[2]:
            age.append(patient[3])
            medicine.append(med)


    data = {"Age": age, "Medications": medicine}
    df = pd.DataFrame(data)


    sns.scatterplot(x='Age', y='Medications', data=df)
    sns.regplot(x='Age', y='Medications', data=df)
    plt.show()

    corr, p_value = pearsonr(df['Age'], df['Medications'])
    print(f'The Pearson correlation coefficient between Age and Medications is {corr:.3f} with a p-value of {p_value:.3f}')



