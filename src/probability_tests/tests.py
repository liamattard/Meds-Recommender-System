import pandas as pd
import collections
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def start(dataset):
    # gender_medicine_stacked_chart(dataset)
    # age_category_medicine_violin_three(dataset)
    # age_category_medicine_violin(dataset)
    # age_medicine(dataset)
    insurance_medicine(dataset)
    # gender_medicine_bar_char_two(dataset)

def gender_medicine_bar_char_two(dataset):
    n = None
    # m = 100

    med_list = list(dataset.voc[0]["med_voc"].idx2word.keys())

    med_len = len(med_list)

    x = np.zeros((2, med_len), dtype='int32')
    for patient in dataset.data[0][0]:
        for med in patient[2]:

            # if med > n - 1:
            #     continue

            # if med < m:
            #     continue

            # med = med -100
            
            gender = dataset.voc[0]["gender_voc"].idx2word[patient[5]]
            if gender == 'F':
                x[0,med] = x[0,med] + 1
            elif gender =='M':
                x[1,med] = x[1,med] + 1
            # data[gender].append(med)

    colors = ['#1f77b4', '#ff7f0e']
    fig, ax = plt.subplots()
    breakpoint()
    ax.bar(med_list, x[:][0], color=colors[0], label='Female')
    ax.bar(med_list, x[:][1], bottom=x[:][0], color=colors[1], label='Male')

    for i, j in enumerate(x[0]):

        if x[1,i] == 0 or x[0,i] == 0:
            ax.get_children()[i].set_color('black')
        else:
            if x[1,i] > j: 
                ax.get_children()[i].set_color(colors[1])
            else:
                ax.get_children()[i].set_color(colors[0])
        
        ax.get_children()[i].set_height(2)

    # Add labels and title
    ax.set_xlabel('Medicine Type')
    ax.set_ylabel('Proportion of Prescriptions')
    ax.set_title('Medicine Prescriptions by Gender')

    # Add a legend
    ax.legend(loc='upper left')

    plt.show()

    breakpoint()

def gender_medicine_stacked_chart(dataset):

    gender= []
    medicine = []
    for patient in dataset.data[0][0]:
        for med in patient[2]:
            # if med < 100:
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
            # if med < 17:
            gender.append(dataset.voc[0]["gender_voc"].idx2word[patient[5]])
            medicine.append(med)
    

    gender.append('M')
    medicine.append(178)

    gender.append('M')
    medicine.append(179)

    gender.append('M')
    medicine.append(179)

    data = {"Gender": gender, "Medication ID": medicine}
    df = pd.DataFrame(data)
    medication_counts = df.groupby(['Gender', 'Medication ID']).size().reset_index(name='Counts')

    male_counts = medication_counts[medication_counts['Gender'] == 'M']['Counts']
    # male_counts.drop('index', inplace=True, axis=1)
    female_counts = medication_counts[medication_counts['Gender'] == 'F']['Counts']

    fig, ax = plt.subplots()

    breakpoint()
    # medicine_ids = list(dataset.voc[0]['med_voc'].idx2word.keys())
    medicine_ids = list(range(180))

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
            insurance.append(dataset.voc[0]["gender_voc"].idx2word[patient[5]])
            medicine.append(med)

    table = pd.crosstab(insurance, medicine)
    chi2, p, dof, expected = chi2_contingency(table)
    print("The chi-squared test statistic is {}".format(chi2))
    print("The p-value is {}".format(p))


    # Plot the heatmap using seaborn
    # sns.countplot(x="Insurance", hue="Medication", data=df)
    sns.heatmap(table, annot=False, fmt='d', cmap="YlOrRd", cbar=False, vmax=3)

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

    plt.show()

def age_dist_histogram(dataset):

    # Child (0-14)
    child = 0

    # Youth (15-24)
    youth = 0

    # Adult (25-64)
    adult = 0

    # Seniors 65+
    seniors = 0

    all_ages = {}
    all_ages[90] = 0

    for patient in dataset.data[0][0]:
        true_age = dataset.voc[0]["age_voc"].idx2word[patient[3]]

        if true_age < 16:
            continue

        if true_age > 90:
            all_ages[90] += 1
        else:
            if true_age in all_ages:
                all_ages[true_age] += 1
            else:
                all_ages[true_age] = 1

        if true_age > 16:
            # child = child + 1
            continue
        elif true_age < 24:
            youth =  youth + 1
        elif true_age < 64:
            adult = adult + 1
        else:
        # if true_age > 90:
            seniors = seniors + 1

    # age = []
    # for patient in dataset.data[0][0]:
    #     true_age = dataset.voc[0]["age_voc"].idx2word[patient[3]]
    #     if true_age > 90:
    #         true_age = 90
    #     age.append(true_age)

    print("This is child: ", child)
    data = {"Youth (15-24)": youth,
            "Adult (25-64)": adult, "Senior 65+": seniors}
    # df = pd.DataFrame(data)

    breakpoint()
    # plt.bar(list(data.keys()), list(data.values()),  edgecolor='black')
    plt.bar(list(all_ages.keys()), list(all_ages.values()),  edgecolor='black')
    plt.title("")
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



def age_category_medicine_violin_two(dataset):

    age = []
    medicine = []
    for patient in dataset.data[0][0]:
        true_age = dataset.voc[0]["age_voc"].idx2word[patient[3]]

        if true_age < 14:
            continue
        elif true_age > 90:
            true_age = 90

        for med in patient[2]:
            if med < 100:
                age.append(true_age)
                medicine.append(med)

    table = pd.crosstab(age, medicine)
    sns.heatmap(table, annot=False, fmt='d', cmap="YlOrRd", cbar=False, vmax=3000)

    # Add labels and title to the plot
    plt.xlabel("Medicine")
    plt.ylabel("Age")
    plt.title("Medication Count by Age Category")

    # Show the plot
    plt.show()

def age_category_medicine_violin_three(dataset):

    age_dict = {}

    for i in range(91)[20: :10]:
        age_dict[i] = []

    for patient in dataset.data[0][0]:
        true_age = dataset.voc[0]["age_voc"].idx2word[patient[3]]
        true_age = round(true_age/10)*10

        if true_age < 16:
            continue
        elif true_age > 90:
            true_age = 90

        for med in patient[2]:
            # if med < 100:
            age_dict[true_age].append(med)


    for age in age_dict:
        x = collections.Counter(age_dict[age])
        y = x.most_common(10)[5:]
        age_dict[age] = list(map(lambda x: x[0], y))


    data = age_dict
    values = np.unique([value for sublist in data.values() for value in sublist])

    # Create a 2D array of values from the dictionary data
    arr = np.zeros((len(data), len(values)))
    for i, label in enumerate(data.keys()):
        for j, value in enumerate(values):
            if value in data[label]:
                arr[i, j] = value

    # Create a DataFrame with the row and column labels
    df = pd.DataFrame(arr, columns=values, index=data.keys())

    # Create the heatmap using seaborn
    ax = sns.heatmap(df, annot=False, cmap="YlGnBu")

    # Set the X-axis label
    ax.set_xlabel('Medicine Code')

    # Set the Y-axis label
    ax.set_ylabel('Rounded Age')

    # Set the X-axis tick labels to the actual values
    ax.set_xticklabels(values)

    # Show the plot
    plt.show()
