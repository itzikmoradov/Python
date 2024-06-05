import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind
import numpy as np
import seaborn as sns

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

""" Stroke Gender Correlation """

gender_stroke_crosstab = pd.crosstab(df['gender'], df['stroke'])
gender_stroke_crosstab_norm = pd.crosstab(df['gender'], df['stroke'], normalize='index')

# Plot of the number of people who had stroke by gender
plt.ion()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
bars = axes[0].bar(gender_stroke_crosstab.index, gender_stroke_crosstab[1].values, color=['pink', 'blue'])

axes[0].set_ylabel('Stroke Count')
axes[0].set_title('Stroke Count by Gender')

# Text values above the bars
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, bar.get_height(), ha='center', va='bottom')

# Plot of stroke incidences proportion by gender
gender_stroke_crosstab_norm.plot(kind='bar', stacked=False, ax=axes[1], xlabel="")
axes[1].set_title('Stroke Incidences Proportion by Gender')
axes[1].set_xticklabels(gender_stroke_crosstab.index, rotation=0)
axes[1].legend(['No Stroke', 'Stroke'], title='Stroke Incidences', bbox_to_anchor=(1, 1))

# Text values above the bars
for container in axes[1].containers:
    axes[1].bar_label(container, label_type='edge', fmt='%.2f')

# Chi-square test
_, p, _, _ = chi2_contingency(gender_stroke_crosstab)

# Text in the plot for the p-value and its interpretation
if p > 0.05:
    axes[0].text(-1.2, -0.2, f'Chi-Square Test:\n', transform=axes[1].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 fontweight='bold')

    axes[0].text(-1.2, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                             f'P-Value > \u03B1 ---> There is no significant association between gender and stroke',
                 transform=axes[1].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

else:
    axes[0].text(-1.2, -0.2, f'Chi-Square Test:\n', transform=axes[1].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 fontweight='bold')

    axes[0].text(-1.2, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                             f'P-Value < \u03B1 ---> There a significant association between gender and stroke',
                 transform=axes[1].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------------------------------
"""Stroke Age Group Correlation"""

# Define a function to label age groups
def age_group_label(age):
    if age < 50:
        return 'under 50'
    elif 50 <= age < 70:
        return '50-70'
    else:
        return 'above 70'


# Create a new dataframe -- age group vs. stroke occurrences
df['age_group'] = df['age'].apply(age_group_label)
df_grouped_as = df.groupby(['age_group', 'stroke']).size().unstack()  # df_grouped_as -- 'as' stands for age stroke
# Normalize the df per index
df_grouped_as_normalized = df_grouped_as.div(df_grouped_as.sum(axis=1), axis=0)
df_grouped_as_normalized.columns = ['No Stroke Count', 'Stroke Count']
df_grouped_as_normalized.sort_values(by='Stroke Count', inplace=True)

# Plot of stroke Incidences proportion by age groups
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
df_grouped_as_normalized.plot(kind='bar', stacked=False, ax=axes[0], figsize=(6, 5))
axes[0].set_title('Proportion of Stroke Cases by Age Groups')
axes[0].set_xlabel('')
axes[0].set_xticklabels(df_grouped_as_normalized.index, rotation=0)
axes[0].legend(['No Stroke', 'Stroke'], loc='upper right')

# Text values above the bars
for container in axes[0].containers:
    axes[0].bar_label(container, label_type='edge', fmt='%.2f')

# Perform Chi-Square Test of Independence to get p-value
_, p, _, _ = chi2_contingency(df_grouped_as)

if p > 0.05:
    axes[0].text(0, -0.15, f'Chi-Square Test:\n', transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 fontweight='bold')

    axes[0].text(0, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                          f'P-Value > \u03B1 ---> There is no significant association between age and stroke',
                 transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

else:
    axes[0].text(0, -0.15, f'Chi-Square Test:', transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 fontweight='bold')

    axes[0].text(0, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                          f'P-Value > \u03B1 ---> There is a significant association between age and stroke',
                 transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

# Present the trendline of the stroke incidences by age groups
axes[1].plot(df_grouped_as_normalized.index, df_grouped_as_normalized.loc[:, 'Stroke Count'].values, linestyle='--')

# Add the values to the plot
for x, y in zip(df_grouped_as_normalized.index, df_grouped_as_normalized.loc[:, 'Stroke Count'].values):
    axes[1].text(x, y, f'{y:.2f}', verticalalignment='bottom', horizontalalignment='right')

axes[1].set_title('Trend line for Stroke Incidence by Age Groups')
axes[1].set_xlabel('Age Groups')
plt.subplots_adjust(bottom=0.25, wspace=0.5)

plt.show()


# Boxplot Stroke Occurrences by Age
no_stroke = df[df['stroke'] == 0]['age']
stroke = df[df['stroke'] == 1]['age']

plt.figure(figsize=(8, 8))
bp = plt.boxplot([no_stroke, stroke], notch=True, tick_labels=['No Stroke', 'Stroke'], patch_artist=True)

colors = ['#1f77b4', '#ff7f0e']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add median, quartiles, and error bars as text annotations
for i, box in enumerate(bp['boxes']):
    y_median = bp['medians'][i].get_ydata()[0]
    y_q1 = bp['whiskers'][i * 2].get_ydata()[0]
    y_q3 = bp['whiskers'][i * 2 + 1].get_ydata()[0]

    x_offset = 0.3
    plt.text(i + 1 - x_offset, y_median, f'Median: {y_median:.2f}', ha='center', va='center', color='black',
             fontweight='bold')
    plt.text(i + 1 - x_offset, y_q1, f'Q1: {y_q1:.2f}', ha='center', va='center', color='black', fontweight='bold')
    plt.text(i + 1 - x_offset, y_q3, f'Q3: {y_q3:.2f}', ha='center', va='center', color='black', fontweight='bold')

# Add labels and title
plt.ylabel('Age')
plt.title('Boxplot Stroke Incidences by Age')

# Calculate the means and standard deviations
mean_stroke = np.mean(stroke)
mean_no_stroke = np.mean(no_stroke)
std_stroke = np.std(stroke, ddof=1)
std_no_stroke = np.std(no_stroke, ddof=1)
#
# Calculate Cohen's d
n_stroke = len(stroke)
n_no_stroke = len(no_stroke)
pooled_std = np.sqrt(
    ((n_stroke - 1) * std_stroke ** 2 + (n_no_stroke - 1) * std_no_stroke ** 2) / (n_stroke + n_no_stroke - 2))
cohen_d = (mean_stroke - mean_no_stroke) / pooled_std
t_stat, p_value = ttest_ind(stroke, no_stroke, equal_var=False)  # Using Welch's t-test
t_test_result = f'\u03B1: 0.05\np-value={p_value:.2f}, cohen_d={cohen_d:.2f}'

# Add the t-test result text below the plot
ax = plt.gca()
plt.figtext(0, -0.1, f'T-test:', ha='left', va='baseline', fontsize=10, transform=ax.transAxes, fontweight='bold')
plt.figtext(0, -0.2, t_test_result, ha='left', va='baseline', fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.show()

"""P-value for strokes by age groups implies that there is an association between them. Therefor the gender is being
added as variable to try to converge on a specific predication according to gender and age"""

# new column 'gender+age group creation'
# taking the first letter of the gender and adding to it the age group
df['Gender + Age Group'] = df['gender'].apply(lambda x: x[0])  + ' ' + df['age_group'].astype(str)
df_grouped_gas = df.groupby(
    ['Gender + Age Group', 'stroke']).size().unstack()  # df_grouped_gas -- 'gas' stands for gender, age, stroke
df_grouped_gas.columns = ['No Stroke Count', 'Stroke Count']

# Normalize the df per index
df_grouped_gas_normalized = df_grouped_gas.div(df_grouped_gas.sum(axis=1), axis=0)
df_grouped_gas_normalized.sort_values(by='Stroke Count', inplace=True)

# Visualization
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
# Plot of stroke incidences proportion by gender+age groups
df_grouped_gas_normalized.plot(kind='bar', stacked=False, ax=axes[0])
axes[0].set_title('Stroke Incidences Proportion by Gender and Age Group')
axes[0].set_xlabel('')
axes[0].set_xticklabels(df_grouped_gas_normalized.index, rotation=45, fontdict={'fontsize': 8})
axes[0].legend(['No Stroke', 'Stroke'], loc='upper right')

# Text values above the bars
for container in axes[0].containers:
    axes[0].bar_label(container, label_type='edge', fmt='%.2f')

# Perform Chi-Square Test of Independence
chi2, p, dof, expected = chi2_contingency(df_grouped_gas)

# Determine if null hypothesis is rejected
if p > 0.05:
    axes[0].text(0, 0, f'Chi-Square Test:\n', transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 fontweight='bold')

    axes[0].text(0, 0, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                       f'P-Value > \u03B1 ---> There is no significant association between age+gender and stroke',
                 transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

else:
    axes[0].text(0, -0.28, f'Chi-Square Test:', transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 fontweight='bold')

    axes[0].text(0, -0.42, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                           f'P-Value > \u03B1 ---> There is a significant association between age+gender and stroke',
                 transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

# Present the trendline of the stroke incidences by age groups+gender
axes[1].plot(df_grouped_gas_normalized.index, df_grouped_gas_normalized.loc[:, 'Stroke Count'].values, linestyle='--')

# Add the values to the plot
for x, y in zip(df_grouped_gas_normalized.index, df_grouped_gas_normalized.loc[:, 'Stroke Count'].values):
    axes[1].text(x, y, f'{y:.2f}', verticalalignment='bottom', horizontalalignment='right')

axes[1].set_title('Trend line for Stroke Incidence by Age Groups and Gender')
axes[1].set_xlabel('Age Groups and Gender')
axes[1].set_xticklabels(df_grouped_gas_normalized.index, rotation=45, fontdict={'fontsize': 8})
plt.subplots_adjust(bottom=0.25, wspace=0.5)


plt.tight_layout()
plt.show()

# Boxplot of Age by Gender and Stroke Status for further insights
male_no_stroke = df[(df['gender'] == 'Male') & (df['stroke'] == 0)]['age']
male_stroke = df[(df['gender'] == 'Male') & (df['stroke'] == 1)]['age']
female_no_stroke = df[(df['gender'] == 'Female') & (df['stroke'] == 0)]['age']
female_stroke = df[(df['gender'] == 'Female') & (df['stroke'] == 1)]['age']

plt.figure(figsize=(8, 6))
bp = plt.boxplot([male_no_stroke, male_stroke, female_no_stroke, female_stroke], notch=True,
                 tick_labels=['Male - No Stroke', 'Male - Stroke', 'Female - No Stroke', 'Female - Stroke'],
                 patch_artist=True)
#
colors = ['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add median, quartiles, and error bars as text annotations
for i, box in enumerate(bp['boxes']):
    y_median = bp['medians'][i].get_ydata()[0]
    y_q1 = bp['whiskers'][i * 2].get_ydata()[0]
    y_q3 = bp['whiskers'][i * 2 + 1].get_ydata()[0]

    x_offset = 0.3
    plt.text(i + 1 - x_offset, y_median, f'Median: {y_median:.2f}', ha='center', va='center', color='black',
             fontweight='bold')
    plt.text(i + 1 - x_offset, y_q1, f'Q1: {y_q1:.2f}', ha='center', va='center', color='black', fontweight='bold')
    plt.text(i + 1 - x_offset, y_q3, f'Q3: {y_q3:.2f}', ha='center', va='center', color='black', fontweight='bold')

# Add labels and title
plt.xlabel('Gender and Stroke Status')
plt.ylabel('Age')
plt.title('Boxplot of Age by Gender and Stroke Status')

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------
""" Stroke and Heart Disease Correlation """

# Create a cross table of heart disease and stroke
df_hd_stroke = df.loc[:, ['heart_disease', 'stroke']]
df_hd_stroke['heart_disease'] = df_hd_stroke['heart_disease'].apply(
    lambda val: 'Heart Disease' if val == 1 else 'No Heart Disease')
df_hd_stroke['stroke'] = df_hd_stroke['stroke'].apply(lambda val: 'Stroke' if val == 1 else 'No Stroke')
hd_stroke_crosstab = df_hd_stroke.groupby(['heart_disease', 'stroke']).size().unstack()

# Normalize the df by index
hd_stroke_crosstab_norm = hd_stroke_crosstab.div(hd_stroke_crosstab.sum(axis=1), axis=0)

# Plot of stroke incidences proportion by heart disease
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
hd_stroke_crosstab_norm.plot(kind='bar', stacked=False, xlabel="", ax=axes[0])
axes[0].set_title('Stroke Incidences Proportion Based on Heart Disease Status', fontsize=9)
axes[0].set_xticklabels(hd_stroke_crosstab_norm.index, rotation=0, fontsize=7)
axes[0].legend(['No Stroke', 'Stroke'], title='Stroke Incidences', loc='best', bbox_to_anchor=(1, 1), fontsize=7)
plt.subplots_adjust(wspace=1.2)

# Text values above the bars
for container in axes[0].containers:
    axes[0].bar_label(container, label_type='edge', fmt='%.2f')

# Chi-square test
_, p, _, _ = chi2_contingency(hd_stroke_crosstab)

if p > 0.05:
    axes[0].text(0, -0.2, f'Chi-Square Test:\n', transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 fontweight='bold')

    axes[0].text(0, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                          f'P-Value > \u03B1 ---> There is no significant association between heart disease and stroke',
                 transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

else:
    axes[0].text(0, -0.2, f'Chi-Square Test:\n', transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 fontweight='bold')

    axes[0].text(0, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                          f'P-Value < \u03B1 ---> There is a significant association between heart disease and stroke',
                 transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

# Boxplot of Age by Heart Disease and Stroke Status for further insights
hd_no_stroke = df[(df['heart_disease'] == 1) & (df['stroke'] == 0)]['age']
hd_stroke = df[(df['heart_disease'] == 1) & (df['stroke'] == 1)]['age']
no_hd_no_stroke = df[(df['heart_disease'] == 0) & (df['stroke'] == 0)]['age']
no_hd_stroke = df[(df['heart_disease'] == 0) & (df['stroke'] == 1)]['age']

bp = axes[1].boxplot([hd_no_stroke, hd_stroke, no_hd_no_stroke, no_hd_stroke], notch=True,
                     patch_artist=True)

colors = ['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add median, quartiles, and error bars as text annotations
for i, box in enumerate(bp['boxes']):
    y_median = bp['medians'][i].get_ydata()[0]
    y_q1 = bp['whiskers'][i * 2].get_ydata()[0]
    y_q3 = bp['whiskers'][i * 2 + 1].get_ydata()[0]

    x_offset = 0.3
    plt.text(i + 1 - x_offset, y_median, f'Median: {y_median:.2f}', ha='center', va='center', color='black',
             fontweight='bold', fontsize=7)
    plt.text(i + 1 - x_offset, y_q1, f'Q1: {y_q1:.2f}', ha='center', va='center', color='black', fontweight='bold',
             fontsize=7)
    plt.text(i + 1 - x_offset, y_q3, f'Q3: {y_q3:.2f}', ha='center', va='center', color='black', fontweight='bold',
             fontsize=7)

# Add labels and title
axes[1].set_xlabel('Heart Disease and Stroke Status')
axes[1].set_ylabel('Age')
axes[1].set_xticklabels(['HD - No Stroke', 'HD - Stroke', 'No HD - No Stroke', 'No HD - Stroke'], rotation=45,
                        fontsize=7)
axes[1].set_title('Boxplot of Heart Disease Status and Stroke Incidences by Age', fontsize=9)

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------
""" Stroke and ResidenceType Correlation """

residence_stroke_df = df[['Residence_type', 'stroke']]
residence_stroke_df['stroke'] = residence_stroke_df['stroke'].replace({0: 'No Stroke', 1: 'Stroke'})
residence_stroke_df = residence_stroke_df.value_counts()
residence_stroke_crosstab = pd.crosstab(df['Residence_type'], df['stroke'])
residence_stroke_crosstab_norm = residence_stroke_crosstab.div(residence_stroke_crosstab.sum(axis=1), axis=0)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot of  stroke incidences proportion by ResidenceType
axes[0] = residence_stroke_crosstab_norm.plot(kind='bar', stacked=False, xlabel="", ax=axes[0])
axes[0].set_xticklabels(residence_stroke_crosstab_norm.index, rotation=0)
axes[0].legend(['No Stroke', 'Stroke'], title='Stroke Incidences', loc='best', bbox_to_anchor=(1, 1))

# Text values above the bars
for container in axes[0].containers:
    axes[0].bar_label(container, label_type='edge', fmt='%.2f')

# Chi-square test
_, p, _, _ = chi2_contingency(residence_stroke_crosstab)

if p > 0.05:
    axes[0].text(0, -0.2, f'Chi-Square Test:\n', transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 fontweight='bold')

    axes[0].text(0, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                          f'P-Value > \u03B1 ---> There is no significant association between ResidenceType and stroke',
                 transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

else:
    axes[0].text(0, -0.2, f'Chi-Square Test:\n', transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 fontweight='bold')

    axes[0].text(0, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                          f'P-Value < \u03B1 ---> There is a significant association between ResidenceType and stroke',
                 transform=axes[0].transAxes,
                 fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

residence_stroke_df.plot.pie(y='Residence_type', autopct='%1.1f%%', ax=axes[1])

fig.suptitle('Distribution of Stroke Incidences Based on ResidenceType', fontsize=16)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------
"""Stroke Occurrences by Average Glucose Levels"""

glucose_stroke_df = df[['avg_glucose_level', 'stroke']]
glucose_stroke_df['stroke'] = glucose_stroke_df['stroke'].replace({0: 'No Stroke', 1: 'Stroke'})
stroke_group = glucose_stroke_df.loc[glucose_stroke_df['stroke'] == 'Stroke']['avg_glucose_level']
no_stroke_group = glucose_stroke_df.loc[glucose_stroke_df['stroke'] == 'No Stroke']['avg_glucose_level']

# Calculate the means and standard deviations
mean_stroke = np.mean(stroke_group)
mean_no_stroke = np.mean(no_stroke_group)
std_stroke = np.std(stroke_group, ddof=1)
std_no_stroke = np.std(no_stroke_group, ddof=1)

# Calculate Cohen's d
n_stroke = len(stroke_group)
n_no_stroke = len(no_stroke_group)
pooled_std = np.sqrt(
    ((n_stroke - 1) * std_stroke ** 2 + (n_no_stroke - 1) * std_no_stroke ** 2) / (n_stroke + n_no_stroke - 2))
cohen_d = (mean_stroke - mean_no_stroke) / pooled_std

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))

bp = axes[0].boxplot([stroke_group, no_stroke_group], notch=True,
                     tick_labels=['Stroke Group', 'No Stroke Group'],
                     patch_artist=True)

colors = ['#ff7f0e', '#1f77b4']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add median, quartiles, and error bars as text annotations
for i, box in enumerate(bp['boxes']):
    y_median = bp['medians'][i].get_ydata()[0]
    y_q1 = bp['whiskers'][i * 2].get_ydata()[0]
    y_q3 = bp['whiskers'][i * 2 + 1].get_ydata()[0]

    x_offset = 0.3
    axes[0].text(i + 1 - x_offset, y_median, f'Median: {y_median:.2f}', ha='center', va='center', color='black',
                 fontweight='bold', fontsize=7)
    axes[0].text(i + 1 - x_offset, y_q1, f'Q1: {y_q1:.2f}', ha='center', va='center', color='black', fontweight='bold',
                 fontsize=7)
    axes[0].text(i + 1 - x_offset, y_q3, f'Q3: {y_q3:.2f}', ha='center', va='center', color='black', fontweight='bold',
                 fontsize=7)

# Add labels and title
axes[0].set_ylabel('Average Glucose Level[mg/dL]')
axes[0].set_title('Boxplot of Stroke Incidences by Average Glucose Levels', fontsize=9)

# Performing T-test
t_stat, p_value = ttest_ind(stroke_group, no_stroke_group, equal_var=False)  # Using Welch's t-test
t_test_result = f'\u03B1: 0.05\np-value={p_value:.2f}, cohen_d={cohen_d:.2f}'

# Add the t-test result text below the plot
axes[0].text(0, -0.13, f'T-test:', ha='left', va='baseline', fontsize=10, transform=axes[0].transAxes,
             fontweight='bold')
axes[0].text(0, -0.25, t_test_result, ha='left', va='baseline', fontsize=10, transform=axes[0].transAxes,
             bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

# Plot the density curve for the 'no-stroke' group
sns.kdeplot(data=glucose_stroke_df[glucose_stroke_df['stroke'] == 'No Stroke'], x='avg_glucose_level', fill=True,
            label='No Stroke', ax=axes[1])

# Plot the density curve for the 'stroke' group
sns.kdeplot(data=glucose_stroke_df[glucose_stroke_df['stroke'] == 'Stroke'], x='avg_glucose_level', fill=True,
            label='Stroke', ax=axes[1])
axes[1].legend()
axes[1].set_title('Average Glucose Levels Density Distribution of Stroke Incidences', fontsize=9)
axes[1].set_xlabel('Average Glucose Levels [mg/dL]')
plt.subplots_adjust(bottom=0.2, wspace=0.5)
plt.show()

# --------------------------------------------------------------------------------------------------
"""Stroke Occurrences by BMI"""

bmi_stroke_df = df[['bmi', 'stroke']].dropna()
bmi_stroke_df['stroke'] = bmi_stroke_df['stroke'].replace({0: 'No Stroke', 1: 'Stroke'})
stroke_group = bmi_stroke_df.loc[bmi_stroke_df['stroke'] == 'Stroke']['bmi']
no_stroke_group = bmi_stroke_df.loc[bmi_stroke_df['stroke'] == 'No Stroke']['bmi']

# Calculate the means and standard deviations
mean_stroke = np.mean(stroke_group)
mean_no_stroke = np.mean(no_stroke_group)
std_stroke = np.std(stroke_group, ddof=1)
std_no_stroke = np.std(no_stroke_group, ddof=1)
#
# Calculate Cohen's d
n_stroke = len(stroke_group)
n_no_stroke = len(no_stroke_group)
pooled_std = np.sqrt(
    ((n_stroke - 1) * std_stroke ** 2 + (n_no_stroke - 1) * std_no_stroke ** 2) / (n_stroke + n_no_stroke - 2))
cohen_d = (mean_stroke - mean_no_stroke) / pooled_std

# boxplot of stroke incidences by BMI
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))

bp = axes[0].boxplot([stroke_group, no_stroke_group], notch=True,
                     tick_labels=['Stroke Group', 'No Stroke Group'],
                     patch_artist=True)

colors = ['#ff7f0e', '#1f77b4']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add median, quartiles, and error bars as text annotations
for i, box in enumerate(bp['boxes']):
    y_median = bp['medians'][i].get_ydata()[0]
    y_q1 = bp['whiskers'][i * 2].get_ydata()[0]
    y_q3 = bp['whiskers'][i * 2 + 1].get_ydata()[0]

    x_offset = 0.3
    axes[0].text(i + 1 - x_offset, y_median, f'Median: {y_median:.2f}', ha='center', va='center', color='black',
                 fontweight='bold', fontsize=7)
    axes[0].text(i + 1 - x_offset, y_q1, f'Q1: {y_q1:.2f}', ha='center', va='center', color='black', fontweight='bold',
                 fontsize=7)
    axes[0].text(i + 1 - x_offset, y_q3, f'Q3: {y_q3:.2f}', ha='center', va='center', color='black', fontweight='bold',
                 fontsize=7)

# Add labels and title
axes[0].set_ylabel('BMI')
axes[0].set_title('Boxplot of Stroke Incidences by BMI', fontsize=9)

# Performing T-test analysis
t_stat, p_value = ttest_ind(stroke_group, no_stroke_group, equal_var=False)  # Using Welch's t-test
t_test_result = f'\u03B1: 0.05\np-value={p_value:.2f}, cohen_d={cohen_d:.2f}'

# Add the t-test result text below the plot
axes[0].text(0, -0.15, f'T-test:', ha='left', va='baseline', fontsize=10, transform=axes[0].transAxes,
             fontweight='bold')
axes[0].text(0, -0.25, t_test_result, ha='left', va='baseline', fontsize=10, transform=axes[0].transAxes,
             bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

# Plot the density curve for the 'no-stroke' group
sns.kdeplot(data=bmi_stroke_df[bmi_stroke_df['stroke'] == 'No Stroke'], x='bmi', fill=True,
            label='No Stroke', ax=axes[1])

# Plot the density curve for the 'stroke' group
sns.kdeplot(data=bmi_stroke_df[bmi_stroke_df['stroke'] == 'Stroke'], x='bmi', fill=True,
            label='Stroke', ax=axes[1])
axes[1].legend()
axes[1].set_title('BMI Density Distribution of Stroke Incidences', fontsize=9)
axes[1].set_xlabel('BMI')
plt.subplots_adjust(bottom=0.2, wspace=0.5)
plt.show()

# --------------------------------------------------------------------------------------------------

"""Stroke Occurrences by Smoking Status"""

smoking_stroke_df = df[['smoking_status', 'stroke']]
smoking_stroke_df['stroke'] = smoking_stroke_df['stroke'].replace({0: 'No Stroke', 1: 'Stroke'})
smoking_stroke_crosstab = smoking_stroke_df.groupby(['smoking_status', 'stroke']).size().unstack()
smoking_stroke_crosstab_norm = smoking_stroke_crosstab.div(smoking_stroke_crosstab.sum(axis=1), axis=0)

# Plot of stroke incidences proportion by smoking status

fig, axes = plt.subplots(figsize=(8, 6))

smoking_stroke_crosstab_norm.plot(kind='bar', stacked=False, xlabel="", ax=axes)

axes.set_xticklabels(smoking_stroke_crosstab_norm.index, fontsize=8, rotation=0)
axes.legend(['No Stroke', 'Stroke'], title='Stroke Incidences', loc='best', bbox_to_anchor=(1, 1))

# Text values above the bars
for container in axes.containers:
    axes.bar_label(container, label_type='edge', fmt='%.2f')

# Chi-square test
_, p, _, _ = chi2_contingency(smoking_stroke_crosstab)

if p > 0.05:
    axes.text(0, 0, f'Chi-Square Test:\n', transform=axes.transAxes,
              fontsize=8, verticalalignment='bottom', horizontalalignment='left',
              fontweight='bold')

    axes.text(0, 0, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                    f'P-Value > \u03B1 ---> There is no significant association between smoking and stroke',
              transform=axes.transAxes,
              fontsize=8, verticalalignment='bottom', horizontalalignment='left',
              bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

else:
    axes.text(0, -0.2, f'Chi-Square Test:\n', transform=axes.transAxes,
              fontsize=8, verticalalignment='bottom', horizontalalignment='left',
              fontweight='bold')

    axes.text(0, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                       f'P-Value < \u03B1 ---> There is a significant association between smoking and stroke',
              transform=axes.transAxes,
              fontsize=8, verticalalignment='bottom', horizontalalignment='left',
              bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

axes.set_title('Stroke Incidences Proportion Based on Smoking Status', fontsize=9)
plt.tight_layout()
plt.show()

# Boxplot of Smoking Status and Stroke Incidences by Age
formerly_smoked_no_stroke = df[(df['smoking_status'] == 'formerly smoked') & (df['stroke'] == 0)]['age']
formerly_smoked_stroke = df[(df['smoking_status'] == 'formerly smoked') & (df['stroke'] == 1)]['age']
never_smoked_no_stroke = df[(df['smoking_status'] == 'never smoked') & (df['stroke'] == 0)]['age']
never_smoked_stroke = df[(df['smoking_status'] == 'never smoked') & (df['stroke'] == 1)]['age']
smokes_no_stroke = df[(df['smoking_status'] == 'smokes') & (df['stroke'] == 0)]['age']
smokes_stroke = df[(df['smoking_status'] == 'smokes') & (df['stroke'] == 1)]['age']
Unknown_no_stroke = df[(df['smoking_status'] == 'Unknown') & (df['stroke'] == 0)]['age']
Unknown_stroke = df[(df['smoking_status'] == 'Unknown') & (df['stroke'] == 1)]['age']

fig, axes = plt.subplots(figsize=(8, 6))
bp = axes.boxplot([formerly_smoked_no_stroke, formerly_smoked_stroke, never_smoked_no_stroke, never_smoked_stroke,
                   smokes_no_stroke, smokes_stroke, Unknown_no_stroke, Unknown_stroke], notch=True,
                  patch_artist=True)

colors = ['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add median, quartiles, and error bars as text annotations
for i, box in enumerate(bp['boxes']):
    y_median = bp['medians'][i].get_ydata()[0]
    y_q1 = bp['whiskers'][i * 2].get_ydata()[0]
    y_q3 = bp['whiskers'][i * 2 + 1].get_ydata()[0]

    x_offset = 0.4
    axes.text(i + 1 - x_offset, y_median, f'Median: {y_median:.2f}', ha='center', va='center', color='black',
              fontweight='bold', fontsize=7)
    axes.text(i + 1 - x_offset, y_q1, f'Q1: {y_q1:.2f}', ha='center', va='center', color='black', fontweight='bold',
              fontsize=7)
    axes.text(i + 1 - x_offset, y_q3, f'Q3: {y_q3:.2f}', ha='center', va='center', color='black', fontweight='bold',
              fontsize=7)

# Add labels and title
axes.set_xticklabels([f'formerly smoked-\nno stroke', f'formerly smoked-\nstroke', f'never smoked-\nno stroke',
                      f'never smoked-\nstroke', f'smokes-\nno stroke', f'smokes-\nstroke',
                      f'Unknown-\nno stroke',
                      f'Unknown-\nstroke'], fontsize=7)

axes.set_ylabel('Age')
axes.set_title('Boxplot of Smoking Status and Stroke Incidences by Age', fontsize=9)

plt.show(block=True)
plt.ioff()
