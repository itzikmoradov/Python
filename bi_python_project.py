import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

""" Stroke Gender Correlation """

gender_stroke_crosstab = pd.crosstab(df['gender'], df['stroke'])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot of the number of people who had stroke by gender
bars = axes[0].bar(gender_stroke_crosstab.index, gender_stroke_crosstab[1].values, color=['pink', 'blue'])

axes[0].set_ylabel('Stroke Count')
axes[0].set_title('Stroke Count by Gender')

# Text values above the bars
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, bar.get_height(), ha='center', va='bottom')

# Plot of total stroke occurrence by gender
gender_stroke_crosstab.plot(kind='bar', stacked=False, ax=axes[1], xlabel="")
axes[1].set_ylabel('Count')
axes[1].set_title('Stroke Occurrence Counts by Gender')
axes[1].set_xticklabels(gender_stroke_crosstab.index, rotation=0)
axes[1].legend(['No Stroke', 'Stroke'], title='Stroke Occurrence', loc='upper right')

# Text values above the bars
for container in axes[1].containers:
    axes[1].bar_label(container, label_type='edge')

# Chi-square test
_, p, _, _ = chi2_contingency(gender_stroke_crosstab)

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
df_grouped_as = df.groupby(['age_group', 'stroke']).size().unstack() # df_grouped_as -- 'as' stands for age stroke

df_grouped_as.columns = ['No Stroke Count', 'Stroke Count']
df_grouped_as.sort_values(by='Stroke Count', inplace=True)

# Plot of total stroke occurrences by age groups
ax_age = df_grouped_as.plot(kind='bar', stacked=False, figsize=(6, 5))
ax_age.set_title('Stroke Occurrences Counts by Age Groups', fontdict={'fontsize': 9})
ax_age.set_xlabel('')
ax_age.set_ylabel('Count')
ax_age.set_xticklabels(df_grouped_as.index, rotation=0)
ax_age.legend(['No Stroke', 'Stroke'], loc='upper right')

# Text values above the bars
for container in ax_age.containers:
    ax_age.bar_label(container, label_type='edge')

# Perform Chi-Square Test of Independence to get p-value
_, p, _, _ = chi2_contingency(df_grouped_as)

if p > 0.05:
    ax_age.text(0, -0.15, f'Chi-Square Test:\n', transform=ax_age.transAxes,
                fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                fontweight='bold')

    ax_age.text(0, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                         f'P-Value > \u03B1 ---> There is no significant association between age and stroke',
                transform=ax_age.transAxes,
                fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

else:
    ax_age.text(0, -0.15, f'Chi-Square Test:', transform=ax_age.transAxes,
                fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                fontweight='bold')

    ax_age.text(0, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                         f'P-Value > \u03B1 ---> There a significant association between age and stroke',
                transform=ax_age.transAxes,
                fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

plt.tight_layout()
plt.show()

"""P-value for strokes by age groups implies that there is an association between them. Therefor the gender is being 
added as variable to try to converge on a specific predication according to gender"""

# new column 'gender+age group creation'
df['gender'] = df['gender'].apply(lambda x: x[0])  # taking the first letter of the gender
df['Gender + Age Group'] = df['gender'] + ' ' + df['age_group'].astype(str)
df_grouped_gas = df.groupby(['Gender + Age Group', 'stroke']).size().unstack()# df_grouped_gas -- 'gas' stands for gender, age, stroke
df_grouped_gas.columns = ['No Stroke Count', 'Stroke Count']
df_grouped_gas.sort_values(by='Stroke Count', inplace=True)

# Visualization
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
# Plot of total stroke occurrences by gender+age groups
ax = df_grouped_gas.plot(kind='bar', stacked=False, figsize=(12, 6), ax=axes[0])
ax.set_title('Stroke Occurrences Counts by Gender and Age Group', fontdict={'fontsize': 9})
ax.set_xlabel('')
ax.set_ylabel('Count')
ax.set_xticklabels(df_grouped_gas.index, rotation=45, fontdict={'fontsize': 8})
ax.legend(['No Stroke', 'Stroke'], loc='upper right')

# Text values above the bars
for container in ax.containers:
    ax.bar_label(container, label_type='edge')

# Adding proportion on stroke/no stroke from total occurrences for each gender+age group
# Visualization of proportions
df_proportions = df_grouped_gas.div(df_grouped_gas.sum(axis=1), axis=0)
ax_proportions = df_proportions.plot(kind='bar', stacked=False, figsize=(12, 6), ax=axes[1])
ax_proportions.set_title('Proportion of Stroke Occurrences Counts by Gender and Age Group', fontdict={'fontsize': 9})
ax_proportions.set_xlabel('')
ax_proportions.set_ylabel('Proportion')
ax_proportions.set_xticklabels(df_proportions.index, rotation=45, fontdict={'fontsize': 8})
ax_proportions.set_ybound(upper=1.2)
ax_proportions.legend(['No Stroke', 'Stroke'], bbox_to_anchor=(1, 1), loc='upper right')

# Add text values above the bars
for container in ax_proportions.containers:
    ax_proportions.bar_label(container, label_type='edge', fmt='%.2f')

# Perform Chi-Square Test of Independence
chi2, p, dof, expected = chi2_contingency(df_grouped_gas)

# Determine if null hypothesis is rejected
if p > 0.05:
    ax.text(0, 0, f'Chi-Square Test:\n', transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='left',
            fontweight='bold')

    ax.text(0, 0, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                  f'P-Value > \u03B1 ---> There is no significant association between age and stroke',
            transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

else:
    ax.text(0, -0.28, f'Chi-Square Test:', transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='left',
            fontweight='bold')

    ax.text(0, -0.42, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                      f'P-Value > \u03B1 ---> There a significant association between age and stroke',
            transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

plt.tight_layout()
plt.show()

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Boxplot of Age by Gender and Stroke Status for further insights
male_no_stroke = df[(df['gender'] == 'Male') & (df['stroke'] == 0)]['age']
male_stroke = df[(df['gender'] == 'Male') & (df['stroke'] == 1)]['age']
female_no_stroke = df[(df['gender'] == 'Female') & (df['stroke'] == 0)]['age']
female_stroke = df[(df['gender'] == 'Female') & (df['stroke'] == 1)]['age']


plt.figure(figsize=(8, 6))
bp = plt.boxplot([male_no_stroke, male_stroke, female_no_stroke, female_stroke], notch=True,
                 labels=['Male - No Stroke', 'Male - Stroke', 'Female - No Stroke', 'Female - Stroke'],
                 patch_artist=True)

colors = ['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add median, quartiles, and error bars as text annotations
for i, box in enumerate(bp['boxes']):
    y_median = bp['medians'][i].get_ydata()[0]
    y_q1 = bp['whiskers'][i * 2].get_ydata()[0]
    y_q3 = bp['whiskers'][i * 2 + 1].get_ydata()[0]

    x_offset = 0.3  # Adjust this value to change the position of the text
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

""" Stroke and Heart Disease Correlation """

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Create a cross table of heart disease and stroke
df_hd_stroke = df.loc[:, ['heart_disease', 'stroke']]
df_hd_stroke['heart_disease'] = df_hd_stroke['heart_disease'].apply(
    lambda val: 'Heart Disease' if val == 1 else 'No Heart Disease')
df_hd_stroke['stroke'] = df_hd_stroke['stroke'].apply(lambda val: 'Stroke' if val == 1 else 'No Stroke')

hd_stroke_crosstab = df_hd_stroke.groupby(['heart_disease', 'stroke']).size().unstack()

# Plot of total stroke occurrence by heart disease
ax = hd_stroke_crosstab.plot(kind='bar', stacked=False, xlabel="")
ax.set_ylabel('Count')
ax.set_title('Distribution of Stroke Occurrences Based on Heart Disease Status')
ax.set_xticklabels(hd_stroke_crosstab.index, rotation=0)
ax.legend(['No Stroke', 'Stroke'], title='Stroke Occurrence', loc='upper right')

# Text values above the bars
for container in ax.containers:
    ax.bar_label(container, label_type='edge')

# Chi-square test
_, p, _, _ = chi2_contingency(hd_stroke_crosstab)

if p > 0.05:
    ax.text(0, -0.2, f'Chi-Square Test:\n', transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='left',
            fontweight='bold')

    ax.text(0, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                        f'P-Value > \u03B1 ---> There is no significant association between heart disease and stroke',
            transform=axes[1].transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

else:
    ax.text(0, -0.2, f'Chi-Square Test:\n', transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='left',
            fontweight='bold')

    ax.text(0, -0.3, f'\u03B1: 0.05,  P-Value: {p:.2f}\n\n'
                        f'P-Value < \u03B1 ---> There a significant association between heart disease and stroke',
            transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))


plt.tight_layout()
plt.show()

# Boxplot of Age by Heart Disease and Stroke Status for further insights
hd_no_stroke = df[(df['heart_disease'] == 1) & (df['stroke'] == 0)]['age']
hd_stroke = df[(df['heart_disease'] == 1) & (df['stroke'] == 1)]['age']
no_hd_no_stroke = df[(df['heart_disease'] == 0) & (df['stroke'] == 0)]['age']
no_hd_stroke = df[(df['heart_disease'] == 0) & (df['stroke'] == 1)]['age']


plt.figure(figsize=(8, 6))
bp = plt.boxplot([hd_no_stroke, hd_stroke, no_hd_no_stroke, no_hd_stroke], notch=True,
                 labels=['HD - No Stroke', 'HD - Stroke', 'No HD - No Stroke', 'No HD - Stroke'],
                 patch_artist=True)

colors = ['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add median, quartiles, and error bars as text annotations
for i, box in enumerate(bp['boxes']):
    y_median = bp['medians'][i].get_ydata()[0]
    y_q1 = bp['whiskers'][i * 2].get_ydata()[0]
    y_q3 = bp['whiskers'][i * 2 + 1].get_ydata()[0]

    x_offset = 0.3  # Adjust this value to change the position of the text
    plt.text(i + 1 - x_offset, y_median, f'Median: {y_median:.2f}', ha='center', va='center', color='black',
             fontweight='bold')
    plt.text(i + 1 - x_offset, y_q1, f'Q1: {y_q1:.2f}', ha='center', va='center', color='black', fontweight='bold')
    plt.text(i + 1 - x_offset, y_q3, f'Q3: {y_q3:.2f}', ha='center', va='center', color='black', fontweight='bold')

# Add labels and title
plt.xlabel('Heart Disease and Stroke Status')
plt.ylabel('Age')
plt.title('Boxplot of Heart Disease Status and Stroke Occurrences by Age')

plt.tight_layout()
plt.show()
