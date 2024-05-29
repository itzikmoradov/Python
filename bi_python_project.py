import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

""" Stroke Gender Correlation """

gender_stroke_crosstab = pd.crosstab(df['gender'], df['stroke'])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot of the number of people who had stroke by gender
bars = axes[0].bar(gender_stroke_crosstab.index, gender_stroke_crosstab[1].values, color=['blue', 'pink'])

axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Stroke Count')
axes[0].set_title('Stroke Count by Gender')

# Text values above the bars
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, bar.get_height(), ha='center', va='bottom')

# Plot of total stroke occurrence by gender
gender_stroke_crosstab.plot(kind='bar', stacked=False, ax=axes[1])
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Count')
axes[1].set_title('Stroke Occurrence Counts by Gender')
axes[1].set_xticklabels(gender_stroke_crosstab.index, rotation=0)
axes[1].legend(['No Stroke', 'Stroke'], title='Stroke Occurrence', loc='upper right')

# Text values above the bars
for container in axes[1].containers:
    axes[1].bar_label(container, label_type='edge')

plt.show()

# Chi-square test
chi2, p, dof, expected = chi2_contingency(gender_stroke_crosstab)
print(f"Chi-Square Test:\nAlpha: 0.05, P-Value: {p:.2f}")

if p > 0.05:
    print(f"P-Value > Alpha ---> There is no significant association between gender and stroke")
else:
    print(f"P-Value < 0.05 ---> There a significant association between gender and stroke")
