# coding: utf-8

# """ Dataset Description
# 1) ID number
# 2) Outcome (R = recur, N = nonrecur)
# 3) Time (recurrence time if field 2 = R, disease-free time if
# 	field 2	= N)
# 4-33) Ten real-valued features are computed for each cell nucleus:
#
# 	a) radius (mean of distances from center to points on the perimeter)
# 	b) texture (standard deviation of gray-scale values)
# 	c) perimeter
# 	d) area
# 	e) smoothness (local variation in radius lengths)
# 	f) compactness (perimeter^2 / area - 1.0)
# 	g) concavity (severity of concave portions of the contour)
# 	h) concave points (number of concave portions of the contour)
# 	i) symmetry
# 	j) fractal dimension ("coastline approximation" - 1)
#
# Several of the papers listed above contain detailed descriptions of
# how these features are computed.
#
# The mean, standard error, and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features.  For instance, field 4 is Mean Radius, field
# 14 is Radius SE, field 24 is Worst Radius.
#
# Values for features 4-33 are recoded with four significant digits.
#
# 34) Tumor size - diameter of the excised tumor in centimeters
# 35) Lymph node status - number of positive axillary lymph nodes
# observed at time of surgery """

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

col_names = ['id',
             'outcome',
             'time',
             'mean_radius',
             'mean_texture',
             'mean_perimeter',
             'mean_area',
             'mean_smoothness',
             'mean_compactness',
             'mean_concavity',
             'mean_concave_points',
             'mean_symmetry',
             'mean_fractal_dimension',
             'se_radius',
             'se_texture',
             'se_perimeter',
             'se_area',
             'se_smoothness',
             'se_compactness',
             'se_concavity',
             'se_concave_points',
             'se_symmetry',
             'se_fractal_dimension',
             'worst_radius',
             'worst_texture',
             'worst_perimeter',
             'worst_area',
             'worst_smoothness',
             'worst_compactness',
             'worst_concavity',
             'worst_concave_points',
             'worst_symmetry',
             'worst_fractal_dimension',
             'tumor_size',
             'lymph_node_status',
             ]

dataset = pd.read_csv('wpbc.data', names=col_names)
print(dataset.head())

# sort dataframe by column mean_perimeter
dataset = dataset.sort_values(by=['mean_perimeter'])
print(dataset.head())

X, y = dataset.mean_perimeter, dataset.mean_area
X = X.values.reshape(198, 1)

# create linear regression object
regr = LinearRegression()
# fit the model
regr.fit(X, y)
predictions = regr.predict(X)

equation = 'Linear Regression: ' + 'y=' + str(round(regr.coef_[0], 3)) + 'x' + str(round(regr.intercept_, 3))

# polynomial regression
# one extra step: transform the array of inputs to include non-linear terms such as X^2
X_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)

# create model and fit
poly_regr = LinearRegression().fit(X_, y)

predictions_ = poly_regr.predict(X_)

equation_ = 'Polynomial Regression: ' + 'y=' + str(round(poly_regr.coef_[0], 3)) + 'x' + str(round(poly_regr.intercept_, 3))

fig, ax = plt.subplots(figsize=(10, 10))
x, y = dataset.mean_perimeter, dataset.mean_area
scale = 50.0
ax.scatter(x, y, c='#6ff4f4', s=scale, label='#6ff4f4',
            alpha=0.6, edgecolors='none')

ax.plot(X, predictions, c='#8789C0', label='#8789C0')
ax.plot(X, predictions_, c='#FF6B6B', label='#FF6B6B')
ax.legend([equation, equation_, 'mean_perimeter', 'se_perimiter', 'worst_perimeter'])
ax.grid(True)

plt.show()

# calculate the Pearsons's correlation between two variables (perimeter and area)
corr, _ = pearsonr(dataset.mean_perimeter, dataset.mean_area)
print(corr)
