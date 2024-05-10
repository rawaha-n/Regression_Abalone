Final Project - STAT 847
================
Rawaha Nakhuda
2024-04-19

### Background

The
[dataset](https://www.kaggle.com/competitions/playground-series-s4e4/)
is part of the Kaggle Playground Series, Season 4 Episode 4. It
describes physical measurements of Abalone, with the goal of the
competition to predict its age (number of rings).

The dataset describes the following features:  
- **Sex**: Nominal — M, F, and I (infant)  
- **Length**: Continuous — mm — Longest shell measurement  
- **Diameter**: Continuous — mm — Measurement perpendicular to length  
- **Height**: Continuous — mm — Height with meat in shell  
- **Whole weight**: Continuous — grams — Whole abalone  
- **whole weight 1**: Continuous — grams — Weight of meat  
- **whole weight 2**: Continuous — grams — Gut weight (after bleeding)  
- **Shell weight**: Continuous — grams — Weight after being dried  
- **Rings**: Integer — +1.5 gives the age in years

``` r
df <- read_csv("abalone.csv",
               show_col_types = FALSE) %>%
  clean_names() %>%
  select(-c(id))

df %>%
  skim_without_charts()
```

|                                                  |            |
|:-------------------------------------------------|:-----------|
| Name                                             | Piped data |
| Number of rows                                   | 90615      |
| Number of columns                                | 9          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |            |
| Column type frequency:                           |            |
| character                                        | 1          |
| numeric                                          | 8          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |            |
| Group variables                                  | None       |

Data summary

**Variable type: character**

| skim_variable | n_missing | complete_rate | min | max | empty | n_unique | whitespace |
|:--------------|----------:|--------------:|----:|----:|------:|---------:|-----------:|
| sex           |         0 |             1 |   1 |   1 |     0 |        3 |          0 |

**Variable type: numeric**

| skim_variable  | n_missing | complete_rate | mean |   sd |   p0 |  p25 |  p50 |   p75 |  p100 |
|:---------------|----------:|--------------:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| length         |         0 |             1 | 0.52 | 0.12 | 0.07 | 0.44 | 0.54 |  0.60 |  0.81 |
| diameter       |         0 |             1 | 0.40 | 0.10 | 0.06 | 0.34 | 0.42 |  0.47 |  0.65 |
| height         |         0 |             1 | 0.14 | 0.04 | 0.00 | 0.11 | 0.14 |  0.16 |  1.13 |
| whole_weight   |         0 |             1 | 0.79 | 0.46 | 0.00 | 0.42 | 0.80 |  1.07 |  2.83 |
| whole_weight_1 |         0 |             1 | 0.34 | 0.20 | 0.00 | 0.18 | 0.33 |  0.46 |  1.49 |
| whole_weight_2 |         0 |             1 | 0.17 | 0.10 | 0.00 | 0.09 | 0.17 |  0.23 |  0.76 |
| shell_weight   |         0 |             1 | 0.23 | 0.13 | 0.00 | 0.12 | 0.22 |  0.30 |  1.00 |
| rings          |         0 |             1 | 9.70 | 3.18 | 1.00 | 8.00 | 9.00 | 11.00 | 29.00 |

### 1. Approaches to the Dataset

**Approach 1:Prediction of Abalone Age**  
Using physical measurements to predict the age through the number of
rings. This is the goal of the project, and would help marine scientists
avoid the difficult process of cutting the abolone and counting the
number of rings physically.

**Approach 2: Dimensionality Reduction for Prediction**  
Dimensionality reduction (e.g., PCA) can help reduce the number of
predictor variables while retaining most of the information. This
approach can help speed the performance of this model, and could group
similar features (such as different weights) together.

### 2. Cleaning Data

The data was imported using the Kaggle competition found
[here](https://www.kaggle.com/competitions/playground-series-s4e4/). The
data was relatively clean, and the following code was used to get it
ready for basic analysis:

``` r
df <- read_csv("abalone.csv",
               show_col_types = FALSE) %>%
  clean_names() %>%
  select(-c(id))

# for basic data understanding:
df %>%
  skim_without_charts()
```

|                                                  |            |
|:-------------------------------------------------|:-----------|
| Name                                             | Piped data |
| Number of rows                                   | 90615      |
| Number of columns                                | 9          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |            |
| Column type frequency:                           |            |
| character                                        | 1          |
| numeric                                          | 8          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |            |
| Group variables                                  | None       |

Data summary

**Variable type: character**

| skim_variable | n_missing | complete_rate | min | max | empty | n_unique | whitespace |
|:--------------|----------:|--------------:|----:|----:|------:|---------:|-----------:|
| sex           |         0 |             1 |   1 |   1 |     0 |        3 |          0 |

**Variable type: numeric**

| skim_variable  | n_missing | complete_rate | mean |   sd |   p0 |  p25 |  p50 |   p75 |  p100 |
|:---------------|----------:|--------------:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| length         |         0 |             1 | 0.52 | 0.12 | 0.07 | 0.44 | 0.54 |  0.60 |  0.81 |
| diameter       |         0 |             1 | 0.40 | 0.10 | 0.06 | 0.34 | 0.42 |  0.47 |  0.65 |
| height         |         0 |             1 | 0.14 | 0.04 | 0.00 | 0.11 | 0.14 |  0.16 |  1.13 |
| whole_weight   |         0 |             1 | 0.79 | 0.46 | 0.00 | 0.42 | 0.80 |  1.07 |  2.83 |
| whole_weight_1 |         0 |             1 | 0.34 | 0.20 | 0.00 | 0.18 | 0.33 |  0.46 |  1.49 |
| whole_weight_2 |         0 |             1 | 0.17 | 0.10 | 0.00 | 0.09 | 0.17 |  0.23 |  0.76 |
| shell_weight   |         0 |             1 | 0.23 | 0.13 | 0.00 | 0.12 | 0.22 |  0.30 |  1.00 |
| rings          |         0 |             1 | 9.70 | 3.18 | 1.00 | 8.00 | 9.00 | 11.00 | 29.00 |

### 3. Most Important Variables

To find the six most important variables, the first thing we will do is
plot a correlation plot between all the variables with the number of
rings (except ID). This will indicate the most important features for
predicting age of Abolone.

``` r
cor_matrix <- (df) %>%
  select_if(is.numeric) %>%
  cor()

ggcorrplot(cor_matrix, hc.order = TRUE, type = "lower", lab = TRUE)
```

<div class="figure" style="text-align: center">

<embed src="847_FinalProject_files/figure-gfm/unnamed-chunk-4-1.pdf" title="Correlation Matrix of Abalone Dataset" type="application/pdf" />
<p class="caption">
Correlation Matrix of Abalone Dataset
</p>

</div>

From the correlation plot, it seems the variables are highly correlated
to each other. It seems the whole_weight variables are least correlated
to the number of rings.

The top six variables would then include the following:  
- sex, whole_weight, diameter, shell_weight, height and rings.

These variables were removed for the following reason:  
- whole_weight_1 and whole_weight_2: high correlation to whole_weight
and shell_weight and low correlation with rings.  
- length: 0.99 correlation to diameter.

The ggpairs plot is then shown below:

``` r
df %>%
  select(sex, diameter, height, whole_weight, shell_weight, rings) %>%
  mutate(sex = as.factor(sex)) %>%  # Ensure 'Sex' is treated as a factor
  ggpairs(mapping = ggplot2::aes(color = sex),
          upper = list(continuous = wrap("density", size = 1, alpha = 0.4),
                       combo = wrap("box_no_facet", size = 1, alpha = 0.4)),
          lower = list(continuous = wrap("points", size = 1, alpha = 0.4),
                       combo = wrap("dot_no_facet", size = 1, alpha = 0.4)),
          diag = list(
      continuous = wrap("densityDiag", alpha = 0.4),
      discrete = wrap("barDiag", alpha = 0.4)
    ))
```

<div class="figure" style="text-align: center">

<embed src="847_FinalProject_files/figure-gfm/unnamed-chunk-5-1.pdf" title="Pair Plot of 6 Most Important Variables" type="application/pdf" />
<p class="caption">
Pair Plot of 6 Most Important Variables
</p>

</div>

``` r
abolone <- df %>%
  select(sex, diameter, height, whole_weight, shell_weight, rings)
```

The trends show that:  
- The ‘I’ sex shows significant deviation from the other M and F sex for
mostly all variables.  
- The height column show the most number of outliers.  
- The rings column does not show significant linear patterns which
matches the correlation plot previously, however the other variables do
seem correlated with one other linearly.

### 4. Classification Tree

In order to build a classification tree, we will split the rings column
into a level of maturity, where greater than 10 rings will be classified
as mature. Then we will develop our classification algorithm based on
the other 5 variables.

``` r
library(rpart)
library(caret)
library(rpart.plot)

df$mature <- ifelse(df$rings > 10, "mature", "immature")
df$mature <- as.factor(df$mature)


set.seed(11)  
train_index <- createDataPartition(df$mature, p = 0.80, list = FALSE)
training_data <- df[train_index, ]
testing_data <- df[-train_index, ]

# building tree
tree_model <- rpart(mature ~ sex + diameter + height + whole_weight + shell_weight, 
                    data = training_data, method = "class")
rpart.plot(tree_model, main="Classification Tree for Abalone Maturity", extra = 100)  
```

<embed src="847_FinalProject_files/figure-gfm/unnamed-chunk-6-1.pdf" style="display: block; margin: auto;" type="application/pdf" />

``` r
predictions <- predict(tree_model, testing_data, type = "class")
confusion_matrix <- confusionMatrix(predictions, testing_data$mature)

# predictions
example_data <- testing_data[1:2, ]
example_predictions <- predict(tree_model, example_data, type = "class")
```

``` r
#printing
kable(as.data.frame.matrix(confusion_matrix$table), caption = "Confusion Matrix for Abalone Maturity", align = 'c')
```

|          | immature | mature |
|:---------|:--------:|:------:|
| immature |  11322   |  2224  |
| mature   |   1388   |  3188  |

Confusion Matrix for Abalone Maturity

``` r
kable(data.frame(example_data[, c("sex", "diameter", "height", "whole_weight", "shell_weight", "mature")], Predicted = example_predictions), caption = "Prediction Table")
```

| sex | diameter | height | whole_weight | shell_weight | mature   | Predicted |
|:----|---------:|-------:|-------------:|-------------:|:---------|:----------|
| F   |    0.430 |   0.15 |       0.7715 |       0.2400 | mature   | immature  |
| I   |    0.425 |   0.13 |       0.7820 |       0.1975 | immature | immature  |

Prediction Table

#### Following down the tree:

1.  Shell Weight \< 0.25 -\> Immature  
2.  Shell Weight \< 0.25 -\> Immature

### 5. Continous Model

We will build a model to predict the rings column, from the other 5 most
important variables. We will use PCA to reduce the dimensions of the
dataset, and try to predict it using nonlinear terms as well!

``` r
ab_scaled <- abolone %>% 
  select(-c(sex, rings)) %>%
  scale()

pca_result  <- prcomp(ab_scaled, center = TRUE, scale. = TRUE)
summary(pca_result)
```

    ## Importance of components:
    ##                           PC1     PC2     PC3     PC4
    ## Standard deviation     1.9419 0.34537 0.27621 0.18239
    ## Proportion of Variance 0.9428 0.02982 0.01907 0.00832
    ## Cumulative Proportion  0.9428 0.97261 0.99168 1.00000

Since PC1 accounts for 94% of our variance, we can just use PC1 for our
new model.

``` r
abalone_pca <- data.frame(PC1 = pca_result$x[, 1], Rings = df$rings)
model <- lm(Rings ~ PC1 + I(PC1^2), data = abalone_pca)
kable(tidy(model), digits = c(1,2,5,1,40), caption = "Linear Regression Model Variables")
```

| term        | estimate | std.error | statistic | p.value |
|:------------|---------:|----------:|----------:|--------:|
| (Intercept) |     9.91 |   0.00991 |     999.6 |       0 |
| PC1         |    -1.09 |   0.00400 |    -273.4 |       0 |
| I(PC1^2)    |    -0.06 |   0.00164 |     -33.8 |       0 |

Linear Regression Model Variables

### 6. 3 Variable ggplot

``` r
ggplot(df, aes(x = whole_weight, y = rings, color = sex)) +
  geom_point(alpha = 0.5, size = 2) +  
  scale_color_manual(values = c("M" = "#00AFBB", "F" = "#E7B800", "I" = "#FC4E07")) +  
  labs(title = "Relationship between Whole Weight, Rings, and Sex",  
       x = "Whole Weight (grams)",
       y = "Rings",
       color = "Sex") +
  theme_minimal() +  
  theme(
    text = element_text(size = 14),  
    plot.title = element_text(hjust = 0.5),  
    panel.grid.minor = element_blank()  
  )
```

<div class="figure" style="text-align: center">

<embed src="847_FinalProject_files/figure-gfm/unnamed-chunk-10-1.pdf" title="Relationship between Whole Weight, Rings, and Sex" type="application/pdf" />
<p class="caption">
Relationship between Whole Weight, Rings, and Sex
</p>

</div>
