# Start coding here...
# revisando si los datos que necesito se encuentran en el directorio home
list.files()

# cargando los datos con la funcion correspondiente
data_recipe_row <- read.csv("recipe_site_traffic_2212.csv")
data_recipe <- data_recipe_row

# revisando la estructura de los datos
head(data_recipe)
cat("Cantidad de filas de la tabla", length(data_recipe$recipe), "\n")


#revisando la 1ra variable "recipe"
cat("La variable recipe es de clase:", class(data_recipe$recipe), "\n")

# revisando si tiene algun valor NA
cat("La variable recipe tiene algun valor NA:", any(is.na(data_recipe$recipe)), "\n")
#como se puede aprecir, esta variable no tiene ningun valor NA ya que es una variable que representa los ID de las recetas, por lo cual,
#no es posible NA aqui


# revisando la 2da variable "calories"
cat("La variable calories es de clase:", class(data_recipe$calories), "\n")
any(is.na(data_recipe$calories))    # revisando si tiene valores faltantes
sum(is.na(data_recipe$calories))    # revisando la cantidad de valores faltantes

# hallando el average de la variable "calories" sin tener en cuenta los valores NA
avg_calories <- round(mean(data_recipe$calories, na.rm = TRUE), 2)

# sustituyendo los valores NA por la media de la variable
data_recipe$calories[is.na(data_recipe$calories)] <- avg_calories

any(is.na(data_recipe$calories))  # revisando nuevamente si despues de la transformacion de los datos queda algun valor NA


# Revisando la 3ra variable "carbohidrate"
cat("La variable carbohydrate es de clase:", class(data_recipe$carbohydrate), "\n")
any(is.na(data_recipe$carbohydrate))    # revisando si tiene valores faltantes
sum(is.na(data_recipe$carbohydrate))    # revisando la cantidad de valores faltantes

# hallando el average de la variable "carbohidrate "sin tener en cuenta los valores NA
avg_carbohydrate <- round(mean(data_recipe$carbohydrate, na.rm = TRUE), 2)

# sustituyendo los valores NA por la media de la variable
data_recipe$carbohydrate[is.na(data_recipe$carbohydrate)] <- avg_carbohydrate
any(is.na(data_recipe$carbohydrate))  # revisando nuevamente si despues de la transformacion de los datos queda algun valor NA


# Revisando la 4ta variable "sugar"
cat("La variable sugar es de clase:", class(data_recipe$sugar), "\n")
any(is.na(data_recipe$sugar))    # revisando si tiene valores faltantes
sum(is.na(data_recipe$sugar))    # revisando la cantidad de valores faltantes

# hallando el average de la variable "sugar "sin tener en cuenta los valores NA
avg_sugar <- round(mean(data_recipe$sugar, na.rm = TRUE), 2)

# sustituyendo los valores NA por la media de la variable
data_recipe$sugar[is.na(data_recipe$sugar)] <- avg_sugar
any(is.na(data_recipe$sugar))  # revisando nuevamente si despues de la transformacion de los datos queda algun valor NA


# Revisando la 5ta variable "protein"
cat("La variable protein es de clase:", class(data_recipe$protein), "\n")
any(is.na(data_recipe$protein))    # revisando si tiene valores faltantes
sum(is.na(data_recipe$protein))    # revisando la cantidad de valores faltantes

# hallando el average de la variable "protein "sin tener en cuenta los valores NA
avg_protein <- round(mean(data_recipe$protein, na.rm = TRUE), 2)

# sustituyendo los valores NA por la media de la variable
data_recipe$protein[is.na(data_recipe$protein)] <- avg_protein
any(is.na(data_recipe$protein))  # revisando nuevamente si despues de la transformacion de los datos queda algun valor NA


# Revisando la 6ta variable "category"
cat("La variable category es de clase:", class(data_recipe$category), "\n")
any(is.na(data_recipe$category))    # revisando si la variable tiene valores faltantes
table(data_recipe$category)
length(table(data_recipe$category))

# estas son las categorias que debe de tener la variable de category
categorias <- c("Lunch/Snacks", "Beverages", "Potato", "Vegetable", "Meat", "Chicken", "Pork", "Dessert", "Breakfast", "One Dish Meal")

# revisando si algunos de los valores de category no estan entre los valores que deben estar en category
any(!names(table(data_recipe$category)) %in% categorias)

# revisando que valores no estan en entre los valores que deben estar
names(table(data_recipe$category))[!names(table(data_recipe$category)) %in% categorias]

# combinando Chicken y Chicken Breast en Chicken(uniendolos en 1 sola categoria)
if(!require(tidyverse)) install.packages("tidyverse")
library(tidyverse)
data_recipe <- data_recipe %>% mutate(category = ifelse(category == "Chicken Breast", "Chicken", category))

# revisando si despues del cambio existe alguna otra categoria indeseada
any(!names(table(data_recipe$category)) %in% categorias)


# revisando la 7ma variable "servings"
cat("La variable servings es de clase:", class(data_recipe$servings), "\n")

# revisando la compocicion de la variable
table(data_recipe$servings)

#realizando trasformaciones en la variable para llevarla a numerica
data_recipe <- data_recipe %>% mutate(servings = round(as.numeric(case_when(servings == "4 as a snack" ~ 4,
                                                                            servings == "6 as a snack" ~ 6,
                                                                            TRUE ~ as.numeric(servings)))))
any(is.na(data_recipe$servings))    # revisando si tiene valores faltantes


# revisando la 8va variable "high_traffic"
cat("La variable high_traffic es de clase:", class(data_recipe$high_traffic), "\n")

# revisando la compocicion de la variable "high_traffic"
table(data_recipe$high_traffic)
any(is.na(data_recipe$high_traffic))    # revisando si tiene valores faltantes
sum(is.na(data_recipe$high_traffic))    # revisando la cantidad de valores faltantes

#trasformando los datos a binarios orque es mas facil trabajar con ellos, donde 1 es high traffic and 0 is low traffic
data_recipe$high_traffic[!is.na(data_recipe$high_traffic)] <- 1
data_recipe$high_traffic[is.na(data_recipe$high_traffic)] <- 0
data_recipe$high_traffic <- as.numeric(data_recipe$high_traffic)



##################################
# Graficando las variables
##################################


if(!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)

# Histogram and density diagram for 'calories'
ggplot(data_recipe, aes(x = calories)) +
  geom_histogram(aes(y = ..density..), binwidth = 50, fill = "steelblue", color = "black", alpha = 0.7) +
  geom_density(alpha = 0.5, fill = "lightblue", color = "blue") +
  labs(title = "Calorie distribution by recipe",
       x = "Calories",
       y = "Density") +
  theme_minimal()

# Bars chart for 'category'
ggplot(data_recipe, aes(x = fct_rev(fct_infreq(category)))) + 
  geom_bar(fill = "coral", color = "black", alpha = 0.8) +
  coord_flip() + 
  labs(title = "Recipe frequency per category",
       x = "Categories",
       y = "Number of recipes") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 10))

# Distribution of categories according to their traffic
data_recipe %>% ggplot(aes(x = category, fill = as.factor(high_traffic))) +
  geom_bar() + scale_fill_manual(name = "Traffic Level",  
                                 labels = c("0" = "Low Traffic", "1" = "High Traffic"),
                                 values = c("0" = "#F8766D", "1" = "#00BFC4")) +
  labs(title = "Distribution of Traffic by Food Category", x = "Food Category", y = "Count") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10), 
                          plot.title = element_text(hjust = 0.5, face = "bold", size = 14), 
                          legend.title = element_text(face = "bold"),
                          panel.grid.major.x = element_blank(),
                          panel.grid.minor.y = element_blank())

data_recipe_row %>% select(category, high_traffic) %>% 
  mutate(high_traffic = as.numeric(ifelse(is.na(high_traffic), 0, 1))) %>% 
  filter(category %in% c("Chicken", "Chicken Breast")) %>% 
  ggplot(aes(category, fill = as.factor(high_traffic))) + geom_bar() + scale_fill_manual(name = "Traffic Level",  
                                                                                         labels = c("0" = "Low Traffic", "1" = "High Traffic"),
                                                                                         values = c("0" = "#F8766D", "1" = "#00BFC4")) +
  labs(title = "Distribution of Traffic by Chicken Category", x = "Chicken Category", y = "Count") +
  theme_minimal() + theme(axis.text.x = element_text(size = 10),
                          plot.title = element_text(hjust = 0.5, face = "bold", size = 14), 
                          legend.title = element_text(face = "bold"),
                          panel.grid.major.x = element_blank(),
                          panel.grid.minor.y = element_blank())

# Traffic according to the number of services
data_recipe %>% ggplot(aes(servings, fill = as.factor(high_traffic))) + geom_bar() + 
  scale_fill_manual(name = "Traffic Level", labels = c("0" = "Low Traffic", "1" = "High Traffic"),
                    values = c("0" = "#F8766D", "1" = "#00BFC4")) +
  labs(title = "Distribution of Traffic by Servings", x = "Servings", y = "Count") +
  theme_minimal() + theme(axis.text.x = element_text(size = 10),
                          plot.title = element_text(hjust = 0.5, face = "bold", size = 14), 
                          legend.title = element_text(face = "bold"),
                          panel.grid.major.x = element_blank(),
                          panel.grid.minor.y = element_blank())

# Distribution of sugar according to traffic in the different categories
data_recipe %>% ggplot(aes(category, calories, fill = as.factor(high_traffic))) + geom_boxplot() +
  scale_fill_manual(name = "Traffic Level", 
                    labels = c("0" = "Low Traffic", "1" = "High Traffic"),
                    values = c("0" = "#F8766D", "1" = "#00BFC4")) +
  labs(title = "Distribution of Traffic by Calories in the categories", x = "Food Category", y = "Calories") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10), 
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14), 
        legend.title = element_text(face = "bold"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_blank())


# Distribution of sugar according to traffic in the different categories
data_recipe %>% ggplot(aes(category, sugar, fill = as.factor(high_traffic))) + geom_boxplot() +
  scale_fill_manual(name = "Traffic Level", 
                    labels = c("0" = "Low Traffic", "1" = "High Traffic"),
                    values = c("0" = "#F8766D", "1" = "#00BFC4")) +
  labs(title = "Distribution of Traffic by Sugar in the categories", x = "Food Category", y = "Sugar") + 
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10), 
                                plot.title = element_text(hjust = 0.5, face = "bold", size = 14), 
                                                          legend.title = element_text(face = "bold"),
                                                          panel.grid.major.x = element_blank(),
                                                          panel.grid.minor.y = element_blank())

# Distribucion de las proteinas segun el trafico en las fiferentes categorias
data_recipe %>% ggplot(aes(category, protein, fill = as.factor(high_traffic))) + geom_boxplot() +
  scale_fill_manual(name = "Traffic Level", 
                    labels = c("0" = "Low Traffic", "1" = "High Traffic"),
                    values = c("0" = "#F8766D", "1" = "#00BFC4")) +
  labs(title = "Distribution of Traffic by Protein in the categories", x = "Food Category", y = "Protein") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, 
                                                     size = 10), 
                          plot.title = element_text(hjust = 0.5, face = "bold", size = 14), 
                          legend.title = element_text(face = "bold"),
                          panel.grid.major.x = element_blank(),
                          panel.grid.minor.y = element_blank())



# Proporcion de resetas que son de 4 servicios
data_recipe %>% filter(servings == 4, high_traffic == 1) %>% summarise(serving_high_traffic = length(servings) / 574)



###################################
# creando el modelo de linea base
###################################

if(!require(caret)) install.packages("caret")
library(caret)
set.seed(1)
# creando un indice para poder separar los datos en entrenamiento y prueba 
train_index <- createDataPartition(data_recipe$high_traffic, times = 1, p = 0.8, list = FALSE)
data_modif <- data_recipe %>% select(-recipe)

# usando el indice para crear el set de entrenamiento y el de prueba
train <- data_modif[train_index,]
test <- data_modif[-train_index,]

# creando el modelo de linea base
model <- lm(high_traffic ~ ., data = train)
summary(model)

# realizando predicciones con el modelo
predicciones <- predict(model, newdata = test)

# convirtiendo las predicciones en binarias
predict_high_trafic <- as.factor(ifelse(predicciones >= 0.5, 1, 0))

# creando una matriz de confucion para poder hallar la precicion del modelo
CM <- confusionMatrix(predict_high_trafic, as.factor(test$high_traffic))
CM


######################################################
# creando modelo para compararlo con el de linea base
######################################################
if(!require(pROC)) install.packages("pROC")
if(!require(MLmetrics)) install.packages("MLmetrics")
if(!require(glmnet)) install.packages("glmnet")

library(glmnet) 
library(pROC)    
library(MLmetrics)

# creando el control en el cual se va a basar el entrenamiento
fit_control <- trainControl(
  method = "repeatedcv",              # Validación cruzada repetida
  number = 10,                        # 10 folds
  repeats = 3,                        # Repetir la CV 3 veces
  summaryFunction = twoClassSummary,  # Para ROC, Sensitivity, Specificity
  classProbs = TRUE,                  # ¡Importante! Necesario para ROC y probabilidades
  verboseIter = TRUE,                 # Muestra el progreso
  allowParallel = TRUE                # Permite paralelización si tienes múltiples cores
)

# hiper parametros que se van a probar para determinar la mejor combinacion para el modelo
glmnet_grid <- expand.grid(alpha = seq(0, 1, length = 5),
                           lambda = 10^seq(-3, 0, length = 20))

# separando las caracteristicas predictivas de los objetivos para el set de entrenamiento y para el set de pueba
x_train <- train %>% select(-high_traffic)
y_train <- train %>% select(high_traffic) %>% pull() %>% as.factor()
levels(y_train) <- make.names(levels(y_train))

x_test <- test %>% select(-high_traffic)
y_test <- test %>% select(high_traffic) %>% pull() %>% as.factor()
levels(y_test) <- make.names(levels(y_test))

# creando varible dummy de los predictores para poder convertir la columna categiroa en 
dummy_obj <- dummyVars(~ ., data = x_train, fullRank = TRUE)

# Aplicar la transformación a x_train y x_test
x_train_processed <- predict(dummy_obj, newdata = x_train)
x_test_processed <- predict(dummy_obj, newdata = x_test)

# Convertir a data.frame si predict devuelve una matriz (opcional pero buena práctica)
x_train <- as.data.frame(x_train_processed)
x_test <- as.data.frame(x_test_processed)


# Utilizando la paralelizacion para aumentar la velocidad del entrenamiento
if(!require(parallel)) install.packages("parallel")
if(!require(doParallel)) install.packages("doParallel")
library(parallel)
library(doParallel)

# Detectar el número de núcleos disponibles
nucleos <- detectCores() - 1  # Usa todos los núcleos menos 1 para evitar congelar el sistema

# Configurar clúster de procesamiento paralelo
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# set a seed para poder reproducir el modelo
set.seed(42) 

# entrenando el modelo
glmnet_model <- train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  trControl = fit_control,
  tuneGrid = glmnet_grid,                      # Puedes usar una rejilla personalizada
  metric = "ROC",                              # Optimizar por ROC AUC
  preProcess = c("center", "scale")
)

# Detener el clúster después de entrenar
stopCluster(cl)

# esto es para tener una vista general de las metricas del modelo
print(glmnet_model)

# Muestra cómo ROC varía con alpha y lambda
plot(glmnet_model) 

# hacemos las predicciones de de los datos objetivos en el conjunto de evaluacion
predictions_glmnet_class <- predict(glmnet_model, newdata = x_test)

# debido a que comvertimos los objetivos en factor, debemos espesificar cual es la clase positiva
POSITIVE_CLASS <- "X1"

# calculamos la matriz de confucion
cm_glmnet <- confusionMatrix(data = predictions_glmnet_class, reference = y_test, positive = POSITIVE_CLASS)
print(cm_glmnet)

cat("La precicion del modelo glmnet es del", round(cm_glmnet$overall["Accuracy"]*100),"%","\n")
cat("La Sensitivity del modelo glmnet es del", round(cm_glmnet$byClass["Sensitivity"]*100),"%","\n")

####################################################################################
# entrenando el modelo con todo el set de datos entero, para mejorar su capacidad
####################################################################################

# uniendo los datos
full_train <- rbind(x_train, x_test)
full_target <- c(y_train, y_test)

# seting the control parametres
fit_control_final <- trainControl(
  method = "none",              # Validación cruzada repetida
  summaryFunction = twoClassSummary,  # Para ROC, Sensitivity, Specificity
  classProbs = TRUE,                  # ¡Importante! Necesario para ROC y probabilidades
  allowParallel = TRUE                # Permite paralelización si tienes múltiples cores
)
# obteniendo los mejores parametros
glmnet_grid_best <- expand.grid(alpha = glmnet_model$bestTune$alpha, lambda = glmnet_model$bestTune$lambda)

# set a seed para poder reproducir el modelo
set.seed(42) 

# entrenando el modelo
glmnet_model_final <- train(
  x = full_train,
  y = full_target,
  method = "glmnet",
  trControl = fit_control_final,
  tuneGrid = glmnet_grid_best,                      # Puedes usar una rejilla personalizada
  metric = "ROC",                                   # Optimizar por ROC AUC
  preProcess = c("center", "scale")
)

# esto es para tener una vista general de las metricas del modelo
print(glmnet_model_final)

