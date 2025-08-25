##Flower Species Predictor

A basic machine learning web application built as part of my learning journey. This simple app predicts iris flower species based on four physical measurements.

**Try the app here:** [https://iris-vision.streamlit.app/](https://iris-vision.streamlit.app/)

##  About

This is a **basic learning project** that demonstrates fundamental machine learning concepts. The app simply takes four flower measurements as input and predicts which of the three iris species the flower belongs to:
- **Iris Setosa**
- **Iris Versicolor** 
- **Iris Virginica**

This project was created to practice:
- Building machine learning models
- Creating web applications with Streamlit
- Deploying apps to the cloud

## 🛠️ Built With

- **Python** - Programming language
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **Pickle** - Model serialization

##  What This App Does

This is a **simple, beginner-friendly project** that:

- **Takes 4 inputs**: Petal length, petal width, sepal length, sepal width
- **Makes a prediction**: Uses machine learning to classify the flower
- **Shows the result**: Displays one of three possible iris species

## Dataset

This project uses the famous **Fisher's Iris Dataset**, which contains:
- 3 species (50 samples each)
- 4 features per sample
- Perfect dataset for classification problems

##  How to Use

1. **Visit the app** at [iris-vision.streamlit.app](https://iris-vision.streamlit.app/)
2. **Enter measurements** for all four flower characteristics:
   - Petal Length (1.0 - 6.9 cm)
   - Petal Width (0.1 - 2.5 cm)
   - Sepal Length (4.3 - 7.9 cm)
   - Sepal Width (2.0 - 4.4 cm)
3. **Click "Predict species"** to get the classification result
4. **View the prediction** - the app will display one of the three iris species

##  Accuracy

The model achieves high accuracy on the iris dataset, which is known for being a well-separated, linearly separable dataset perfect for classification tasks.

## Deployment

This application is deployed on **Streamlit Cloud** and automatically updates when changes are pushed to the main branch.

