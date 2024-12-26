
# Formula 1 World Championship Prediction

This project aims to predict the Formula 1 World Championship winner using machine learning techniques. The application is built using Python, Streamlit for the web interface, and various machine learning libraries for data processing and model training.

## Features

- Data Loading and Preprocessing: Load and preprocess the data to prepare it for model training
- Model Training: Train a Random Forest classifier to predict the championship winner.
- Prediction: Predict the championship winner based on user input.
- Visualization: Visualize feature importance and driver performance.


## Functions
- load_data()
    : Loads the race data and merged dataset.

- filter_final_race_data(races_data, merged_data)
    : Filters the merged dataset to include only the last races of each season.

- preprocess_data(data)
    : Preprocesses the data by converting race positions to numeric, calculating wins, weighted wins, average positions, podiums, and normalized points.

- train_model(data)
    : Trains a Random Forest classifier using the preprocessed data and handles class imbalance using SMOTE.

- predict_championship(model)
    : Predicts the championship winner based on user input.

- plot_feature_importance(model, feature_names)
    : Plots the feature importance of the trained model.

- plot_driver_performance(data)
    : Plots driver performance against qualification.
## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.



## License

[MIT](https://choosealicense.com/licenses/mit/)


## Acknowledgements

 - [streamlit](https://streamlit.io)
 - [scikit-learn](https://scikit-learn.org/stable/)
 - [imbalanced-learn](https://imbalanced-learn.org/stable/)
 - [plotly](https://plotly.com)

