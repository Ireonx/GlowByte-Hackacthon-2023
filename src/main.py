import pickle
import sys
import warnings
from os import path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Ignore performance warnings from pandas
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Set option to display all columns in pandas
pd.set_option("display.max_columns", None)


class Application:
    def __init__(self):
        self._file_path: str = ""
        self._dataset: pd.DataFrame = pd.DataFrame()
        self._copy_dataset: pd.DataFrame = pd.DataFrame()

    def write_csv(self) -> None:
        """
        Saves the dataset to a CSV file.

        Prompts the user to specify a file path, or it defaults to 'new_dataset.csv'.

        Args:
            None

        Returns:
            None
        """
        file_path_: str = input("Specify the directory path where you want to save the file. "
                                "By default, the file is saved in the dataset directory with the name "
                                "'new_dataset.csv'\n")
        if not file_path_ or not file_path_.endswith(".csv"):
            file_path_ = "../datasets/new_dataset.csv"
        self._dataset.to_csv(file_path_, index=False)

    def plot_graph(self) -> None:
        """
        Plots a graph of date vs. target.

        Args:
            None

        Returns:
            None
        """
        plt.figure(figsize=(20, 14))
        plt.plot(self._dataset["date"], self._dataset["target"], marker="o", color="#FF0055")
        plt.xlabel("Date")
        plt.ylabel("Target")
        plt.title("Test Graph Date vs Target")
        plt.legend(["Target"], loc="upper right")

        ax: plt.Axes = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()

    def fill_with_zeros(self) -> None:
        """
        Fills missing values in 'temp_pred_x' and 'weather_pred_x' columns with corresponding values.

        Args:
            None

        Returns:
            None
        """
        self._dataset.loc[self._dataset["temp_pred_x"].isna(), "temp_pred_x"] = \
            self._dataset.loc[self._dataset["temp_pred_x"].isna(), "temp_pred_y"]

        self._dataset.loc[self._dataset["weather_pred_x"].isna(), "weather_pred_x"] = \
            self._dataset.loc[self._dataset["weather_pred_x"].isna(), "weather_pred_y"]

        self._dataset.loc[self._dataset["weather_pred_x"] == "0", "temp_pred_x"] = \
            self._dataset.loc[self._dataset["weather_pred_x"] == "0", "temp_pred_y"]

        self._dataset.loc[self._dataset["weather_pred_x"] == "0", "weather_pred_x"] = \
            self._dataset.loc[self._dataset["weather_pred_x"] == "0", "weather_pred_y"]

        self._dataset.loc[self._dataset["weather_pred_x"] == "?", "weather_pred_x"] = \
            self._dataset.loc[self._dataset["weather_pred_x"] == "?", "weather_pred_y"]

    def add_weather_type(self) -> None:
        """
        Adds columns 'fallout', 'wind', 'sun', and 'clouds' to the dataset.

        Args:
            None

        Returns:
            None
        """
        self._dataset["fallout"] = 0
        self._dataset["wind"] = 0
        self._dataset["sun"] = 0
        self._dataset["clouds"] = 0

    def update_weather_type(self) -> None:
        """
        Updates weather type and sets corresponding flags based on keywords.

        Args:
            None

        Returns:
            None
        """
        self._dataset.loc[self._dataset['weather_pred'].str.contains(r'вет|шторм'), 'wind'] = 1
        self._dataset.loc[self._dataset['weather_pred'].str.contains(r'ясн|солн'), 'sun'] = 1
        self._dataset.loc[self._dataset['weather_pred'].str.contains(r'гроз|дож|лив|шторм|сн|%'), 'fallout'] = 1
        self._dataset.loc[self._dataset['weather_pred'].str.contains(r'пасм|об'), 'clouds'] = 1

        self._dataset.loc[self._dataset['weather_pred'].str.contains(r'вет|шторм'), 'weather_pred'] = 'Del'
        self._dataset.loc[self._dataset['weather_pred'].str.contains(r'пасм|об'), 'weather_pred'] = 'Del'
        self._dataset.loc[self._dataset['weather_pred'].str.contains(r'ясн|солн'), 'weather_pred'] = 'Del'
        self._dataset.loc[self._dataset['weather_pred'].str.contains(r'гроз|дож|лив|шторм|сн'), 'weather_pred'] = 'Del'

    def drop_useless_fields(self) -> None:
        """
        Removes unnecessary columns and renames some columns for clarity.

        Args:
            None

        Returns:
            None
        """
        self._dataset.drop(["weather_pred_y", "temp_pred_y", "weather_fact"], axis=1, inplace=True)
        self._dataset.set_index("date_time", inplace=True)
        self._dataset.rename(columns={"weather_pred_x": "weather_pred",
                                      "temp_pred_x": "temp_pred"}, inplace=True)

    def add_time_sc(self):
        """
        Adds sine and cosine transformations of the 'time' column to capture time of day information.

        Args:
            None

        Returns:
            None
        """
        self._dataset['hour_sin'] = np.sin(2 * np.pi * self._dataset['time'] / 24.0)
        self._dataset['hour_cos'] = np.cos(2 * np.pi * self._dataset['time'] / 24.0)
        self._dataset.drop('time', axis=1, inplace=True)
        self._dataset.drop('weather_pred', axis=1, inplace=True)

    def make_features(self, start_lag: int, max_lag: int) -> pd.DataFrame:
        """
        Creates lag features for time series forecasting.

        Args:
            start_lag (int): The starting lag value.
            max_lag (int): The maximum lag value.

        Returns:
            pd.DataFrame: DataFrame with lag features.
        """
        df_copied: pd.DataFrame = self._dataset.copy()

        for lag in range(start_lag, max_lag + 1):
            df_copied["lag_{}".format(lag)] = df_copied["target"].shift(lag)
            df_copied["temp_lag_{}".format(lag)] = df_copied["temp"].shift(lag)

        df_copied.dropna(inplace=True)
        return df_copied

    def merge_with_temp_weather(self) -> None:
        """
        Merges the dataset with temperature and weather data from external CSV files.

        Args:
            None

        Returns:
            None
        """
        weather: pd.DataFrame = pd.read_csv("../datasets/weather.csv")
        temp: pd.DataFrame = pd.read_csv("../datasets/temp.csv")

        merge_list = [temp, weather]
        for j in merge_list:
            self._dataset = self._dataset.merge(
                how="left",
                right=j,
                left_on=["time", "month", "day"],
                right_on=["time", "month", "day"])

    def run(self) -> None:
        """
        Executes the data processing pipeline, including data loading, feature engineering, and machine learning prediction.

        Args:
            None

        Returns:
            None
        """
        self._file_path = self.get_path()
        self._dataset = self.load_dataframe(self._file_path)
        self._dataset = self.add_feature_dwm()
        self.merge_with_temp_weather()

        self.fill_with_zeros()

        self.drop_useless_fields()

        self.add_weather_type()
        self.update_weather_type()

        self.add_time_sc()

        self._copy_dataset: pd.DataFrame = self.make_features(25, 96)
        self._copy_dataset.drop("temp", axis=1, inplace=True)

        x_train: pd.DataFrame = self._copy_dataset.drop('target', axis=1)

        pickled_model = pickle.load(open('model.pkl', 'rb'))
        predict_ = [0 for _ in range(x_train.shape[0])]

        self._copy_dataset['prediction'] = predict_
        result_dataset = self._copy_dataset.groupby(['month', 'day']).agg('sum')

        print(f"MAE Value: {mean_absolute_error(self._copy_dataset['target'], predict_)}")

        print('Training dataset dimensions:', x_train.shape)

    def add_feature_dwm(self) -> pd.DataFrame:
        """
        Adds additional time-related features 'month', 'day', and 'weekday' to the dataset.

        Args:
            None

        Returns:
            pd.DataFrame: DataFrame with added features.
        """
        temp_datasets: pd.DataFrame = self._dataset
        temp_datasets["date"] = pd.to_datetime(temp_datasets["date"])
        temp_datasets["date_time"] = temp_datasets["date"]
        temp_datasets.set_index("date", inplace=True)
        temp_datasets["month"] = temp_datasets.index.month
        temp_datasets["day"] = temp_datasets.index.day
        temp_datasets["weekday"] = temp_datasets.index.dayofweek

        return temp_datasets

    @staticmethod
    def get_path() -> str:
        """
        Retrieves the file path to the dataset from the command line arguments.

        Args:
            None

        Returns:
            str: The file path to the dataset.
        """
        if len(sys.argv) == 2:
            file_path: str = sys.argv[-1]
            if path.isabs(file_path):
                file_path = path.normpath(file_path)
                if not file_path.endswith(".csv"):
                    print("Select a file with a .csv extension next time")
                    sys.exit(1)
                elif path.isfile(file_path):
                    print("Success!! File exists\n")
                    return file_path
                else:
                    print(f"Error!! File {file_path} not found\n")
                    sys.exit(1)
            else:
                print("You entered a non-absolute path")
                sys.exit(1)
        else:
            print("Too many/few parameters. Specify only the path relative to the root next time")
            sys.exit(1)

    @staticmethod
    def load_dataframe(file_path: str) -> pd.DataFrame:
        """
        Loads the dataset from the specified CSV file and returns it.

        Args:
            file_path (str): The file path to the CSV dataset.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        current_data: pd.DataFrame = pd.read_csv(file_path)
        return current_data

    @staticmethod
    def description(ax: plt.Axes, x_label: str, y_label: str, title: str) -> None:
        """
        Sets labels and titles for a given matplotlib axis.

        Args:
            ax (plt.Axes): The matplotlib axis.
            x_label (str): The x-axis label.
            y_label (str): The y-axis label.
            title (str): The plot title.

        Returns:
            None
        """

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)


if __name__ == "__main__":
    app: Application = Application()
    app.run()
