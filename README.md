# Project-
{
  "metadata": {
    "kernelspec": {
      "language": "python",
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.11.11",
      "mimetype": "text/x-python",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "file_extension": ".py"
    },
    "kaggle": {
      "accelerator": "none",
      "dataSources": [
        {
          "sourceId": 11124361,
          "sourceType": "datasetVersion",
          "datasetId": 6937364
        }
      ],
      "dockerImageVersionId": 31012,
      "isInternetEnabled": true,
      "language": "python",
      "sourceType": "notebook",
      "isGpuEnabled": false
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat_minor": 0,
  "nbformat": 4,
  "cells": [
    {
      "source": [
        "# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,\n",
        "# THEN FEEL FREE TO DELETE THIS CELL.\n",
        "# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON\n",
        "# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR\n",
        "# NOTEBOOK.\n",
        "import kagglehub\n",
        "khushikyad001_india_road_accident_dataset_predictive_analysis_path = kagglehub.dataset_download('khushikyad001/india-road-accident-dataset-predictive-analysis')\n",
        "\n",
        "print('Data source import complete.')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "avpTFr5sxmLi",
        "outputId": "1663aa0d-ba66-4ac7-e220-62755b26b032"
      },
      "cell_type": "code",
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading from https://www.kaggle.com/api/v1/datasets/download/khushikyad001/india-road-accident-dataset-predictive-analysis?dataset_version_number=1...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 68.2k/68.2k [00:00<00:00, 41.0MB/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Extracting files...\n",
            "Data source import complete.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Table of Content\n",
        "\n",
        "1) Importing Libraries\n",
        "\n",
        "2) Loading the Data\n",
        "\n",
        "3) Cleaning the Data\n",
        "\n",
        "4) Date related Analysis\n",
        "\n",
        "5) Categorical Analysis\n",
        "\n",
        "6) Numerical Analysis\n",
        "\n",
        "7) Conclusion: Indian Road Accident Insights"
      ],
      "metadata": {
        "id": "ha4G7X-7xmLi"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Importing Libraries"
      ],
      "metadata": {
        "id": "TmNcHDJKxmLi"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-04-17T13:50:14.759037Z",
          "iopub.execute_input": "2025-04-17T13:50:14.759965Z",
          "iopub.status.idle": "2025-04-17T13:50:14.765345Z",
          "shell.execute_reply.started": "2025-04-17T13:50:14.759912Z",
          "shell.execute_reply": "2025-04-17T13:50:14.764242Z"
        },
        "id": "DXT_D3fmxmLj"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "data=pd.read_csv('/content/accident_prediction_india.csv')\n",
        "data.info()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-04-17T13:50:14.767126Z",
          "iopub.execute_input": "2025-04-17T13:50:14.767455Z",
          "iopub.status.idle": "2025-04-17T13:50:14.883362Z",
          "shell.execute_reply.started": "2025-04-17T13:50:14.767426Z",
          "shell.execute_reply": "2025-04-17T13:50:14.882343Z"
        },
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Wf2nxgFbxmLj",
        "outputId": "8dd2c167-9af5-46bb-dd98-cbfac322abca"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 3000 entries, 0 to 2999\n",
            "Data columns (total 22 columns):\n",
            " #   Column                       Non-Null Count  Dtype \n",
            "---  ------                       --------------  ----- \n",
            " 0   State Name                   3000 non-null   object\n",
            " 1   City Name                    3000 non-null   object\n",
            " 2   Year                         3000 non-null   int64 \n",
            " 3   Month                        3000 non-null   object\n",
            " 4   Day of Week                  3000 non-null   object\n",
            " 5   Time of Day                  3000 non-null   object\n",
            " 6   Accident Severity            3000 non-null   object\n",
            " 7   Number of Vehicles Involved  3000 non-null   int64 \n",
            " 8   Vehicle Type Involved        3000 non-null   object\n",
            " 9   Number of Casualties         3000 non-null   int64 \n",
            " 10  Number of Fatalities         3000 non-null   int64 \n",
            " 11  Weather Conditions           3000 non-null   object\n",
            " 12  Road Type                    3000 non-null   object\n",
            " 13  Road Condition               3000 non-null   object\n",
            " 14  Lighting Conditions          3000 non-null   object\n",
            " 15  Traffic Control Presence     2284 non-null   object\n",
            " 16  Speed Limit (km/h)           3000 non-null   int64 \n",
            " 17  Driver Age                   3000 non-null   int64 \n",
            " 18  Driver Gender                3000 non-null   object\n",
            " 19  Driver License Status        2025 non-null   object\n",
            " 20  Alcohol Involvement          3000 non-null   object\n",
            " 21  Accident Location Details    3000 non-null   object\n",
            "dtypes: int64(6), object(16)\n",
            "memory usage: 515.8+ KB\n"
          ]
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Cleaning the Data"
      ],
      "metadata": {
        "id": "Fa25Ot-OxmLj"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Check for null values:"
      ],
      "metadata": {
        "id": "sAtclp6UxmLj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#First it's better to take a copy of dataset and work on a copy\n",
        "df=data.copy()\n",
        "\n",
        "#Checking null values accross the dataset\n",
        "df.isnull().sum().sort_values(ascending=False)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-04-17T13:50:14.884455Z",
          "iopub.execute_input": "2025-04-17T13:50:14.884757Z",
          "iopub.status.idle": "2025-04-17T13:50:14.903152Z",
          "shell.execute_reply.started": "2025-04-17T13:50:14.884728Z",
          "shell.execute_reply": "2025-04-17T13:50:14.902059Z"
        },
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 774
        },
        "id": "WA_5VbyMxmLj",
        "outputId": "3a09b40e-4bc5-4fd8-9f02-43cd45027481"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Driver License Status          975\n",
              "Traffic Control Presence       716\n",
              "Year                             0\n",
              "Month                            0\n",
              "State Name                       0\n",
              "City Name                        0\n",
              "Time of Day                      0\n",
              "Day of Week                      0\n",
              "Accident Severity                0\n",
              "Number of Vehicles Involved      0\n",
              "Number of Fatalities             0\n",
              "Weather Conditions               0\n",
              "Vehicle Type Involved            0\n",
              "Number of Casualties             0\n",
              "Road Condition                   0\n",
              "Road Type                        0\n",
              "Speed Limit (km/h)               0\n",
              "Lighting Conditions              0\n",
              "Driver Age                       0\n",
              "Driver Gender                    0\n",
              "Alcohol Involvement              0\n",
              "Accident Location Details        0\n",
              "dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Driver License Status</th>\n",
              "      <td>975</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Traffic Control Presence</th>\n",
              "      <td>716</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Year</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Month</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>State Name</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>City Name</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Time of Day</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Day of Week</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Accident Severity</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Number of Vehicles Involved</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Number of Fatalities</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Weather Conditions</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Vehicle Type Involved</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Number of Casualties</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Road Condition</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Road Type</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Speed Limit (km/h)</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Lighting Conditions</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Driver Age</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Driver Gender</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Alcohol Involvement</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Accident Location Details</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "df['Driver License Status'].value_counts()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-04-17T13:50:14.905201Z",
          "iopub.execute_input": "2025-04-17T13:50:14.905472Z",
          "iopub.status.idle": "2025-04-17T13:50:14.920599Z",
          "shell.execute_reply.started": "2025-04-17T13:50:14.905446Z",
          "shell.execute_reply": "2025-04-17T13:50:14.919538Z"
        },
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 178
        },
        "id": "GkRoWBqVxmLj",
        "outputId": "7fefce19-2ee2-4851-813d-df35d43d1a7c"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Driver License Status\n",
              "Valid      1057\n",
              "Expired     968\n",
              "Name: count, dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Driver License Status</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Valid</th>\n",
              "      <td>1057</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Expired</th>\n",
              "      <td>968</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "df['Traffic Control Presence'].value_counts()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-04-17T13:50:14.922425Z",
          "iopub.execute_input": "2025-04-17T13:50:14.923307Z",
          "iopub.status.idle": "2025-04-17T13:50:14.94471Z",
          "shell.execute_reply.started": "2025-04-17T13:50:14.923212Z",
          "shell.execute_reply": "2025-04-17T13:50:14.943528Z"
        },
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 209
        },
        "id": "N_A7t9I7xmLj",
        "outputId": "8a6b3519-94af-434c-9d98-611a48d66c62"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Traffic Control Presence\n",
              "Signs               812\n",
              "Signals             736\n",
              "Police Checkpost    736\n",
              "Name: count, dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Traffic Control Presence</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Signs</th>\n",
              "      <td>812</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Signals</th>\n",
              "      <td>736</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Police Checkpost</th>\n",
              "      <td>736</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "df['Driver License Status']=df['Driver License Status'].fillna('Unknown')\n",
        "df['Traffic Control Presence']=df['Traffic Control Presence'].fillna('Unknown')"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-04-17T13:50:14.946145Z",
          "iopub.execute_input": "2025-04-17T13:50:14.946517Z",
          "iopub.status.idle": "2025-04-17T13:50:14.962417Z",
          "shell.execute_reply.started": "2025-04-17T13:50:14.946485Z",
          "shell.execute_reply": "2025-04-17T13:50:14.961281Z"
        },
        "id": "7icd4WOYxmLj"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "Check for duplicated rows:"
      ],
      "metadata": {
        "id": "3NplcBjjxmLk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.duplicated().sum()"
      ],
