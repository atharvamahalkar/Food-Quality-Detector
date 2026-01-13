
# Food Quality Detector

## Description

The Food Quality Detector is a cutting-edge, web-based application aimed at enhancing customer trust and satisfaction within the hyperlocal food delivery market. Utilizing a Convolutional Neural Network (CNN) model built with TensorFlow, this AI-powered tool efficiently addresses food quality complaints by allowing users to upload images of their food for real-time quality assessment. This initiative not only streamlines the complaint resolution process but also provides valuable feedback to partner restaurants, ultimately reducing churn and fostering a more trustworthy service environment. The frontend is developed using React, offering a user-friendly interface, while the backend is powered by Django REST Framework to handle image processing and prediction.

## Features

-   Image upload functionality for food quality analysis.
-   Real-time image preview before submission.
-   Deep Learning model for accurate classification of food quality.
-   Responsive web interface for a seamless experience across various devices.

## Output
![Screenshot 2024-02-24 at 4 30 14 AM](https://github.com/adil200/Food-Quality-Detector/assets/75264739/fd25ac6a-5a58-47d0-9448-f06c88e5155a)
![Screenshot 2024-02-24 at 4 30 31 AM](https://github.com/adil200/Food-Quality-Detector/assets/75264739/6ab03a84-ed46-4225-9245-70a208f78a91)
![Screenshot 2024-02-24 at 4 30 38 AM](https://github.com/adil200/Food-Quality-Detector/assets/75264739/889880fb-6092-4ef0-b92c-08080ad0fb14)
![Screenshot 2024-02-24 at 4 30 52 AM](https://github.com/adil200/Food-Quality-Detector/assets/75264739/0db45e45-bc9a-4dcc-ba3f-4f800f0f708b)

## Project Context
This project was conceptualized to address a critical need in the hyperlocal food delivery sector—maintaining high customer satisfaction by ensuring food quality. The Automated Food Quality Control system encapsulates a seamless process for reporting and resolving quality issues, leveraging AI to automate the assessment of food quality from customer-uploaded images. By providing a direct channel for customer feedback and actionable insights to partner restaurants, the project aims to enhance the overall food delivery experience, ensuring that FundsRoom stands out in a competitive landscape.

## Warning Note
Please note that due to the limitations in the available dataset size and the computational resources, the accuracy of the Food Quality Detector's predictions may currently be below optimal levels. For improved performance and more accurate predictions, access to a larger dataset and enhanced computational power would be necessary. This is an area of potential enhancement as the project scales and gains additional resources.

## Installation

### Prerequisites

-   Python 3.8 or higher
-   Node.js 12.x or higher
-   npm (Node Package Manager)

### Model File Download

Before setting up the backend, download the pre-trained model file (`model.h5`) from Google Drive as the model file exceed the Github file size limit:

1.  Go to [this Google Drive link](https://drive.google.com/file/d/1GR_QzWme_pT7Kaxb9dQRiWqpDZ-s7o8T/view?usp=sharing).
2.  Download the `model.h5` file.
3.  Place the downloaded file into the `backend/backend` directory of your project.

### Backend Setup

1.  Clone the repository and navigate to the backend directory.
2.  Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate` 
``` 

3.  Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

4.  Navigate to the backend directory and run the Django server:

```bash
python manage.py runserver
```

### Frontend Setup

1.  Navigate to the frontend directory.
2.  Install the required npm packages:

```bash
npm install
```

3.  Start the React development server:

```bash
npm start
```

The application will be available at `http://localhost:3000/`.

## Usage

1.  Open the web application in your browser.
2.  Click on "Choose File" to upload an image of the food item.
3.  Click on "Submit" to send the image for quality prediction.
4.  The prediction result will be displayed below the upload button.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   Special thanks to [Fresh and Rotten Classification dataset](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification) from Kaggle for providing the dataset used to train the model.
- This project is developed as part of the selection process for the FundsRoom interview round, aiming to showcase innovative solutions in automated food quality control for the hyperlocal business model.

----------
