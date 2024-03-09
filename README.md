# Search Relevancy Ranking Model

Just a simple try to implement the model which finds out the probability of click for each search_id

## Authors

- **SONAI** - [SONAI](https://github.com/SONAIII)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Docker: Follow the [official Docker installation guide](https://docs.docker.com/get-docker/) for your operating system.

### Installing

A step-by-step series of examples that tell you how to get a development environment running.

1. **Clone the Repository**

   First, clone this repository to your local machine:

   ```sh
   git clone https://github.com/SONAIII/SearchRelevancyPredictionModel.git
   ```

2. **Build the Docker Image**

   Navigate to the directory containing the Dockerfile and build the Docker image:

   ```sh
   cd SearchRelevancyPredictionModel
   docker build -t SearchRelevancyPredictionModel .
   ```


3. **Run the Docker Container**

   After the image is built, you can run it:

   ```sh
   docker run -p 4000:80 SearchRelevancyPredictionModel
   ```

   This command runs the Docker container and maps port 80 inside the container to port 4000 on your host machine.


## Built With

List all the major frameworks/libraries used to bootstrap your project:

- Numpy, Pandas, Matplotlib
- [Scikit-Learn](https://scikit-learn.org/stable/) - Machine learning library for Python.
- [Docker](https://www.docker.com/) - Containerization platform.


