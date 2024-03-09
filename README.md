# Search Relevancy Ranking Model
Just a simple try to implement the model which finds out the probability of click for each search_id

## Authors
- **SONAI** - [SONAI](https://github.com/SONAIII)

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- Docker: Follow the [official Docker installation guide](https://docs.docker.com/get-docker/) for your operating system.

### Installing

1. **Clone the Repository**
   ```sh
   git clone https://github.com/SONAIII/SearchRelevancyPredictionModel.git
   ```
   
2. **Build the Docker Image**
   ```sh
   cd SearchRelevancyPredictionModel
   docker build -t search_rel_model .
   ```

3. **Run the Docker Container**
   ```sh
   docker run -p 4000:80 search_rel_model
   ```

## Built With
- Numpy, Pandas, Matplotlib
- [Scikit-Learn](https://scikit-learn.org/stable/) - Machine learning library for Python.
- [Docker](https://www.docker.com/) - Containerization platform.


